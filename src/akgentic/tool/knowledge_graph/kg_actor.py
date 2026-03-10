"""KnowledgeGraph actor with batch CRUD operations.

Mirrors the PlanActor pattern: Akgent subclass with typed state,
error-collection semantics, and ``BaseState.notify_state_change()``
for orchestrator integration.

Source: V2 redesign of akgentic-framework KnowledgeGraphManager.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from pydantic import Field

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.tool.errors import RetriableError
from akgentic.tool.knowledge_graph.models import (
    Entity,
    EntityCreate,
    EntityUpdate,
    GetGraphQuery,
    GraphView,
    KnowledgeGraphState,
    ManageGraph,
    Relation,
    RelationCreate,
    RelationDelete,
    SearchHit,
    SearchQuery,
    SearchResult,
    VectorEntry,
)
from akgentic.tool.knowledge_graph.vector_index import EmbeddingService, VectorIndex

logger = logging.getLogger(__name__)

KG_ACTOR_NAME: str = "#KnowledgeGraphTool"
"""Singleton actor name registered with the orchestrator."""

KG_ACTOR_ROLE: str = "ToolActor"
"""Actor role constant for ToolCard integration."""


class KnowledgeGraphConfig(BaseConfig):
    """Configuration for ``KnowledgeGraphActor`` with embedding provider settings.

    Extends ``BaseConfig`` with the embedding model and provider fields so
    the actor can initialize ``EmbeddingService`` from its config object.
    """

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier",
    )
    embedding_provider: Literal["openai", "azure"] = Field(
        default="openai",
        description="Which OpenAI SDK variant to use for embeddings",
    )


class KnowledgeGraphActor(Akgent[KnowledgeGraphConfig, KnowledgeGraphState]):
    """Centralized, thread-safe knowledge graph with batch CRUD.

    All mutations flow through ``update_graph(ManageGraph)`` which
    executes operations in a deterministic order, collects all errors,
    notifies the orchestrator via ``state.notify_state_change()``, and
    raises a single ``RetriableError`` when any errors occurred.

    Operation order:
        create_entities → create_relations → update_entities
        → delete_relations → delete_entities
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:  # noqa: ANN201
        """Initialize state, attach observer, and set up the vector index."""
        self.state = KnowledgeGraphState()
        self.state.observer(self)
        self._vector_index = VectorIndex()
        self._embedding_svc: EmbeddingService | None = None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_or_create_embedding_svc(self) -> EmbeddingService | None:
        """Return the ``EmbeddingService``, creating it lazily on first call.

        Returns:
            The service instance, or ``None`` if creation fails (e.g. missing deps).
        """
        if self._embedding_svc is None:
            try:
                self._embedding_svc = EmbeddingService(
                    model=self.config.embedding_model,
                    provider=self.config.embedding_provider,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Failed to initialize EmbeddingService: %s", self.config.name, exc
                )
                return None
        return self._embedding_svc

    def _embed_entity(self, entity: Entity) -> None:
        """Embed an entity and store the result in the vector index.

        Silently logs a WARNING and returns if the embedding service is
        unavailable or if the API call fails (AC-7).
        """
        svc = self._get_or_create_embedding_svc()
        if svc is None:
            return
        try:
            text = f"{entity.name}: {entity.description}"
            vectors = svc.embed([text])
            self._vector_index.add(
                VectorEntry(
                    ref_type="entity",
                    ref_id=str(entity.id),
                    text=text,
                    vector=vectors[0],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to embed entity '%s': %s", self.config.name, entity.name, exc
            )

    def _embed_relation(self, relation: Relation) -> None:
        """Embed a relation if it has a non-empty description.

        Relations without descriptions are not embedded (no meaningful text).
        Silently logs WARNING on embedding failure (AC-7).
        """
        if not relation.description:
            return
        svc = self._get_or_create_embedding_svc()
        if svc is None:
            return
        try:
            text = (
                f"{relation.from_entity} {relation.relation_type} "
                f"{relation.to_entity}: {relation.description}"
            )
            vectors = svc.embed([text])
            self._vector_index.add(
                VectorEntry(
                    ref_type="relation",
                    ref_id=str(relation.id),
                    text=text,
                    vector=vectors[0],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to embed relation '%s→%s': %s",
                self.config.name,
                relation.from_entity,
                relation.to_entity,
                exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _graph(self) -> KnowledgeGraphState:
        """Shortcut to the knowledge graph container inside state."""
        return self.state

    def _entity_map(self) -> dict[str, Entity]:
        """Build a name → Entity lookup from current graph."""
        return {e.name: e for e in self.state.knowledge_graph.entities}

    # ------------------------------------------------------------------
    # CRUD: Create
    # ------------------------------------------------------------------

    def _create_entities(self, entities: list[EntityCreate]) -> list[str]:
        """Add new entities, skipping duplicates by name.

        Returns:
            List of error strings for duplicate names already in graph.
        """
        errors: list[str] = []
        existing_names = {e.name for e in self.state.knowledge_graph.entities}

        for ec in entities:
            if ec.name in existing_names:
                errors.append(f"Entity '{ec.name}' already exists")
                continue
            entity = Entity(
                id=uuid.uuid4(),
                name=ec.name,
                entity_type=ec.entity_type,
                description=ec.description,
                observations=list(ec.observations),
                is_root=ec.is_root,
            )
            self.state.knowledge_graph.entities.append(entity)
            existing_names.add(ec.name)
            self._embed_entity(entity)
        return errors

    def _create_relations(self, relations: list[RelationCreate]) -> list[str]:
        """Add new relations, validating entity references and deduplicating.

        Returns:
            List of error strings for missing entity references or duplicates.
        """
        errors: list[str] = []
        existing_names = {e.name for e in self.state.knowledge_graph.entities}
        existing_triples = {
            (r.from_entity, r.to_entity, r.relation_type)
            for r in self.state.knowledge_graph.relations
        }

        for rc in relations:
            if rc.from_entity not in existing_names:
                errors.append(f"Relation references non-existent from_entity '{rc.from_entity}'")
                continue
            if rc.to_entity not in existing_names:
                errors.append(f"Relation references non-existent to_entity '{rc.to_entity}'")
                continue

            triple = (rc.from_entity, rc.to_entity, rc.relation_type)
            if triple in existing_triples:
                # Silently skip duplicate relations (same as V1)
                continue

            relation = Relation(
                id=uuid.uuid4(),
                from_entity=rc.from_entity,
                to_entity=rc.to_entity,
                relation_type=rc.relation_type,
                description=rc.description,
            )
            self.state.knowledge_graph.relations.append(relation)
            existing_triples.add(triple)
            self._embed_relation(relation)
        return errors

    # ------------------------------------------------------------------
    # CRUD: Update
    # ------------------------------------------------------------------

    def _update_entities(self, updates: list[EntityUpdate]) -> list[str]:
        """Apply partial updates to existing entities.

        Only non-None fields are applied. ``add_observations`` appends,
        ``remove_observations`` removes from the existing list.

        Returns:
            List of error strings for entities not found.
        """
        errors: list[str] = []
        entity_map = self._entity_map()

        for eu in updates:
            entity = entity_map.get(eu.name)
            if entity is None:
                errors.append(f"Entity '{eu.name}' not found for update")
                continue

            if eu.description is not None and eu.description != entity.description:
                self._vector_index.remove({str(entity.id)})
                entity.description = eu.description
                self._embed_entity(entity)
            elif eu.description is not None:
                entity.description = eu.description
            if eu.entity_type is not None:
                entity.entity_type = eu.entity_type
            if eu.is_root is not None:
                entity.is_root = eu.is_root
            if eu.add_observations is not None:
                entity.observations.extend(eu.add_observations)
            if eu.remove_observations is not None:
                to_remove = set(eu.remove_observations)
                entity.observations = [o for o in entity.observations if o not in to_remove]
        return errors

    # ------------------------------------------------------------------
    # CRUD: Delete
    # ------------------------------------------------------------------

    def _delete_entities(self, names: list[str]) -> list[str]:
        """Remove entities by name with cascade deletion of relations.

        Also removes any associated VectorEntry embeddings from the vector index.

        Returns:
            List of error strings for entities not found.
        """
        errors: list[str] = []
        existing_names = {e.name for e in self.state.knowledge_graph.entities}

        for name in names:
            if name not in existing_names:
                errors.append(f"Entity '{name}' not found for deletion")

        names_set = set(names) & existing_names

        if names_set:
            # Collect UUIDs of deleted entities for embedding removal
            _deleted_ids = [
                e.id for e in self.state.knowledge_graph.entities if e.name in names_set
            ]

            # Collect UUIDs of cascade-deleted relations
            _deleted_rel_ids = [
                r.id
                for r in self.state.knowledge_graph.relations
                if r.from_entity in names_set or r.to_entity in names_set
            ]

            # Remove entities
            self.state.knowledge_graph.entities = [
                e for e in self.state.knowledge_graph.entities if e.name not in names_set
            ]

            # Cascade-delete relations referencing deleted entities
            self.state.knowledge_graph.relations = [
                r
                for r in self.state.knowledge_graph.relations
                if r.from_entity not in names_set and r.to_entity not in names_set
            ]

            self._vector_index.remove(
                {str(eid) for eid in _deleted_ids} | {str(rid) for rid in _deleted_rel_ids}
            )

        return errors

    def _delete_relations(self, deletions: list[RelationDelete]) -> list[str]:
        """Remove relations by ``(from_entity, to_entity, relation_type)`` triple.

        Also removes any associated VectorEntry embeddings from the vector index.

        Returns:
            List of error strings for relations not found.
        """
        errors: list[str] = []
        existing_triples = {
            (r.from_entity, r.to_entity, r.relation_type): r
            for r in self.state.knowledge_graph.relations
        }

        triples_to_delete: set[tuple[str, str, str]] = set()
        for rd in deletions:
            triple = (rd.from_entity, rd.to_entity, rd.relation_type)
            if triple not in existing_triples:
                errors.append(
                    f"Relation ({rd.from_entity} -> {rd.to_entity} [{rd.relation_type}]) not found"
                )
            else:
                triples_to_delete.add(triple)

        if triples_to_delete:
            # Collect UUIDs for embedding removal
            _deleted_rel_ids = [
                r.id
                for r in self.state.knowledge_graph.relations
                if (r.from_entity, r.to_entity, r.relation_type) in triples_to_delete
            ]

            self.state.knowledge_graph.relations = [
                r
                for r in self.state.knowledge_graph.relations
                if (r.from_entity, r.to_entity, r.relation_type) not in triples_to_delete
            ]

            self._vector_index.remove({str(rid) for rid in _deleted_rel_ids})

        return errors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_graph(self, update: ManageGraph) -> str:
        """Execute batch mutations in deterministic order.

        Order: create_entities → create_relations → update_entities
        → delete_relations → delete_entities.

        All errors are collected across all operations. After mutations,
        ``state.notify_state_change()`` is called regardless of errors.
        A single ``RetriableError`` is raised if any errors occurred.

        Args:
            update: Batch mutation request.

        Returns:
            ``"Done"`` on success.

        Raises:
            RetriableError: With all collected error messages joined by "; ".
        """
        errors: list[str] = []

        errors.extend(self._create_entities(update.create_entities))
        errors.extend(self._create_relations(update.create_relations))
        errors.extend(self._update_entities(update.update_entities))
        errors.extend(self._delete_relations(update.delete_relations))
        errors.extend(self._delete_entities(update.delete_entities))

        self.state.notify_state_change()

        if errors:
            raise RetriableError("Update errors: " + "; ".join(errors))
        return "Done"

    def get_graph(self, query: GetGraphQuery | None = None) -> GraphView:
        """Return a full or filtered subgraph view.

        When ``query`` is None or has empty ``entity_names`` (and
        ``roots_only`` is False), the full graph is returned.  Otherwise
        BFS expansion is performed from the seed entities up to
        ``depth`` hops, optionally filtered by ``relation_types``.

        When ``roots_only`` is True, entities with ``is_root=True`` are
        used as BFS seeds and ``entity_names`` is ignored.  Depth
        defaults to 0 (no expansion) for ``roots_only`` mode.

        Args:
            query: Subgraph query parameters.  Defaults to full graph.

        Returns:
            GraphView with discovered entities and connecting relations.
        """
        if query is None:
            query = GetGraphQuery()

        # roots_only takes precedence over entity_names (AC-6)
        if query.roots_only:
            root_names = [e.name for e in self.state.knowledge_graph.entities if e.is_root]
            if not root_names:
                return GraphView()
            # depth=None → 0 for roots_only (AC-4); explicit depth used as-is (AC-5)
            effective_depth = query.depth if query.depth is not None else 0
            return self._get_subgraph(
                entity_names=root_names,
                depth=effective_depth,
                relation_types=query.relation_types,
            )

        if not query.entity_names:
            return GraphView(
                entities=list(self.state.knowledge_graph.entities),
                relations=list(self.state.knowledge_graph.relations),
            )

        # depth=None → 1 for entity_names mode (backward compat)
        effective_depth = query.depth if query.depth is not None else 1
        return self._get_subgraph(
            entity_names=query.entity_names,
            depth=effective_depth,
            relation_types=query.relation_types,
        )

    # ------------------------------------------------------------------
    # Subgraph BFS
    # ------------------------------------------------------------------

    def _get_subgraph(
        self,
        entity_names: list[str],
        depth: int,
        relation_types: list[str],
    ) -> GraphView:
        """BFS expansion from root entities.

        At each hop, relations connecting a frontier entity to an
        unvisited entity are traversed.  ``relation_types`` limits which
        edge types are followed.  After expansion completes, only
        relations whose **both** endpoints are in the visited set are
        included in the result.
        """
        graph = self.state.knowledge_graph
        entity_map = {e.name: e for e in graph.entities}

        visited: set[str] = set()
        frontier: set[str] = set()
        for name in entity_names:
            if name in entity_map:
                frontier.add(name)

        # BFS expansion: depth=0 means roots only (no expansion)
        for _ in range(depth):
            visited |= frontier
            next_frontier: set[str] = set()
            for rel in graph.relations:
                if relation_types and rel.relation_type not in relation_types:
                    continue
                if rel.from_entity in frontier and rel.to_entity not in visited:
                    next_frontier.add(rel.to_entity)
                if rel.to_entity in frontier and rel.from_entity not in visited:
                    next_frontier.add(rel.from_entity)
            frontier = next_frontier

        visited |= frontier

        result_entities = [e for e in graph.entities if e.name in visited]
        result_relations = [
            r for r in graph.relations if r.from_entity in visited and r.to_entity in visited
        ]
        return GraphView(entities=result_entities, relations=result_relations)

    # ------------------------------------------------------------------
    # Search (keyword / vector / hybrid)
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> SearchResult:
        """Search the knowledge graph by keyword, vector, or hybrid mode.

        ``mode="keyword"``: case-insensitive substring matching across
        entity and relation fields.  ``mode="vector"``: embeds the query
        and returns top-k results by cosine similarity; returns empty when
        no embeddings exist or the embedding service is unavailable.
        ``mode="hybrid"``: merges keyword and vector results with weighted
        scoring; falls back to keyword-only when embeddings are unavailable.

        Args:
            query: Search parameters.

        Returns:
            SearchResult with ranked hits.
        """
        if query.mode == "vector":
            return self._vector_search(query.query, query.top_k)
        if query.mode == "hybrid":
            return self._hybrid_search(query.query, query.top_k)
        return self._keyword_search(query.query, query.top_k)

    def _resolve_ref(self, ref_id: str) -> Entity | Relation | None:
        """Resolve a ``ref_id`` UUID string to the corresponding Entity or Relation.

        Scans entities first, then relations.  Returns ``None`` when the
        ref_id is not found (stale index entry).

        Args:
            ref_id: UUID string from a ``VectorEntry``.

        Returns:
            The matching ``Entity`` or ``Relation``, or ``None``.
        """
        for entity in self.state.knowledge_graph.entities:
            if str(entity.id) == ref_id:
                return entity
        for relation in self.state.knowledge_graph.relations:
            if str(relation.id) == ref_id:
                return relation
        return None

    def _vector_search(self, query_text: str, top_k: int) -> SearchResult:
        """Embed ``query_text`` and return top-k results by cosine similarity.

        Returns an empty ``SearchResult`` when the embedding service is
        unavailable, the index is empty, or the embedding call fails.

        Args:
            query_text: The natural-language query to embed and search.
            top_k: Maximum number of results to return.

        Returns:
            ``SearchResult`` with hits ranked by cosine similarity score.
        """
        svc = self._get_or_create_embedding_svc()
        if svc is None or len(self._vector_index) == 0:
            return SearchResult(hits=[])
        try:
            vectors = svc.embed([query_text])
            query_vector = vectors[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Vector search embedding failed: %s", self.config.name, exc)
            return SearchResult(hits=[])

        pairs = self._vector_index.search_cosine(query_vector, top_k)
        hits: list[SearchHit] = []
        for ref_id, score in pairs:
            obj = self._resolve_ref(ref_id)
            if obj is None:
                continue
            if isinstance(obj, Entity):
                hits.append(SearchHit(ref_type="entity", ref_id=ref_id, score=score, entity=obj))
            else:
                hits.append(
                    SearchHit(ref_type="relation", ref_id=ref_id, score=score, relation=obj)
                )
        return SearchResult(hits=hits)

    def _hybrid_search(self, query_text: str, top_k: int) -> SearchResult:
        """Merge keyword and vector search results with weighted scoring.

        Scoring rules:
        - Keyword-only hit: ``score = 1.0``
        - Vector-only hit: ``score = cosine_similarity``
        - Hit in both: ``score = cosine_similarity + 0.5`` (keyword boost)

        Falls back to keyword-only when the vector search returns no hits
        (embedding service unavailable or index empty — AC-4).

        Args:
            query_text: The query string.
            top_k: Maximum number of results to return.

        Returns:
            ``SearchResult`` with merged hits ranked by combined score.
        """
        keyword_result = self._keyword_search(query_text, top_k * 2)
        vector_result = self._vector_search(query_text, top_k * 2)

        if not vector_result.hits:
            # Graceful fallback: no embeddings → keyword only (AC-4)
            return SearchResult(hits=keyword_result.hits[:top_k])

        merged: dict[str, SearchHit] = {}

        for hit in vector_result.hits:
            merged[hit.ref_id] = hit  # start with vector score

        for kw_hit in keyword_result.hits:
            if kw_hit.ref_id in merged:
                existing = merged[kw_hit.ref_id]
                merged[kw_hit.ref_id] = existing.model_copy(
                    update={"score": existing.score + 0.5}
                )
            else:
                merged[kw_hit.ref_id] = kw_hit  # keyword-only hit (score = 1.0)

        ranked = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return SearchResult(hits=ranked[:top_k])

    def _keyword_search(self, query: str, top_k: int) -> SearchResult:
        """Case-insensitive substring search across entity and relation fields."""
        hits: list[SearchHit] = []
        q = query.lower()

        for entity in self.state.knowledge_graph.entities:
            if (
                q in entity.name.lower()
                or q in entity.description.lower()
                or q in entity.entity_type.lower()
            ):
                hits.append(
                    SearchHit(
                        ref_type="entity",
                        ref_id=str(entity.id),
                        score=1.0,
                        entity=entity,
                    )
                )

        for relation in self.state.knowledge_graph.relations:
            if (
                q in relation.relation_type.lower()
                or q in relation.description.lower()
                or q in relation.from_entity.lower()
                or q in relation.to_entity.lower()
            ):
                hits.append(
                    SearchHit(
                        ref_type="relation",
                        ref_id=str(relation.id),
                        score=1.0,
                        relation=relation,
                    )
                )

        return SearchResult(hits=sorted(hits, key=lambda h: h.score, reverse=True)[:top_k])
