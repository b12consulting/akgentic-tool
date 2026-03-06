"""KnowledgeGraph actor with batch CRUD operations.

Mirrors the PlanActor pattern: Akgent subclass with typed state,
error-collection semantics, and ``BaseState.notify_state_change()``
for orchestrator integration.

Source: V2 redesign of akgentic-framework KnowledgeGraphManager.
"""

from __future__ import annotations

import uuid

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
)

KG_ACTOR_NAME: str = "#KnowledgeGraphTool"
"""Singleton actor name registered with the orchestrator."""

KG_ACTOR_ROLE: str = "ToolActor"
"""Actor role constant for ToolCard integration."""


class KnowledgeGraphActor(Akgent[BaseConfig, KnowledgeGraphState]):
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
        """Initialize state and attach observer for orchestrator notifications."""
        self.state = KnowledgeGraphState()
        self.state.observer(self)

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

            if eu.description is not None:
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

        Also removes any associated VectorEntry embeddings (no-op until Epic 2).

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
            # Collect UUIDs of deleted entities for future embedding removal
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

            # TODO(Epic 2): Remove associated VectorEntry embeddings by ref_id
            # self._remove_embeddings(_deleted_ids + _deleted_rel_ids)

        return errors

    def _delete_relations(self, deletions: list[RelationDelete]) -> list[str]:
        """Remove relations by ``(from_entity, to_entity, relation_type)`` triple.

        Also removes any associated VectorEntry embeddings (no-op until Epic 2).

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
            # Collect UUIDs for future embedding removal
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

            # TODO(Epic 2): Remove associated VectorEntry embeddings by ref_id
            # self._remove_embeddings(_deleted_rel_ids)

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
    # Keyword search
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> SearchResult:
        """Search the knowledge graph by keyword, vector, or hybrid mode.

        ``mode="keyword"``: case-insensitive substring matching across
        entity and relation fields.  ``mode="vector"``: placeholder
        returning empty results (Epic 2).  ``mode="hybrid"``: keyword
        only for now (vector component added in Epic 2).

        Args:
            query: Search parameters.

        Returns:
            SearchResult with ranked hits.
        """
        if query.mode == "vector":
            return SearchResult(hits=[])
        # hybrid and keyword both use keyword-only search for now
        return self._keyword_search(query.query, query.top_k)

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
