# Review of issue #340 implementation

Issue: [b12consulting/akgentic-tool#340](https://github.com/b12consulting/akgentic-tool/issues/340)  
Reviewed change: current working-tree diff against `8e9183d`

## Overall assessment

The initial review found two high-severity Qdrant defects, two medium-severity registry/configuration defects, and two documentation gaps. All findings below have been addressed in the current working tree, together with follow-up fixes for empty-state restoration, deliberate built-in registry replacements, workspace routing, cosine-score enforcement, and the documented Qdrant query parameter shape.

## Findings

### 1. High: Qdrant point IDs collide across teams and overwrite data — resolved

**Files:** `src/akgentic/tool/vector_store/qdrant.py:57-65`, `src/akgentic/tool/vector_store/qdrant.py:223-231`, `src/akgentic/tool/vector_store/qdrant.py:414-417`

Qdrant point IDs are derived only from `ref_id`. Collection names are shared by teams, and common references—particularly planning task IDs such as `"3"`—therefore produce the same point ID for different teams. Qdrant upsert replaces the existing point before payload filtering is relevant, so filtering searches by `team_id` does not provide isolation.

For example, if `team-a` and `team-b` both insert `ref_id="3"` into the `planning` collection, the second upsert replaces the first team's point. Searches for the first team then return no point because the surviving payload belongs to the second team.

Point identity must include the isolation boundary, such as the team ID and effective tenant, while retaining the original `ref_id` in the payload. This is a merge blocker because the current behavior can silently destroy another team's data.

### 2. High: The supported Qdrant dependency range does not provide the API used by search — resolved

**Files:** `pyproject.toml:42`, `pyproject.toml:54`, `src/akgentic/tool/vector_store/qdrant.py:292-301`

The optional dependency permits `qdrant-client` 1.9.x, but `QdrantBackend.search()` calls `QdrantClient.query_points`, which is not available in 1.9. A valid installation under the declared dependency constraint can ingest data but fails whenever it searches.

The actor converts this backend exception into an empty result, making the incompatibility appear to callers as a successful search with no matches. The dependency floor should be raised to the first version that supports `query_points` (at least 1.10), or the backend should use an API compatible with the declared range.

### 3. Medium: `persists_in_actor_state` is nonfunctional for registered third-party backends — resolved

**Files:** `src/akgentic/tool/vector_store/actor.py:282-285`, `src/akgentic/tool/vector_store/actor.py:350-360`, `src/akgentic/tool/vector_store/actor.py:400-408`, `src/akgentic/tool/vector_store/actor.py:473-481`, `src/akgentic/tool/vector_store/actor.py:554-561`

The actor reads the selected backend's `persists_in_actor_state` capability, but `_sync_backend_state()` always snapshots the built-in in-memory backend. Restoration is also hard-coded to `InMemoryBackend.restore_state()`. A third-party backend can declare the capability, but its state is neither saved nor restored.

This violates the capability-driven extension contract: registering such a backend appears supported but loses its state when the actor restarts. Snapshot and restoration need to operate on the selected backend through an explicit state persistence contract. If persistence is intentionally exclusive to the in-memory backend, the generic capability should not be exposed as though third-party implementations can use it.

### 4. Medium: Qdrant can pass the build-time guard without its required client library — resolved

**Files:** `src/akgentic/tool/vector_store/qdrant.py:95-116`, `src/akgentic/tool/vector_store/qdrant.py:122-141`, `src/akgentic/tool/vector_store/qdrant.py:488-505`, `src/akgentic/tool/vector_store/actor.py:222-237`, `src/akgentic/tool/vector_store/actor.py:331-362`

`require_qdrant_configured()` checks only whether the Qdrant URL is present. When the URL is configured but the optional `qdrant-client` dependency is absent, card construction succeeds. Backend construction fails later, after which the actor suppresses the failure and skips collection creation. Mutations and searches then degrade into no-ops or empty results.

This does not satisfy the requirement that an unprovisioned backend fail at card-build time with remediation guidance. The Qdrant configuration guard should verify both connection configuration and availability of the optional dependency, and its error should include the installation command.

## Acceptance-criteria gaps

### Optional dependency is imported during package initialization — resolved

**Files:** `src/akgentic/tool/vector_store/qdrant.py:122-127`, `src/akgentic/tool/vector_store/__init__.py:27-30`

Importing `akgentic.tool.vector_store` loads the Qdrant module, which attempts to import `qdrant_client`. Although the missing import is handled, this is not a fully lazy import and differs from the documented recommendation to import optional clients inside the backend factory.

### Root installation documentation does not include Qdrant — resolved

**File:** `README.md:114-128`, `README.md:1573-1586`

The root README does not list the `qdrant` extra, does not include it in the "Everything" installation command, and describes optional-backend failure behavior only for Weaviate. The vector-store README documents Qdrant, but the package-level installation guidance remains incomplete.

## Recommendation

The identified blockers and follow-up defects are resolved in the current working tree and covered by focused regression tests.
