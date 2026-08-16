"""Topological ordering of ``ToolCard`` instances by their ``depends_on`` declarations."""

from collections import deque

from .card import ToolCard


def _topological_sort(cards: list[ToolCard]) -> list[ToolCard]:
    """Return ``cards`` topologically sorted by ``ToolCard.depends_on``.

    Dependency keys are matched against ``type(card).__name__``. The sort uses
    Kahn's algorithm with a FIFO queue seeded in input order, which produces a
    deterministic ordering: independent nodes retain their relative input order.

    Duplicate class names in ``cards`` (e.g. two ``VectorStoreTool`` instances
    with different configuration) are permitted — later entries overwrite
    earlier entries in the internal name→card map. Dependency relationships
    are at the class level, not per-instance.

    Args:
        cards: Tool cards to sort. Input order is preserved for independent
            nodes.

    Returns:
        A new list containing the same cards in dependency-respecting order
        (prerequisites before dependents).

    Raises:
        ValueError: If a declared dependency is not present in ``cards``
            (message names both the dependent and the missing class), or if
            the dependency graph contains a cycle (message contains ``"cycle"``
            and lists the class names involved).
    """
    # Name → card map. Later duplicates overwrite earlier entries — dependency
    # relationships are at the class level, not per-instance.
    by_name: dict[str, ToolCard] = {type(card).__name__: card for card in cards}

    _validate_dependencies_present(cards, by_name)
    in_degree, dependents = _build_dependency_graph(by_name)
    ordered_names = _kahn_order(cards, in_degree, dependents)

    if len(ordered_names) < len(by_name):
        remaining = sorted(set(by_name) - set(ordered_names))
        raise ValueError(f"ToolCard dependency cycle detected: {remaining}")

    return _expand_names_to_instances(cards, ordered_names)


def _validate_dependencies_present(cards: list[ToolCard], by_name: dict[str, ToolCard]) -> None:
    """Fail fast if any card declares a dependency absent from the card list.

    Raises:
        ValueError: Naming both the dependent card and the missing class.
    """
    for card in cards:
        for dep in card.depends_on:
            if dep not in by_name:
                raise ValueError(
                    f"{type(card).__name__} depends on {dep} but it was not found in the tool list"
                )


def _build_dependency_graph(
    by_name: dict[str, ToolCard],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Return ``(in_degree, dependents)`` keyed by class name.

    Both maps are keyed by class name rather than instance, so duplicate class
    names collapse to a single node — dependencies are class-level.
    """
    in_degree: dict[str, int] = dict.fromkeys(by_name, 0)
    dependents: dict[str, list[str]] = {name: [] for name in by_name}
    for name, card in by_name.items():
        for dep in card.depends_on:
            in_degree[name] += 1
            dependents[dep].append(name)
    return in_degree, dependents


def _kahn_order(
    cards: list[ToolCard],
    in_degree: dict[str, int],
    dependents: dict[str, list[str]],
) -> list[str]:
    """Return class names in dependency order via Kahn's algorithm.

    The queue is seeded with zero-in-degree names in input order (FIFO ⇒
    deterministic for the same input), so independent nodes keep their relative
    position. ``in_degree`` is consumed in place. A shorter-than-expected result
    means a cycle; the caller reports it.
    """
    queue: deque[str] = deque()
    seen: set[str] = set()
    for card in cards:
        name = type(card).__name__
        if name in seen:
            continue
        seen.add(name)
        if in_degree[name] == 0:
            queue.append(name)

    ordered_names: list[str] = []
    while queue:
        name = queue.popleft()
        ordered_names.append(name)
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    return ordered_names


def _expand_names_to_instances(cards: list[ToolCard], ordered_names: list[str]) -> list[ToolCard]:
    """Map sorted class names back to instances, preserving input order per class.

    Duplicate class names emit their instances in the order they appeared in the
    input, grouped by that class's position in ``ordered_names``.
    """
    by_name_instances: dict[str, list[ToolCard]] = {}
    for card in cards:
        by_name_instances.setdefault(type(card).__name__, []).append(card)
    return [card for name in ordered_names for card in by_name_instances[name]]
