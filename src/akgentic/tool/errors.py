"""Tool-layer exceptions for error signaling.

These exceptions are framework-agnostic — the integration layer (e.g., akgentic-team)
translates them into the appropriate retry mechanism (e.g., pydantic-ai ModelRetry).
"""


class RetriableError(Exception):
    """Raised by tool implementations when the LLM should retry with corrected input.

    Tools raise this to signal a recoverable error. The consuming framework
    (via ToolFactory's retry_exception injection) converts it to the appropriate
    retry mechanism.
    """

    pass


class RoleNotHireableError(RetriableError):
    """Raised when a role exists in the catalog but the team does not permit hiring it.

    This is not a lookup miss: the agent card was found. Its ``can_be_hired`` is
    ``False``, so the hire is refused before any actor is created.

    Subclassing :class:`RetriableError` is load-bearing rather than cosmetic —
    the consuming framework's retry translation catches ``RetriableError``, so a
    refusal reaches the model as a retry with corrected input, exactly like a
    missing role. The subclass exists so callers that need to tell the two apart
    can, without parsing message text.
    """

    pass


class ToolObserverGone(RuntimeError):  # noqa: N818 — name mirrors CommandNotRecognized precedent
    """A tool callable ran after its owning agent was stopped."""

    pass


class CommandNotRecognized(Exception):  # noqa: N818 — story-mandated name; identification signal
    """Raised when the first dispatch token is not a known command (ADR-028 §Decision 4).

    This is an *identification-failure* signal, not a tool execution error: it
    tells the agent message handler "this was never a command — treat it as plain
    text and fall back to normal LLM processing." It is deliberately NOT a
    :class:`RetriableError` subclass; a retriable error means "retry with corrected
    input," which is a semantically distinct outcome. The two hierarchies are
    kept disjoint on purpose.
    """

    pass
