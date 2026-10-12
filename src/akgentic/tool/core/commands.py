"""Signature-derived command entries and the ``/``-dispatching ``CommandRegistry``."""

import inspect
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import TypeAdapter

from akgentic.tool.core.event import CommandArg, CommandDescriptor
from akgentic.tool.errors import CommandNotRecognized


class _RestOfLineMarker:
    """The ``Annotated`` metadata that makes a command's last parameter rest-of-line."""

    def __repr__(self) -> str:
        return "RestOfLine"


_REST_OF_LINE = _RestOfLineMarker()

RestOfLine = Annotated[str, _REST_OF_LINE]
"""A command's last, required parameter that receives the raw rest of the line.

The parameters before it are tokenised with ``shlex`` as usual; this one gets
whatever text follows them **verbatim** — leading whitespace stripped, inner and
trailing whitespace, newlines and quotes kept — so ``/workspace_write notes.md
it's a "quoted" word`` writes exactly ``it's a "quoted" word``. It is never bound
as ``name=value``: ``content=x`` in the rest of the line is part of the content.
"""


def _is_rest_of_line(annotation: Any) -> bool:
    """Whether *annotation* carries the :data:`RestOfLine` marker."""
    return get_origin(annotation) is Annotated and _REST_OF_LINE in get_args(annotation)[1:]


@dataclass(frozen=True)
class _CommandArgSpec:
    """Ordered metadata + per-arg coercion adapter for one command parameter.

    Runtime-only (not serialized). Captured from the callable signature at
    registry-construction time. Drives both positional coercion (dispatch) and
    :class:`CommandDescriptor` building.
    """

    name: str
    annotation: Any
    required: bool
    adapter: TypeAdapter[Any]


@dataclass(frozen=True)
class _CommandEntry:
    """Runtime record for a single registered command (not a serialized model).

    Holds the callable plus the ordered, per-argument coercion metadata derived
    from its signature. Per Golden Rule #1b, runtime callables live here in a
    plain dataclass — never inside a serialized Pydantic field.
    """

    name: str
    fn: Callable[..., Any]
    args: tuple[_CommandArgSpec, ...]
    tool_card: str
    rest_of_line: bool = False
    """Whether the last argument receives the raw rest of the line (:data:`RestOfLine`)."""

    @property
    def required_count(self) -> int:
        """Number of leading required (no-default) parameters."""
        return sum(1 for spec in self.args if spec.required)


def _json_type_name(annotation: Any) -> str:
    """Return the JSON-schema type name for a parameter annotation.

    Falls back to ``"string"`` when the schema has no top-level ``type`` (e.g.
    a union like ``str | None`` produces ``anyOf``), matching how the human help
    surface renders un-typed-or-optional args.
    """
    try:
        schema = TypeAdapter(annotation).json_schema()
    except Exception:
        return "string"
    type_name = schema.get("type", "string")
    return type_name if isinstance(type_name, str) else "string"


def _build_command_entry(fn: Callable[..., Any], tool_card: str) -> _CommandEntry:
    """Derive a per-command arg model + ordered metadata from a callable signature.

    Mirrors how pydantic-ai derives a tool schema from a function signature:
    inspect the parameters, reject anything un-derivable (``*args``, ``**kwargs``,
    or an un-annotated parameter), and build a :class:`TypeAdapter` over the
    ordered positional parameter types for later coercion.

    The hints are read with ``include_extras=True``: without it ``Annotated``
    metadata is stripped and a :data:`RestOfLine` parameter is indistinguishable
    from a plain ``str``.

    Raises:
        ValueError: If the signature has a ``VAR_POSITIONAL`` (``*args``),
            ``VAR_KEYWORD`` (``**kwargs``), or un-annotated parameter, or a
            :data:`RestOfLine` parameter that is not the last or not required. The
            message names the command and the offending parameter.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    specs: list[_CommandArgSpec] = []
    for pname, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ValueError(
                f"Command '{fn.__name__}' cannot be registered: parameter '*{pname}' "
                "(*args) has no derivable argument schema."
            )
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(
                f"Command '{fn.__name__}' cannot be registered: parameter '**{pname}' "
                "(**kwargs) has no derivable argument schema."
            )
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ValueError(
                f"Command '{fn.__name__}' cannot be registered: parameter '{pname}' "
                "has no type annotation."
            )
        required = param.default is inspect.Parameter.empty
        if _is_rest_of_line(annotation):
            _check_rest_of_line(fn, pname, required, last=pname == list(sig.parameters)[-1])
            annotation = str
        specs.append(
            _CommandArgSpec(
                name=pname,
                annotation=annotation,
                required=required,
                adapter=TypeAdapter(annotation),
            )
        )

    rest_of_line = bool(specs) and _is_rest_of_line(hints.get(specs[-1].name))
    return _CommandEntry(
        name=fn.__name__,
        fn=fn,
        args=tuple(specs),
        tool_card=tool_card,
        rest_of_line=rest_of_line,
    )


def _check_rest_of_line(fn: Callable[..., Any], pname: str, required: bool, last: bool) -> None:
    """Refuse a :data:`RestOfLine` parameter anywhere but last, or with a default.

    Raises:
        ValueError: Naming the command and the parameter.
    """
    if not last:
        raise ValueError(
            f"Command '{fn.__name__}' cannot be registered: RestOfLine parameter "
            f"'{pname}' must be the last parameter."
        )
    if not required:
        raise ValueError(
            f"Command '{fn.__name__}' cannot be registered: RestOfLine parameter "
            f"'{pname}' must be required."
        )


def _split_leading(rest: str, count: int) -> tuple[list[str], str]:
    """Tokenise the first *count* arguments of *rest* and return the raw remainder.

    The lexer is configured exactly as ``shlex.split`` configures its own —
    POSIX, whitespace-split, no comment characters — so the leading arguments
    tokenise as they always have. It stops after *count* tokens, and whatever it
    has not consumed is the remainder, verbatim but for leading whitespace.

    Raises:
        ValueError: When a leading argument's quote does not close.
    """
    stream = StringIO(rest)
    lexer = shlex.shlex(stream, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[str] = []
    while len(tokens) < count:
        token = lexer.get_token()
        if token is None:
            return tokens, ""
        tokens.append(token)
    return tokens, stream.read().lstrip()


class CommandRegistry:
    """Name-keyed registry of command callables with signature-derived dispatch.

    Built by :meth:`ToolFactory.get_command_registry` from the ``get_commands()``
    output of every wired :class:`ToolCard`. Each command is keyed by its
    callable's ``__name__`` (e.g. ``hire_member``). The registry exposes a typed
    programmatic surface (:meth:`callable`), a membership test (:meth:`has`),
    discovery metadata (:meth:`descriptors`), and a human text surface
    (:meth:`dispatch`) that parses ``/``-prefixed commands.

    This is a runtime object holding callables — deliberately a plain class, not a
    ``BaseModel`` (Golden Rule #1b): runtime callables must never live in a
    serialized field.
    """

    def __init__(self, entries: dict[str, _CommandEntry]) -> None:
        """Store the name → command-entry mapping. Use the factory to build one."""
        self._entries = entries

    def has(self, name: str) -> bool:
        """Return ``True`` if a command named *name* is registered."""
        return name in self._entries

    def callable(self, name: str) -> Callable[..., Any]:
        """Return the bound, typed command callable for programmatic invocation.

        The returned callable preserves its **native** (non-stringified) return
        value, so callers (e.g. ``StructuredOutput`` hire-by-role) can invoke it
        with native arguments and use the result directly.

        Raises:
            CommandNotRecognized: If *name* is not a registered command.
        """
        try:
            return self._entries[name].fn
        except KeyError:
            raise CommandNotRecognized(name) from None

    def descriptors(self) -> list[CommandDescriptor]:
        """Return serializable discovery metadata, one entry per command."""
        result: list[CommandDescriptor] = []
        for entry in self._entries.values():
            args = [
                CommandArg(
                    name=spec.name,
                    type=_json_type_name(spec.annotation),
                    required=spec.required,
                )
                for spec in entry.args
            ]
            description = inspect.getdoc(entry.fn) or ""
            result.append(
                CommandDescriptor(
                    name=entry.name,
                    description=description,
                    args=args,
                    tool_card=entry.tool_card,
                )
            )
        return result

    def dispatch(self, text: str) -> str | None:
        """Parse a ``/``-prefixed command, invoke it, and return its result surface.

        Strips the leading ``/``, resolves the first **whitespace** token to a
        command, ``shlex.split``s the rest, classifies those tokens as positional or
        ``name=value`` keyword arguments, coerces and merges them, invokes the
        command, and renders the result.

        **The name is not taken from ``shlex``**, because ``shlex`` raises on a
        broken quote and the name decides which of the two failure contracts
        applies. A text naming a registered command whose arguments will not
        tokenise is that command's failure, returned as a result string like any
        other; a text naming no command still raises ``CommandNotRecognized``.

        A command returning ``None`` dispatches to ``None`` — it handled itself
        and has nothing to report, and the caller says nothing rather than
        echoing the literal string ``"None"`` (ADR-040 §4). Every other result
        is string-rendered as before.

        A token is a **keyword** only when the text before its first ``=`` matches a
        real parameter name on the command; otherwise it is positional (so values
        containing ``=`` are never silently swallowed). Positionals must precede
        keywords. ``key=value`` is opt-in — purely-positional dispatch is unchanged.

        Raises:
            CommandNotRecognized: If the first token does not name a known command
                (so the caller may fall back to normal LLM processing). No command
                is invoked in this case.

        Post-identification failures (missing/extra args, coercion errors, unknown
        keyword, duplicate binding, positional-after-keyword, or the command body
        raising) are caught **inside** this method and returned as a plain result
        string — ``CommandNotRecognized`` is never raised once a command has been
        identified.
        """
        head = (text[1:] if text.startswith("/") else text).split(maxsplit=1)
        if not head:
            raise CommandNotRecognized(text)
        name = head[0]
        if name not in self._entries:
            raise CommandNotRecognized(name)
        return self._invoke(name, head[1] if len(head) > 1 else "")

    def _invoke(self, name: str, rest: str) -> str | None:
        """Tokenise *rest*, then classify, merge, coerce for command *name* and invoke it.

        A ``None`` result is propagated unchanged; every other result is
        string-rendered, so commands returning a non-``str`` value still reach
        the caller as text.

        Any failure (a quote that does not close, positional-after-keyword, too
        many/missing args, unknown keyword, duplicate binding, coercion error, or the
        command body raising) is caught and returned as a result string.
        """
        entry = self._entries[name]
        try:
            positional, keyword = self._tokenise(entry, rest)
            bound = self._bind(entry, positional, keyword)
            result = entry.fn(**bound)
            return None if result is None else str(result)
        except Exception as exc:  # noqa: BLE001 — failures become result strings (ADR-028 §4)
            return f"Command '{name}' failed: {exc}"

    @classmethod
    def _tokenise(cls, entry: _CommandEntry, rest: str) -> tuple[list[str], dict[str, str]]:
        """Split *rest* into ``(positional, keyword)`` for *entry*.

        A plain command ``shlex.split``s it whole. A :data:`RestOfLine` command
        tokenises only the arguments before the last, and binds the raw remainder
        to the last by name — so it can never be supplied as ``name=value``, and
        an empty remainder leaves it unbound, which :meth:`_bind` reports as a
        missing argument.
        """
        if not entry.rest_of_line:
            return cls._classify_tokens(entry, shlex.split(rest))
        leading, remainder = _split_leading(rest, len(entry.args) - 1)
        positional, keyword = cls._classify_tokens(entry, leading, rest_of_line=True)
        if remainder:
            keyword[entry.args[-1].name] = remainder
        return positional, keyword

    @staticmethod
    def _classify_tokens(
        entry: _CommandEntry, args: list[str], rest_of_line: bool = False
    ) -> tuple[list[str], dict[str, str]]:
        """Partition *args* into ``(positional, keyword)`` for *entry*.

        A token is a keyword iff it contains ``=`` AND the substring before the
        **first** ``=`` is a known parameter name on *entry*; the value is the
        remainder after that first ``=``. All other tokens are positional. A
        positional token appearing after any keyword token is rejected.

        With *rest_of_line*, the last parameter is not a keyword candidate: its
        value is the raw remainder, bound by :meth:`_tokenise`.

        Raises:
            ValueError: If a positional token follows a keyword token (names the
                offending positional value).
        """
        specs = entry.args[:-1] if rest_of_line else entry.args
        names = {spec.name for spec in specs}
        positional: list[str] = []
        keyword: dict[str, str] = {}
        for token in args:
            key, sep, value = token.partition("=")
            if sep and key in names:
                keyword[key] = value
            elif keyword:
                raise ValueError(
                    f"positional argument '{token}' cannot follow a keyword argument"
                )
            else:
                positional.append(token)
        return positional, keyword

    @staticmethod
    def _bind(
        entry: _CommandEntry, positional: list[str], keyword: dict[str, str]
    ) -> dict[str, Any]:
        """Merge *positional* + *keyword* onto *entry*'s params and coerce each.

        Maps positionals onto the leading parameters in signature order, then binds
        keywords by name, detecting unknown names and duplicate bindings. Validates
        arity (at least ``required_count``, at most ``len(args)``) over the merged
        set, then coerces every bound value through its per-arg :class:`TypeAdapter`.
        Unbound trailing optionals are omitted so the callable applies its defaults.

        Raises:
            ValueError: Too many positionals, unknown keyword, duplicate binding, a
                required parameter left unbound, or a coercion failure.
        """
        if len(positional) > len(entry.args):
            raise ValueError(
                f"accepts at most {len(entry.args)} argument(s), got {len(positional)}"
            )
        raw: dict[str, str] = {
            spec.name: token
            for spec, token in zip(entry.args, positional, strict=False)
        }
        specs_by_name = {spec.name: spec for spec in entry.args}
        for key, value in keyword.items():
            if key not in specs_by_name:
                raise ValueError(f"unknown keyword argument '{key}'")
            if key in raw:
                raise ValueError(f"got multiple values for argument '{key}'")
            raw[key] = value
        missing = [spec.name for spec in entry.args if spec.required and spec.name not in raw]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")
        return {
            name: specs_by_name[name].adapter.validate_python(token)
            for name, token in raw.items()
        }
