"""Signature-derived command entries and the ``/``-dispatching ``CommandRegistry``."""

import inspect
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from akgentic.tool.errors import CommandNotRecognized
from akgentic.tool.event import CommandArg, CommandDescriptor


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

    Raises:
        ValueError: If the signature has a ``VAR_POSITIONAL`` (``*args``),
            ``VAR_KEYWORD`` (``**kwargs``), or un-annotated parameter. The message
            names the command and the offending parameter.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
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
        specs.append(
            _CommandArgSpec(
                name=pname,
                annotation=annotation,
                required=required,
                adapter=TypeAdapter(annotation),
            )
        )

    return _CommandEntry(name=fn.__name__, fn=fn, args=tuple(specs), tool_card=tool_card)


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

    def dispatch(self, text: str) -> str:
        """Parse a ``/``-prefixed command, invoke it, and return a result string.

        Strips the leading ``/``, ``shlex.split``s the remainder, resolves the
        first token to a command, classifies the remaining tokens as positional or
        ``name=value`` keyword arguments, coerces and merges them, invokes the
        command, and string-renders the result.

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
        tokens = shlex.split(text[1:] if text.startswith("/") else text)
        if not tokens:
            raise CommandNotRecognized(text)
        name, args = tokens[0], tokens[1:]
        if name not in self._entries:
            raise CommandNotRecognized(name)
        return self._invoke(name, args)

    def _invoke(self, name: str, args: list[str]) -> str:
        """Classify, merge, coerce *args* for command *name* and invoke it.

        Any failure (positional-after-keyword, too many/missing args, unknown
        keyword, duplicate binding, coercion error, or the command body raising) is
        caught and returned as a result string.
        """
        entry = self._entries[name]
        try:
            positional, keyword = self._classify_tokens(entry, args)
            bound = self._bind(entry, positional, keyword)
            return str(entry.fn(**bound))
        except Exception as exc:  # noqa: BLE001 — failures become result strings (ADR-028 §4)
            return f"Command '{name}' failed: {exc}"

    @staticmethod
    def _classify_tokens(
        entry: _CommandEntry, args: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        """Partition *args* into ``(positional, keyword)`` for *entry*.

        A token is a keyword iff it contains ``=`` AND the substring before the
        **first** ``=`` is a known parameter name on *entry*; the value is the
        remainder after that first ``=``. All other tokens are positional. A
        positional token appearing after any keyword token is rejected.

        Raises:
            ValueError: If a positional token follows a keyword token (names the
                offending positional value).
        """
        names = {spec.name for spec in entry.args}
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
