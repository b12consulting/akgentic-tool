"""Signature-derived command entries and the ``/``-dispatching ``CommandRegistry``."""

import inspect
import shlex
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import StringIO
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from griffe import Docstring, DocstringSectionParameters, GoogleOptions
from pydantic import BaseModel, TypeAdapter

from akgentic.tool.core.event import CommandArg, CommandDescriptor
from akgentic.tool.errors import CommandNotRecognized


class _RestOfLineMarker:
    """The ``Annotated`` metadata that makes a command's last parameter rest-of-line."""

    def __repr__(self) -> str:
        return "RestOfLine"


_REST_OF_LINE = _RestOfLineMarker()

RestOfLine = Annotated[str, _REST_OF_LINE]
"""A command's last parameter, which receives the raw rest of the line — its *text*.

A rest-of-line command reads ``/cmd [positional …] [name=value …] <text…>``. The
parameters before the text are tokenised with ``shlex``: those without a default
are positional (and may still be named), those with a default are *options* —
keyword-only, written ``name=value``, never bound positionally. The text starts
at the first token that is not a known option once every positional is bound,
and runs **verbatim** to the end of the line — leading whitespace stripped, inner
and trailing whitespace, newlines and quotes kept — so ``/workspace_write notes.md
it's a "quoted" word`` writes exactly ``it's a "quoted" word``, and
``/search_graph mailing mode=vector`` searches for ``mailing mode=vector``. The
text is never bound as ``name=value``: ``content=x`` in it is part of the content.

It may have a default, which an empty text leaves in place; without one, an
empty text is a missing argument.
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
    required: bool
    adapter: TypeAdapter[Any]
    keyword_only: bool = False
    """Whether this is an option of a rest-of-line command (``name=value`` only)."""
    rest_of_line: bool = False
    """Whether this is a rest-of-line command's text (:data:`RestOfLine`)."""


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


def _json_type_name(adapter: TypeAdapter[Any]) -> str:
    """Return the JSON-schema type name of a parameter's coercion adapter.

    Falls back to ``"string"`` when the schema has no top-level ``type`` (e.g.
    a union like ``str | None`` produces ``anyOf``), matching how the human help
    surface renders un-typed-or-optional args, and when the schema cannot be
    generated at all.
    """
    try:
        schema = adapter.json_schema()
    except Exception:
        return "string"
    type_name = schema.get("type", "string")
    return type_name if isinstance(type_name, str) else "string"


def _choices(adapter: TypeAdapter[Any]) -> list[str] | None:
    """Return the closed set of values a parameter's coercion adapter accepts.

    Read from the JSON schema: the top-level ``enum``, or the ``enum`` of the one
    non-``null`` branch of an ``anyOf`` (``Literal[...] | None``), following a
    ``$ref`` into ``$defs`` in either position (``MyEnum | None``). A ``boolean``
    is the closed set ``["true", "false"]``. ``None`` when the set is open, and
    when the schema cannot be generated at all.
    """
    try:
        schema = adapter.json_schema()
    except Exception:
        return None
    branches = [b for b in schema.get("anyOf", [schema]) if b.get("type") != "null"]
    if len(branches) != 1:
        return None
    branch = branches[0]
    if "$ref" in branch:
        branch = schema.get("$defs", {}).get(branch["$ref"].removeprefix("#/$defs/"), {})
    if "enum" in branch:
        return [str(value) for value in branch["enum"]]
    return ["true", "false"] if branch.get("type") == "boolean" else None


def _arg_descriptions(fn: Callable[..., Any]) -> dict[str, str]:
    """Map each parameter documented in *fn*'s Google-style ``Args:`` section to its text.

    Parsed with griffe, the parser pydantic-ai reads tool docstrings with.
    ``warnings=False`` because no ``parent`` is given, so griffe cannot find the
    parameters' annotations and would log about each one.
    """
    if not fn.__doc__:
        return {}
    docstring = Docstring(
        fn.__doc__, parser="google", parser_options=GoogleOptions(warnings=False)
    )
    return {
        param.name: param.description
        for section in docstring.parse()
        if isinstance(section, DocstringSectionParameters)
        for param in section.value
    }


def _build_command_entry(fn: Callable[..., Any], tool_card: str) -> _CommandEntry:
    """Derive a per-command arg model + ordered metadata from a callable signature.

    Mirrors how pydantic-ai derives a tool schema from a function signature:
    inspect the parameters, reject anything un-derivable (``*args``, ``**kwargs``,
    an un-annotated parameter, or one no text token can produce), and build a
    :class:`TypeAdapter` over the ordered positional parameter types for later
    coercion.

    The hints are read with ``include_extras=True``: without it ``Annotated``
    metadata is stripped and a :data:`RestOfLine` parameter is indistinguishable
    from a plain ``str``.

    Raises:
        ValueError: If the signature has a ``VAR_POSITIONAL`` (``*args``),
            ``VAR_KEYWORD`` (``**kwargs``), or un-annotated parameter, a parameter
            typed as — or containing — a Pydantic model (bare, or inside
            ``Optional`` / a union / a container), a :data:`RestOfLine` parameter
            that is not the last, or — in a rest-of-line command — a positional
            parameter after an option. The message names the command and the
            offending parameter.
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
        _check_producible_from_text(fn, pname, annotation)
        required = param.default is inspect.Parameter.empty
        if _is_rest_of_line(annotation):
            _check_rest_of_line(fn, pname, last=pname == list(sig.parameters)[-1])
            annotation = str
        specs.append(
            _CommandArgSpec(
                name=pname,
                required=required,
                adapter=TypeAdapter(annotation),
            )
        )

    rest_of_line = bool(specs) and _is_rest_of_line(hints.get(specs[-1].name))
    return _CommandEntry(
        name=fn.__name__,
        fn=fn,
        args=_mark_options(fn, specs) if rest_of_line else tuple(specs),
        tool_card=tool_card,
        rest_of_line=rest_of_line,
    )


def _mark_options(
    fn: Callable[..., Any], specs: list[_CommandArgSpec]
) -> tuple[_CommandArgSpec, ...]:
    """Mark a rest-of-line command's options and its text parameter.

    A parameter before the text that has a default is an option. One without a
    default after an option is refused: the splitter binds positionals first and
    then options, so it could only ever be reached by name. Python allows that
    shape only as a keyword-only parameter (``mode="a", *, path: str, ...``).

    Raises:
        ValueError: Naming the command and the positional parameter.
    """
    *leading, text = specs
    marked: list[_CommandArgSpec] = []
    for spec in leading:
        if spec.required and any(option.keyword_only for option in marked):
            raise ValueError(
                f"Command '{fn.__name__}' cannot be registered: positional parameter "
                f"'{spec.name}' follows an option (a parameter with a default)."
            )
        marked.append(replace(spec, keyword_only=not spec.required))
    return (*marked, replace(text, rest_of_line=True))


def _contains_model(annotation: Any) -> bool:
    """Whether *annotation* is, or has among its type arguments, a Pydantic model.

    Recurses through every ``get_args`` entry, so ``Model | None``,
    ``Optional[Model]``, ``list[Model]``, ``dict[str, Model]`` and
    ``Annotated[Model, ...]`` are all seen. Anything that is not a class
    (``Annotated`` metadata such as the :data:`RestOfLine` marker, ``Literal``
    values, ``Ellipsis``) is not a model.
    """
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return True
    return any(_contains_model(arg) for arg in get_args(annotation))


def _check_producible_from_text(fn: Callable[..., Any], pname: str, annotation: Any) -> None:
    """Refuse a parameter no text token can produce: one that is or contains a model.

    A dispatched token is coerced through the parameter's :class:`TypeAdapter`,
    and a string never validates as a model — so such a command could only ever
    answer a validation error. Refusing it here makes that a wiring failure.

    Raises:
        ValueError: Naming the command and the parameter.
    """
    if _contains_model(annotation):
        raise ValueError(
            f"Command '{fn.__name__}' cannot be registered: parameter '{pname}' is typed "
            f"{annotation!r}, which a text token cannot produce."
        )


def _check_rest_of_line(fn: Callable[..., Any], pname: str, last: bool) -> None:
    """Refuse a :data:`RestOfLine` parameter anywhere but last.

    Raises:
        ValueError: Naming the command and the parameter.
    """
    if not last:
        raise ValueError(
            f"Command '{fn.__name__}' cannot be registered: RestOfLine parameter "
            f"'{pname}' must be the last parameter."
        )


def _split_leading(rest: str, entry: _CommandEntry) -> tuple[list[str], str]:
    """Tokenise the positionals and options of a rest-of-line command; return the text.

    The lexer is configured exactly as ``shlex.split`` configures its own —
    POSIX, whitespace-split, no comment characters — so the leading arguments
    tokenise as they always have. Every token is consumed while a positional is
    unbound; after that only ``name=value`` tokens naming an option are, and the
    text starts at the first other token. Whatever has not been consumed is the
    text, verbatim but for leading whitespace.

    Raises:
        ValueError: When a quote does not close before every positional is bound.
            After that, a token that cannot be lexed is the start of the text.
    """
    stream = StringIO(rest)
    lexer = shlex.shlex(stream, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    leading = entry.args[:-1]
    tokens, bound = _take_positionals(lexer, leading)
    if not bound:  # the line ended first
        return tokens, ""
    options = {spec.name for spec in leading if spec.keyword_only}
    if not options:  # nothing more to consume: never lex the text
        return tokens, stream.read().lstrip()
    return _take_options(lexer, stream, options, tokens)


def _take_positionals(
    lexer: shlex.shlex, leading: tuple[_CommandArgSpec, ...]
) -> tuple[list[str], bool]:
    """Consume tokens until every positional of *leading* is bound, by position or name.

    Returns:
        The tokens consumed, and whether every positional was bound before the
        line ended.
    """
    names = {spec.name for spec in leading}
    unbound = [spec.name for spec in leading if not spec.keyword_only]
    tokens: list[str] = []
    while unbound:
        token = lexer.get_token()
        if token is None:
            return tokens, False
        tokens.append(token)
        key, sep, _ = token.partition("=")
        if sep and key in names:
            if key in unbound:
                unbound.remove(key)
        else:
            unbound.pop(0)
    return tokens, True


def _take_options(
    lexer: shlex.shlex, stream: StringIO, options: set[str], tokens: list[str]
) -> tuple[list[str], str]:
    """Consume ``name=value`` option tokens; the text starts at the first other token.

    ``mark`` is taken before every read and the stream is sought back to it when
    the text starts, so the text is the raw source — quotes and escapes kept — and
    not the lexed token. A token that cannot be lexed starts the text too.
    """
    while True:
        mark = stream.tell()
        try:
            token = lexer.get_token()
        except ValueError:  # a broken quote: text, not a failure
            token = ""
        if token is None:
            return tokens, ""
        key, sep, _ = token.partition("=")
        if not (sep and key in options):
            stream.seek(mark)
            return tokens, stream.read().lstrip()
        tokens.append(token)


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
            documented = _arg_descriptions(entry.fn)
            args = [
                CommandArg(
                    name=spec.name,
                    type=_json_type_name(spec.adapter),
                    required=spec.required,
                    description=documented.get(spec.name) or None,
                    keyword_only=spec.keyword_only,
                    rest_of_line=spec.rest_of_line,
                    choices=_choices(spec.adapter),
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
        command, ``shlex``-tokenises the rest, classifies those tokens as positional or
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
        A command whose last parameter is :data:`RestOfLine` reads positionals,
        then ``name=value`` options, then free text (see :data:`RestOfLine`).

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
        tokenises only its positionals and options (:func:`_split_leading`), and
        binds the raw remainder to the last by name — so it can never be supplied
        as ``name=value``, and an empty remainder leaves it unbound: its default
        applies, or :meth:`_bind` reports it as a missing argument.
        """
        if not entry.rest_of_line:
            return cls._classify_tokens(entry, shlex.split(rest))
        leading, remainder = _split_leading(rest, entry)
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
        positional token appearing after any keyword token is rejected, and so is
        a keyword given twice.

        With *rest_of_line*, the last parameter is not a keyword candidate: its
        value is the raw remainder, bound by :meth:`_tokenise`.

        Raises:
            ValueError: If a positional token follows a keyword token (names the
                offending positional value), or a keyword repeats (names it).
        """
        specs = entry.args[:-1] if rest_of_line else entry.args
        names = {spec.name for spec in specs}
        positional: list[str] = []
        keyword: dict[str, str] = {}
        for token in args:
            key, sep, value = token.partition("=")
            if sep and key in names:
                if key in keyword:
                    raise ValueError(f"got multiple values for argument '{key}'")
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
        arity over the merged set (every required parameter bound, at most
        ``len(args)`` positionals), then coerces every bound value through its
        per-arg :class:`TypeAdapter`.
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
