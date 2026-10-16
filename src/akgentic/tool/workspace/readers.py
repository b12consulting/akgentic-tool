"""Binary document reader -- MarkItDown-based extraction with two-pass LLM fallback."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Self

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from markitdown import MarkItDown
    from openai import OpenAI

_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class MediaContent(BaseModel):
    """In-memory binary image content with MIME type.

    Plain ``BaseModel`` (NOT ``SerializableBaseModel``) — in-memory only,
    never wire-serialized.  No pydantic-ai imports — framework-agnostic by design.
    """

    data: bytes
    media_type: str


logger = logging.getLogger(__name__)

LlmClient = Literal["openai", "azure"]
"""The OpenAI-compatible clients this reader can build. MarkItDown only calls
``client.chat.completions.create``, which both ``OpenAI`` and ``AzureOpenAI`` serve."""

_DEFAULT_LLM_CLIENT: LlmClient = "openai"
_DEFAULT_LLM_MODEL = "gpt-6-luna"
LLM_CLIENT_ENV = "AKGENTIC_DOCUMENT_READER_PROVIDER"
LLM_MODEL_ENV = "AKGENTIC_DOCUMENT_READER_MODEL"
"""Environment variables a deployment sets to override the reader's defaults.

The tool keeps its own defaults; a deployment sets these names to override them
(akgentic-infra's worker settings read the same two), and an empty value counts
as unset."""

_CLIENT_FOR_PROVIDER: dict[str, LlmClient] = {
    "openai": "openai",
    "openai-chat": "openai",
    "azure": "azure",
    "azure-chat": "azure",
}
"""``akgentic.llm.ModelConfig`` provider ids mapped to the client that serves them, so
one provider setting can drive both an agent model and this reader."""


def _default_llm_client() -> LlmClient:
    """``AKGENTIC_DOCUMENT_READER_PROVIDER`` mapped to a client, else the tool default.

    A provider this reader cannot serve (``anthropic``, ``google-gla``…) is logged
    and ignored rather than failing every read.
    """
    value = os.environ.get(LLM_CLIENT_ENV)
    if not value:
        return _DEFAULT_LLM_CLIENT
    client = _CLIENT_FOR_PROVIDER.get(value)
    if client is None:
        logger.warning(
            "%s=%r is not a provider the document reader supports; using %r",
            LLM_CLIENT_ENV,
            value,
            _DEFAULT_LLM_CLIENT,
        )
        return _DEFAULT_LLM_CLIENT
    return client


def _default_llm_model() -> str:
    """``AKGENTIC_DOCUMENT_READER_MODEL`` when set, else the tool default."""
    return os.environ.get(LLM_MODEL_ENV) or _DEFAULT_LLM_MODEL


DOCINTEL_ENDPOINT_ENV = "AKGENTIC_DOCUMENT_READER_DOCINTEL_ENDPOINT"
DOCINTEL_KEY_ENV = "AKGENTIC_DOCUMENT_READER_DOCINTEL_KEY"
"""Azure Document Intelligence for Pass 2, which OCRs scanned PDFs (MarkItDown's PDF converter
ignores the LLM client). The key is read when Pass 2 is first built and never serialised; without
it the reader authenticates with ``DefaultAzureCredential`` (managed identity)."""


def _markitdown_class() -> type[MarkItDown]:
    """Import ``MarkItDown`` on first use; ``markitdown`` is an optional extra.

    Raises:
        ImportError: If ``markitdown`` is not installed.
    """
    try:
        from markitdown import MarkItDown as _MarkItDown  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            'markitdown not installed. Run: pip install "akgentic-tool[docs]"'
        ) from exc
    return _MarkItDown


TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".html",
        ".xml",
        ".rst",
        ".cfg",
        ".ini",
        ".log",
    }
)


class FileTypeReader(Protocol):
    """Protocol for file type readers that extract text from binary content."""

    extensions: frozenset[str]

    def extract_text(self, content: bytes, path: str) -> str:
        """Extract text content from binary file bytes.

        Args:
            content: Raw bytes of the file.
            path: Original file path (used for suffix detection).

        Returns:
            Extracted text as a string.
        """
        ...


class DocumentReader(BaseModel):
    """MarkItDown-based document reader, fully Pydantic-serializable.

    Pass 1: Extract text via ``MarkItDown()`` (no LLM).
    Pass 2 (optional): If Pass 1 yields fewer than 50 non-whitespace characters
    and ``llm_client`` or ``docintel_endpoint`` is set, retries with a lazily
    constructed ``OpenAI()`` (or ``AzureOpenAI()``) and/or Azure Document Intelligence.
    If both passes yield fewer than 50 non-whitespace characters, returns a
    placeholder comment.

    Each pass's ``MarkItDown`` is built on first use and reused for the reader's
    lifetime: construction loads a Magika ONNX model into native memory the process
    does not give back, so building one per extraction grows a worker without bound.
    """

    extensions: ClassVar[frozenset[str]] = frozenset(
        {
            ".pdf",
            ".docx",
            ".xlsx",
            ".xls",
            ".pptx",
            ".msg",
            ".epub",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
        }
    )

    llm_client: LlmClient | None = Field(default_factory=_default_llm_client)
    llm_model: str = Field(default_factory=_default_llm_model)
    docintel_endpoint: str | None = Field(
        default_factory=lambda: os.environ.get(DOCINTEL_ENDPOINT_ENV) or None
    )

    _openai_client: OpenAI | None = PrivateAttr(default=None)
    _markitdown: MarkItDown | None = PrivateAttr(default=None)
    _markitdown_pass2: MarkItDown | None = PrivateAttr(default=None)
    _converter_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:  # BaseModel's signature
        """Copy the configuration; the copy starts with empty caches and builds its own.

        The cached clients and converters hold native sessions, and the lock cannot be
        copied at all, so the default deep copy of private attributes would fail.
        ``model_construct`` gives every private attribute its default.
        """
        return self.model_construct(
            _fields_set=set(self.model_fields_set), **deepcopy(self.__dict__, memo)
        )

    def __copy__(self) -> Self:
        """Copy the configuration shallowly; the copy, too, starts with empty caches.

        ``model_copy(update=...)`` copies through here and then changes fields, so a copy
        that kept the source's converters would go on using the configuration they were
        built from.
        """
        return self.model_construct(_fields_set=set(self.model_fields_set), **self.__dict__)

    def __eq__(self, other: object) -> bool:
        """Readers are equal when their configuration is; the caches and lock do not count.

        ``BaseModel`` also compares private attributes, which would make two fresh readers
        differ by their locks and a used reader differ from its own configuration.
        """
        if not isinstance(other, BaseModel):
            return NotImplemented
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def _get_markitdown(self) -> MarkItDown:
        """The Pass 1 converter (no LLM), built once under the lock."""
        if self._markitdown is None:
            with self._converter_lock:
                if self._markitdown is None:
                    self._markitdown = _markitdown_class()()
        return self._markitdown

    def _get_markitdown_pass2(self) -> MarkItDown | None:
        """The Pass 2 converter, built once under the lock; None when Pass 2 is not configured.

        Its credentials are captured at construction, so a rotated
        ``AKGENTIC_DOCUMENT_READER_DOCINTEL_KEY`` reaches only a new reader.
        """
        if self._markitdown_pass2 is None:
            with self._converter_lock:
                if self._markitdown_pass2 is None:
                    openai_client = self._get_openai_client()
                    if openai_client is None and not self.docintel_endpoint:
                        return None
                    self._markitdown_pass2 = _markitdown_class()(
                        llm_client=openai_client,
                        llm_model=self.llm_model,
                        **self._docintel_kwargs(),
                    )
        return self._markitdown_pass2

    def _get_openai_client(self) -> OpenAI | None:
        """Lazily create and cache the client ``llm_client`` names.

        ``azure`` builds ``AzureOpenAI()``, which reads ``AZURE_OPENAI_ENDPOINT``,
        ``AZURE_OPENAI_API_KEY`` and ``OPENAI_API_VERSION`` — the variables
        akgentic-llm's ``azure`` provider already uses; ``llm_model`` is then the
        Azure deployment name. Returns None if ``llm_client`` is not set.
        """
        if self.llm_client is None:
            return None
        if self._openai_client is None:
            if self.llm_client == "azure":
                from openai import AzureOpenAI  # noqa: PLC0415

                self._openai_client = AzureOpenAI()
            else:
                from openai import OpenAI as _OpenAI  # noqa: PLC0415

                self._openai_client = _OpenAI()
        return self._openai_client

    def _docintel_kwargs(self) -> dict[str, Any]:
        """MarkItDown's Document Intelligence arguments, empty when no endpoint is set.

        The credential is explicit because MarkItDown otherwise takes ``AZURE_API_KEY``, a name
        hosts already use for Azure OpenAI.
        """
        if not self.docintel_endpoint:
            return {}
        from azure.core.credentials import AzureKeyCredential  # noqa: PLC0415
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415

        key = os.environ.get(DOCINTEL_KEY_ENV)
        return {
            "docintel_endpoint": self.docintel_endpoint,
            "docintel_credential": AzureKeyCredential(key) if key else DefaultAzureCredential(),
        }

    @staticmethod
    def _convert_via_tempfile(md: MarkItDown, content: bytes, suffix: str) -> str:
        """Write content to a temp file, convert via MarkItDown, and clean up.

        Uses ``delete=False`` to avoid Windows file-locking issues when
        MarkItDown re-opens the file by name.

        Args:
            md: A ``MarkItDown`` instance (plain or LLM-enabled).
            content: Raw bytes to write.
            suffix: File suffix for the temp file (e.g. ".pdf").

        Returns:
            Extracted text content, or empty string if None.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()
            result = md.convert(tmp.name)
            return result.text_content or ""
        finally:
            os.unlink(tmp.name)

    def extract_text(self, content: bytes, path: str) -> str:
        """Extract text from binary file content using MarkItDown.

        Args:
            content: Raw bytes of the file.
            path: Original file path (used for suffix detection).

        Returns:
            Extracted Markdown text, or a placeholder comment if extraction
            yields no meaningful content.

        Raises:
            ImportError: If ``markitdown`` is not installed.
        """
        suffix = Path(path).suffix or ".bin"

        # Pass 1: plain MarkItDown (no LLM)
        text = self._convert_via_tempfile(self._get_markitdown(), content, suffix)

        # Pass 2: LLM vision fallback if Pass 1 yielded insufficient content.
        # Only build the Pass 2 converter when Pass 1 came up short — otherwise
        # a successful Pass 1 would needlessly require credentials it never uses.
        if len("".join(text.split())) < 50:
            md_vision = self._get_markitdown_pass2()
            if md_vision is not None:
                text = self._convert_via_tempfile(md_vision, content, suffix)

        if len("".join(text.split())) < 50:
            return "<!-- markitdown: no text extracted -->"

        return text
