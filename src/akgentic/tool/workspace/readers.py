"""Binary document reader -- MarkItDown-based extraction with two-pass LLM fallback."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
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
    and ``llm_client`` is set, lazily constructs ``OpenAI()`` (or ``AzureOpenAI()``)
    and retries.
    If both passes yield fewer than 50 non-whitespace characters, returns a
    placeholder comment.
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

    _openai_client: OpenAI | None = PrivateAttr(default=None)

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

    @staticmethod
    def _convert_via_tempfile(md: Any, content: bytes, suffix: str) -> str:
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
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise ImportError(
                'markitdown not installed. Run: pip install "akgentic-tool[docs]"'
            ) from exc

        suffix = Path(path).suffix or ".bin"

        # Pass 1: plain MarkItDown (no LLM)
        text = self._convert_via_tempfile(MarkItDown(), content, suffix)

        # Pass 2: LLM vision fallback if Pass 1 yielded insufficient content.
        # Only construct the OpenAI client when Pass 1 came up short — otherwise
        # a successful Pass 1 would needlessly require credentials it never uses.
        if len("".join(text.split())) < 50:
            openai_client = self._get_openai_client()
            if openai_client is not None:
                md_vision = MarkItDown(llm_client=openai_client, llm_model=self.llm_model)
                text = self._convert_via_tempfile(md_vision, content, suffix)

        if len("".join(text.split())) < 50:
            return "<!-- markitdown: no text extracted -->"

        return text
