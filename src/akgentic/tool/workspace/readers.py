"""Binary document reader -- MarkItDown-based extraction with two-pass LLM fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from openai import OpenAI

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


class DocumentReader:
    """MarkItDown-based document reader with optional LLM vision fallback.

    Pass 1: Extract text via ``MarkItDown()`` (no LLM).
    Pass 2 (optional): If Pass 1 yields fewer than 50 non-whitespace characters
    and ``llm_client`` is configured, retry with ``MarkItDown(llm_client=...)``.
    If both passes yield fewer than 50 non-whitespace characters, returns a
    placeholder comment.
    """

    extensions: frozenset[str] = frozenset(
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

    def __init__(
        self,
        llm_client: OpenAI | None = None,
        llm_model: str = "gpt-4o",
    ) -> None:
        self._llm_client = llm_client
        self._llm_model = llm_model

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
            from markitdown import MarkItDown  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                'markitdown not installed. Run: pip install "akgentic-tool[docs]"'
            ) from exc

        suffix = Path(path).suffix or ".bin"

        # Pass 1: plain MarkItDown (no LLM)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            md = MarkItDown()
            result = md.convert(tmp.name)
            text = result.text_content or ""

        # Pass 2: LLM vision fallback if Pass 1 yielded insufficient content
        if len("".join(text.split())) < 50 and self._llm_client is not None:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp2:
                tmp2.write(content)
                tmp2.flush()
                md_vision = MarkItDown(
                    llm_client=self._llm_client, llm_model=self._llm_model
                )
                result2 = md_vision.convert(tmp2.name)
                text = result2.text_content or ""

        if len("".join(text.split())) < 50:
            return "<!-- markitdown: no text extracted -->"

        return text
