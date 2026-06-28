"""
Text Cleaner
============
Normalises raw extracted text before chunking.
Removes artefacts introduced by PDF extraction or DOCX conversion
while preserving meaningful structure (newlines that delimit sections).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and normalises raw document text."""

    # Patterns for noise removal
    _PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _HEADER_FOOTER_RE = re.compile(
        r"(confidential|all rights reserved|www\.\S+|©.*?\d{4})", re.IGNORECASE
    )

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to raw document text.

        Steps:
        1. Normalise line endings.
        2. Remove lone page numbers.
        3. Remove common header/footer boilerplate.
        4. Collapse multiple spaces on the same line.
        5. Collapse excessive blank lines.
        6. Strip leading/trailing whitespace.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text ready for chunking.
        """
        if not text:
            return ""

        text = self._normalise_line_endings(text)
        text = self._remove_page_numbers(text)
        text = self._remove_header_footer_noise(text)
        text = self._collapse_spaces(text)
        text = self._collapse_newlines(text)
        text = text.strip()

        logger.debug("Text cleaned: %d characters remaining.", len(text))
        return text

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_line_endings(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_page_numbers(self, text: str) -> str:
        return self._PAGE_NUMBER_RE.sub("", text)

    def _remove_header_footer_noise(self, text: str) -> str:
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            # Drop very short lines that match footer patterns
            if len(line.strip()) < 60 and self._HEADER_FOOTER_RE.search(line):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _collapse_spaces(self, text: str) -> str:
        lines = text.split("\n")
        return "\n".join(self._MULTI_SPACE_RE.sub(" ", line) for line in lines)

    def _collapse_newlines(self, text: str) -> str:
        return self._MULTI_NEWLINE_RE.sub("\n\n", text)
