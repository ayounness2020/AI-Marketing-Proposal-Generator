from __future__ import annotations
import logging
import re
import uuid
from dataclasses import dataclass, field
import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def word_count(self):
        return len(self.text.split())


class SectionChunker:

    def __init__(self, section_headers=None, max_chunk_words=config.MAX_CHUNK_TOKENS):
        self._headers = [h.lower() for h in (section_headers or config.SECTION_HEADERS)]
        self._max_words = max_chunk_words
        escaped = [re.escape(h) for h in self._headers]
        pattern = r"(?im)^[ \t]*(" + "|".join(escaped) + r")[ \t]*[\n:—\-]{0,5}"
        self._header_re = re.compile(pattern)

    def chunk(self, text, source_file, document_type):
        sections = self._split_into_sections(text)
        if not sections:
            sections = self._paragraph_fallback(text)

        chunks = []
        for section_name, section_text in sections:
            sub_chunks = self._maybe_split_large_section(section_name, section_text)
            for idx, sub_text in enumerate(sub_chunks):
                part_suffix = f" Part {idx + 1}" if len(sub_chunks) > 1 else ""
                full_section = section_name + part_suffix
                enriched = (
                    "Document: " + source_file + "\n"
                    "Type: " + document_type + "\n"
                    "Section: " + full_section + "\n\n"
                    + sub_text.strip()
                )
                chunk = Chunk(
                    text=enriched,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "source_file": source_file,
                        "document_type": document_type,
                        "section_name": full_section,
                    },
                )
                chunks.append(chunk)

        logger.info("Chunked %s into %d chunks.", source_file, len(chunks))
        return chunks

    def _split_into_sections(self, text):
        matches = list(self._header_re.finditer(text))
        if not matches:
            return []
        sections = []
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("Introduction", preamble))
        for i, match in enumerate(matches):
            section_name = match.group(1).strip().title()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((section_name, body))
        return sections

    def _paragraph_fallback(self, text):
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return [(f"Paragraph {i + 1}", p) for i, p in enumerate(paragraphs)]

    def _maybe_split_large_section(self, section_name, section_text):
        words = section_text.split()
        if len(words) <= self._max_words:
            return [section_text]
        sub_chunks = []
        for i in range(0, len(words), self._max_words):
            sub_chunks.append(" ".join(words[i:i + self._max_words]))
        return sub_chunks
