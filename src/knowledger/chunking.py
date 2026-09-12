"""保留标题和页码来源的结构化文本切分。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n{2,}")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size < 32:
            raise ValueError("chunk_size 必须至少为 32")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须位于 0..chunk_size-1")


def _stable_chunk_id(document_id: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{ordinal}:{text}".encode()).hexdigest()
    return f"{document_id[:12]}-{ordinal:04d}-{digest[:8]}"


def _split_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= 700:
            units.append(paragraph)
            continue
        units.extend(piece.strip() for piece in _SENTENCE_RE.split(paragraph) if piece.strip())
    return units


def _sliding_text(units: list[str], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > config.chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(unit):
                end = min(start + config.chunk_size, len(unit))
                piece = unit[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(unit):
                    break
                start = max(end - config.chunk_overlap, start + 1)
            continue
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > config.chunk_size:
            chunks.append(current)
            tail = current[-config.chunk_overlap :] if config.chunk_overlap else ""
            current = f"{tail}\n{unit}".strip() if tail else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _section_for_offset(document: Document, line_index: int) -> str:
    # Document 模型保留章节列表；文本格式中标题通常出现在前部，
    # 这里以顺序启发式映射，避免向模型或向量库丢弃可解释的章节名。
    if not document.sections:
        return "正文"
    return document.sections[min(line_index, len(document.sections) - 1)].title


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> tuple[Chunk, ...]:
    config = config or ChunkingConfig()
    if not document.text.strip():
        return ()
    # 按空行建立结构块，同时用块序近似章节序；页码来源由 PDF 的 section metadata 提供。
    paragraphs = [piece.strip() for piece in re.split(r"\n{2,}", document.text) if piece.strip()]
    result: list[Chunk] = []
    ordinal = 0
    for block_index, paragraph in enumerate(paragraphs):
        units = _split_units(paragraph)
        for piece in _sliding_text(units, config):
            section = _section_for_offset(document, block_index)
            page = document.sections[min(block_index, len(document.sections) - 1)].page
            result.append(
                Chunk(
                    chunk_id=_stable_chunk_id(document.document_id, ordinal, piece),
                    document_id=document.document_id,
                    filename=document.filename,
                    text=piece,
                    section=section,
                    page=page,
                    ordinal=ordinal,
                    metadata={
                        "media_type": document.media_type,
                        "source": "local_parse",
                    },
                )
            )
            ordinal += 1
    return tuple(result)
