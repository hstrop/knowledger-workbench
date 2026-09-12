"""知识库领域模型与 API 返回模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    level: int = 0
    page: int | None = None
    order: int = 0


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    filename: str
    media_type: str
    size_bytes: int
    sections: tuple[Section, ...]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    section: str
    page: int | None
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Evidence:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    fused_score: float
    rank: int

    def source(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "filename": self.chunk.filename,
            "section": self.chunk.section,
            "page": self.chunk.page,
            "vector_score": round(self.vector_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "fused_score": round(self.fused_score, 6),
        }


@dataclass(frozen=True, slots=True)
class QueryResponse:
    answer: str
    mode: str
    query: str
    evidence: tuple[Evidence, ...]
    prefetch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "query": self.query,
            "prefetch_count": self.prefetch_count,
            "sources": [item.source() for item in self.evidence],
        }

