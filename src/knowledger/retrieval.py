"""确定性向量近邻 + 轻量 BM25 + RRF 混合检索。

这里的向量是本地哈希向量，目标是让无密钥环境具备可复现的检索链路；
生产环境可以用 ``integrations/milvus.py`` 替换向量层，而无需改变证据模型。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import Chunk, Evidence

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> tuple[str, ...]:
    """中英文混合轻量分词；中文按字切分以保持离线可用。"""

    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _hashed_vector(text: str, dimension: int = 96) -> tuple[float, ...]:
    values = [0.0] * dimension
    for token in tokenize(text):
        # 稳定的 Python-independent hash，避免不同进程 PYTHONHASHSEED 影响结果。
        digest = __import__("hashlib").sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(values)
    return tuple(value / norm for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right))))


@dataclass(slots=True)
class _Bm25Document:
    chunk: Chunk
    counts: Counter[str]
    length: int


class HybridRetriever:
    """内存索引，暴露和 Milvus 适配器相同的证据输出形状。"""

    def __init__(self, *, rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k 必须为正整数")
        self.rrf_k = rrf_k
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._bm25: dict[str, _Bm25Document] = {}
        self._document_ids: dict[str, set[str]] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = _hashed_vector(chunk.text)
            tokens = tokenize(chunk.text)
            self._bm25[chunk.chunk_id] = _Bm25Document(chunk, Counter(tokens), len(tokens))
            self._document_ids.setdefault(chunk.document_id, set()).add(chunk.chunk_id)

    def remove_document(self, document_id: str) -> None:
        for chunk_id in self._document_ids.pop(document_id, set()):
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
            self._bm25.pop(chunk_id, None)

    def _bm25_scores(self, query_tokens: tuple[str, ...]) -> dict[str, float]:
        if not self._bm25 or not query_tokens:
            return {}
        n = len(self._bm25)
        avgdl = sum(item.length for item in self._bm25.values()) / max(n, 1)
        query_freq = Counter(query_tokens)
        scores: dict[str, float] = {}
        k1, b = 1.5, 0.75
        for chunk_id, item in self._bm25.items():
            score = 0.0
            for token, qf in query_freq.items():
                df = sum(token in candidate.counts for candidate in self._bm25.values())
                if not df:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                frequency = item.counts.get(token, 0)
                denominator = frequency + k1 * (1 - b + b * item.length / max(avgdl, 1))
                score += idf * (frequency * (k1 + 1) / denominator) * min(qf, 3)
            scores[chunk_id] = score
        return scores

    def search(self, query: str, *, top_k: int = 5) -> tuple[Evidence, ...]:
        if not query.strip() or top_k < 1 or not self._chunks:
            return ()
        query_vector = _hashed_vector(query)
        vector_scores = {
            chunk_id: _cosine(query_vector, vector)
            for chunk_id, vector in self._vectors.items()
        }
        bm25_scores = self._bm25_scores(tokenize(query))
        vector_ranked = sorted(vector_scores, key=lambda key: (-vector_scores[key], key))
        bm25_ranked = sorted(
            bm25_scores,
            key=lambda key: (-bm25_scores[key], key),
        )
        vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ranked, 1)}
        bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ranked, 1)}
        candidates = set(vector_rank) | set(bm25_rank)
        fused = {
            chunk_id: 1 / (self.rrf_k + vector_rank.get(chunk_id, len(vector_rank) + 1))
            + 1 / (self.rrf_k + bm25_rank.get(chunk_id, len(bm25_rank) + 1))
            for chunk_id in candidates
        }
        ordered = sorted(candidates, key=lambda key: (-fused[key], key))[:top_k]
        return tuple(
            Evidence(
                chunk=self._chunks[chunk_id],
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                fused_score=fused[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered, 1)
        )

    def chunks_for_document(self, document_id: str) -> tuple[Chunk, ...]:
        ids = self._document_ids.get(document_id, set())
        return tuple(sorted((self._chunks[item] for item in ids), key=lambda item: item.ordinal))
