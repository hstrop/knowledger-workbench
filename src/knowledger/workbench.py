"""知识库工作台：入库、切分、预检索、回答、删除与状态持久化。"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .answering import AnswerGenerator
from .chunking import ChunkingConfig, chunk_document
from .config import Settings
from .errors import DocumentNotFoundError
from .models import Chunk, Document, QueryResponse
from .parsers import OcrAdapter, parse_bytes, parse_file
from .retrieval import HybridRetriever


def _section_to_dict(section: Any) -> dict[str, Any]:
    return {
        "title": section.title,
        "level": section.level,
        "page": section.page,
        "order": section.order,
    }


class KnowledgeWorkbench:
    """默认内存检索器 + JSON 元数据快照，适合本地 demo 和测试。"""

    def __init__(self, settings: Settings | None = None, *, ocr: OcrAdapter | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.workspace = self.settings.workspace.expanduser()
        self.state_path = self.workspace / "index.json"
        self.ocr = ocr
        self.retriever = HybridRetriever()
        self.documents: dict[str, Document] = {}
        self.answerer = AnswerGenerator(self.settings)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            for item in payload.get("documents", []):
                sections = tuple(
                    __import__("knowledger.models", fromlist=["Section"]).Section(
                        title=section["title"],
                        level=section.get("level", 0),
                        page=section.get("page"),
                        order=section.get("order", 0),
                    )
                    for section in item.get("sections", [])
                )
                document = Document(
                    document_id=item["document_id"],
                    filename=item["filename"],
                    media_type=item["media_type"],
                    size_bytes=item["size_bytes"],
                    sections=sections,
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                )
                self.documents[document.document_id] = document
                self.retriever.add(chunk_document(document, self._chunking_config()))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # 损坏的本地快照不应让离线 HTTP 服务崩溃；用户可通过 rebuild 清空。
            self.documents.clear()
            self.retriever = HybridRetriever()

    def _chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(self.settings.chunk_size, self.settings.chunk_overlap)

    def _save_state(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "media_type": document.media_type,
                    "size_bytes": document.size_bytes,
                    "sections": [_section_to_dict(section) for section in document.sections],
                    "text": document.text,
                    "metadata": document.metadata,
                }
                for document in self.documents.values()
            ],
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=".knowledger-", suffix=".json", dir=self.workspace)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def ingest_bytes(self, filename: str, content: bytes) -> Document:
        document = parse_bytes(filename, content, ocr=self.ocr)
        self.retriever.remove_document(document.document_id)
        self.documents[document.document_id] = document
        self.retriever.add(chunk_document(document, self._chunking_config()))
        self._save_state()
        return document

    def ingest_file(self, path: Path) -> Document:
        document = parse_file(path, ocr=self.ocr)
        self.retriever.remove_document(document.document_id)
        self.documents[document.document_id] = document
        self.retriever.add(chunk_document(document, self._chunking_config()))
        self._save_state()
        return document

    def list_documents(self) -> tuple[Document, ...]:
        return tuple(sorted(self.documents.values(), key=lambda item: item.filename.casefold()))

    def get_chunks(self, document_id: str) -> tuple[Chunk, ...]:
        if document_id not in self.documents:
            raise DocumentNotFoundError(document_id)
        return self.retriever.chunks_for_document(document_id)

    def delete_document(self, document_id: str) -> None:
        if document_id not in self.documents:
            raise DocumentNotFoundError(document_id)
        self.documents.pop(document_id)
        self.retriever.remove_document(document_id)
        self._save_state()

    def rebuild(self) -> None:
        self.retriever = HybridRetriever()
        for document in self.documents.values():
            self.retriever.add(chunk_document(document, self._chunking_config()))
        self._save_state()

    def prefetch(self, query: str, *, top_k: int | None = None):
        return self.retriever.search(query, top_k=top_k or self.settings.top_k)

    async def ask(self, query: str, *, top_k: int | None = None) -> QueryResponse:
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        evidence = self.prefetch(query, top_k=top_k)
        answer, mode = await self.answerer.generate(query, evidence)
        return QueryResponse(answer, mode, query, evidence, len(evidence))

    def ask_sync(self, query: str, *, top_k: int | None = None) -> QueryResponse:
        return asyncio.run(self.ask(query, top_k=top_k))

