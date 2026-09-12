"""FastAPI HTTP 入口：上传、检索问答、文档与 Chunk 观测。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .errors import KnowLedgerError
from .models import Chunk, Document
from .workbench import KnowledgeWorkbench


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class TextIngestRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=20 * 1024 * 1024)


def _document_dict(document: Document) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "media_type": document.media_type,
        "size_bytes": document.size_bytes,
        "text_length": len(document.text),
        "sections": [
            {
                "title": section.title,
                "level": section.level,
                "page": section.page,
                "order": section.order,
            }
            for section in document.sections
        ],
        "metadata": document.metadata,
    }


def _chunk_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "text": chunk.text,
        "section": chunk.section,
        "page": chunk.page,
        "ordinal": chunk.ordinal,
        "metadata": chunk.metadata,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    selected = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.workbench = KnowledgeWorkbench(selected)
        yield

    app = FastAPI(
        title="KnowLedger 企业知识库 RAG 工作台",
        version="0.1.0",
        description="可离线运行的文档解析、混合检索和来源可解释问答服务",
        lifespan=lifespan,
    )

    def workbench(request: Request) -> KnowledgeWorkbench:
        return request.app.state.workbench

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        current = workbench(request)
        return {
            "status": "ok",
            "mode": selected.mode,
            "documents": len(current.documents),
            "chunks": current.retriever.size,
        }

    @app.get("/v1/documents")
    async def list_documents(request: Request) -> dict[str, Any]:
        current = workbench(request)
        return {"documents": [_document_dict(item) for item in current.list_documents()]}

    @app.post("/v1/documents/text")
    async def ingest_text(payload: TextIngestRequest, request: Request) -> dict[str, Any]:
        try:
            document = workbench(request).ingest_bytes(
                payload.filename, payload.text.encode("utf-8")
            )
        except (KnowLedgerError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document": _document_dict(document)}

    @app.post("/v1/documents/upload")
    async def upload_document(file: UploadFile, request: Request) -> dict[str, Any]:
        filename = file.filename or "upload.txt"
        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                raise ValueError("上传文件不能超过 20 MiB")
            document = workbench(request).ingest_bytes(filename, content)
        except (KnowLedgerError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()
        return {"document": _document_dict(document)}

    @app.get("/v1/documents/{document_id}/chunks")
    async def document_chunks(document_id: str, request: Request) -> dict[str, Any]:
        try:
            chunks = workbench(request).get_chunks(document_id)
        except KnowLedgerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"chunks": [_chunk_dict(item) for item in chunks]}

    @app.delete("/v1/documents/{document_id}")
    async def delete_document(document_id: str, request: Request) -> dict[str, Any]:
        try:
            workbench(request).delete_document(document_id)
        except KnowLedgerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": document_id}

    @app.post("/v1/index/rebuild")
    async def rebuild_index(request: Request) -> dict[str, Any]:
        current = workbench(request)
        current.rebuild()
        return {"status": "rebuilt", "chunks": current.retriever.size}

    @app.post("/v1/query")
    async def query(payload: QueryRequest, request: Request) -> dict[str, Any]:
        try:
            response = await workbench(request).ask(payload.query, top_k=payload.top_k)
        except (KnowLedgerError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.to_dict()

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run("knowledger.api:app", host=settings.host, port=settings.port, reload=False)

