"""FastAPI HTTP 入口：上传、检索问答、文档与 Chunk 观测。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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


DEMO_TEXT = """# 企业协作手册

## 研发流程
需求进入开发前，需要明确负责人、优先级和验收标准。提交评审前应补充单元测试，并在合并请求中说明改动范围与验证方式。

## 报销制度
差旅报销需要在出差结束后三十天内提交，附件应包含发票和行程单。部门负责人完成审批后，财务在五个工作日内处理。

## 远程办公
周三可以远程办公，申请需提前一天在项目看板登记；紧急情况请同步直属负责人。

## 文档维护
内部技术文档需要标注维护人和最后更新时间，接口变更时同步更新 API 示例和兼容性说明。
"""


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
    seed_on_start = settings is None and selected.mode == "offline"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.workbench = KnowledgeWorkbench(selected)
        if seed_on_start and not application.state.workbench.documents:
            application.state.workbench.ingest_bytes(
                "company-handbook-demo.md", DEMO_TEXT.encode("utf-8")
            )
        yield

    app = FastAPI(
        title="KnowLedger 企业知识库 RAG 工作台",
        version="0.1.0",
        description="可离线运行的文档解析、混合检索和来源可解释问答服务",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir / "assets"
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/meta")
    async def meta() -> dict[str, Any]:
        return {
            "name": "KnowLedger",
            "description": "企业知识库检索与证据问答工作台",
            "mode": selected.mode,
            "features": ["文档解析", "混合检索", "Prefetch", "来源溯源", "索引运维"],
        }

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

    @app.post("/v1/demo/seed")
    async def seed_demo(request: Request) -> dict[str, Any]:
        """载入可立即提问的演示手册；相同内容重复载入保持幂等。"""
        document = workbench(request).ingest_bytes("company-handbook-demo.md", DEMO_TEXT.encode("utf-8"))
        return {"document": _document_dict(document), "seeded": True}

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
