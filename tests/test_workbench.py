from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledger.api import create_app
from knowledger.chunking import ChunkingConfig, chunk_document
from knowledger.config import Settings
from knowledger.errors import ConfigurationError, UnsupportedDocumentError
from knowledger.models import Chunk
from knowledger.parsers import parse_bytes
from knowledger.retrieval import HybridRetriever, tokenize
from knowledger.workbench import KnowledgeWorkbench


def make_settings(tmp_path: Path) -> Settings:
    return Settings(workspace=tmp_path / "data")


def test_parse_and_chunk_preserve_markdown_sections() -> None:
    document = parse_bytes(
        "guide.md",
        "# 报销\n\n差旅报销需要在三十天内提交。\n\n## 审批\n\n部门负责人审批。".encode(),
    )

    assert document.filename == "guide.md"
    assert [section.title for section in document.sections] == ["报销", "审批"]
    chunks = chunk_document(document, ChunkingConfig(chunk_size=80, chunk_overlap=10))
    assert chunks
    assert chunks[0].document_id == document.document_id
    assert "报销" in chunks[0].text
    assert chunks[0].chunk_id.startswith(document.document_id[:12])


def test_parse_minimal_docx_and_reject_unknown_suffix() -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>入职须知</w:t></w:r></w:p>
        <w:p><w:r><w:t>请在首日完成账号激活。</w:t></w:r></w:p>
      </w:body>
    </w:document>"""
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    document = parse_bytes("onboarding.docx", content.getvalue())
    assert document.text == "入职须知\n请在首日完成账号激活。"
    assert document.sections[0].title == "入职须知"

    with pytest.raises(UnsupportedDocumentError):
        parse_bytes("data.csv", b"a,b\n1,2")


def test_long_text_chunking_is_bounded_and_stable() -> None:
    text = "# 规范\n\n" + ("这是一个需要被切分的句子。" * 40)
    document = parse_bytes("long.md", text.encode("utf-8"))
    config = ChunkingConfig(chunk_size=64, chunk_overlap=8)
    first = chunk_document(document, config)
    second = chunk_document(document, config)
    assert len(first) > 1
    assert first == second
    assert all(0 < len(item.text) <= config.chunk_size for item in first)


def test_hybrid_retriever_combines_keyword_and_vector_signals() -> None:
    chunks = [
        Chunk("a", "doc", "policy.md", "差旅报销需要在三十天内提交", "报销", None, 0),
        Chunk("b", "doc", "policy.md", "代码评审需要补充单元测试", "研发", None, 1),
    ]
    retriever = HybridRetriever()
    retriever.add(chunks)

    results = retriever.search("报销期限", top_k=2)
    assert results
    assert results[0].chunk.chunk_id == "a"
    assert results[0].bm25_score >= 0
    assert results[0].fused_score > 0
    assert "报" in tokenize("报销")


def test_workbench_snapshot_idempotency_and_query(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = KnowledgeWorkbench(settings)
    content = "# 报销\n\n报销应在三十天内提交。".encode()
    document = first.ingest_bytes("policy.md", content)
    first_size = first.retriever.size
    first.ingest_bytes("policy.md", content)
    assert len(first.list_documents()) == 1
    assert first_size > 0
    assert first.retriever.size == first_size

    response = first.ask_sync("报销应在多久内提交？")
    assert response.prefetch_count >= 1
    assert any(source["document_id"] == document.document_id for source in response.to_dict()["sources"])
    source = response.to_dict()["sources"][0]
    assert {"chunk_id", "filename", "section", "page", "vector_score", "bm25_score", "fused_score"} <= set(source)

    restored = KnowledgeWorkbench(settings)
    assert [item.document_id for item in restored.list_documents()] == [document.document_id]
    assert restored.retriever.size == first_size

    restored.delete_document(document.document_id)
    assert restored.list_documents() == ()
    assert restored.retriever.size == 0


def test_http_api_ingest_query_and_missing_document(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/meta").json()["name"] == "KnowLedger"
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "offline"

        ingest = client.post(
            "/v1/documents/text",
            json={"filename": "handbook.md", "text": "# 值班\n\n周三可以远程办公。"},
        )
        assert ingest.status_code == 200
        document_id = ingest.json()["document"]["document_id"]

        query = client.post("/v1/query", json={"query": "周几可以远程办公？"})
        assert query.status_code == 200
        assert query.json()["sources"]

        chunks = client.get(f"/v1/documents/{document_id}/chunks")
        assert chunks.status_code == 200
        assert chunks.json()["chunks"]

        missing = client.get("/v1/documents/not-found/chunks")
        assert missing.status_code == 404

        upload = client.post(
            "/v1/documents/upload",
            files={
                "file": (
                    "uploaded.md",
                    "# 上传\n\n文件入库成功。".encode(),
                    "text/markdown",
                )
            },
        )
        assert upload.status_code == 200
        assert upload.json()["document"]["filename"] == "uploaded.md"


def test_settings_reject_invalid_overlap() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({"KNOWLEDGER_CHUNK_SIZE": "32", "KNOWLEDGER_CHUNK_OVERLAP": "32"})
