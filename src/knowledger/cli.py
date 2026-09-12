"""KnowLedger 命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .errors import KnowLedgerError
from .workbench import KnowledgeWorkbench


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KnowLedger 企业知识库 RAG 工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="解析并加入本地文档")
    ingest.add_argument("path", type=Path)

    ingest_text = sub.add_parser("ingest-text", help="将一段文本作为文档加入知识库")
    ingest_text.add_argument("filename")
    ingest_text.add_argument("text")

    query = sub.add_parser("query", help="检索证据并生成回答")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=None)

    sub.add_parser("list", help="列出已入库文档")
    chunks = sub.add_parser("chunks", help="查看文档 Chunk")
    chunks.add_argument("document_id")
    delete = sub.add_parser("delete", help="删除文档及其索引")
    delete.add_argument("document_id")
    sub.add_parser("rebuild", help="按当前切分配置重建索引")
    return parser


async def _query(workbench: KnowledgeWorkbench, question: str, top_k: int | None) -> None:
    response = await workbench.ask(question, top_k=top_k)
    print(response.answer)
    print("\n来源：")
    for source in response.to_dict()["sources"]:
        print(json.dumps(source, ensure_ascii=False))


def main() -> None:
    arguments = _build_parser().parse_args()
    workbench = KnowledgeWorkbench(Settings.from_env())
    try:
        if arguments.command == "ingest":
            document = workbench.ingest_file(arguments.path)
            print(json.dumps({"document_id": document.document_id, "filename": document.filename}, ensure_ascii=False))
        elif arguments.command == "ingest-text":
            document = workbench.ingest_bytes(arguments.filename, arguments.text.encode("utf-8"))
            print(json.dumps({"document_id": document.document_id, "filename": document.filename}, ensure_ascii=False))
        elif arguments.command == "query":
            asyncio.run(_query(workbench, arguments.question, arguments.top_k))
        elif arguments.command == "list":
            print(json.dumps([{"document_id": item.document_id, "filename": item.filename} for item in workbench.list_documents()], ensure_ascii=False, indent=2))
        elif arguments.command == "chunks":
            print(json.dumps([{"chunk_id": item.chunk_id, "text": item.text, "section": item.section, "page": item.page} for item in workbench.get_chunks(arguments.document_id)], ensure_ascii=False, indent=2))
        elif arguments.command == "delete":
            workbench.delete_document(arguments.document_id)
            print(f"已删除：{arguments.document_id}")
        elif arguments.command == "rebuild":
            workbench.rebuild()
            print(f"索引已重建，Chunk 数：{workbench.retriever.size}")
    except (KnowLedgerError, ValueError) as exc:
        raise SystemExit(f"操作失败：{exc}") from exc


if __name__ == "__main__":
    main()

