"""原生文档解析器：PDF、DOCX、Markdown 与纯文本。

解析器只负责恢复可观察的文本和章节元数据；OCR 是单独的可选边界，
不会在没有适配器时伪造识别结果。
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

from .errors import ParseError, UnsupportedDocumentError
from .models import Document, Section

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".md", ".markdown", ".txt"})
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_XML_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class OcrAdapter(Protocol):
    """可选 OCR 边界；实现方需明确返回识别文本或 ``None``。"""

    def extract(self, *, source_path: Path, page_numbers: list[int] | None = None) -> str | None:
        ...


def _document_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(filename.encode("utf-8") + b"\0" + content).hexdigest()
    return digest[:20]


def _normalise_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    # 保留段落边界，但去掉成片空行，便于后续结构化切分。
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                compact.append("")
            blank = True
        else:
            compact.append(line)
            blank = False
    return "\n".join(compact).strip()


def _sections_from_markdown(text: str) -> tuple[Section, ...]:
    sections: list[Section] = []
    order = 0
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            sections.append(Section(match.group(2).strip(), len(match.group(1)), None, order))
            order += 1
    return tuple(sections or (Section("正文", 0, None, 0),))


def _sections_from_plain_text(text: str) -> tuple[Section, ...]:
    sections: list[Section] = []
    order = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 教学资料常以短行标题/冒号结尾；仅对很短的行启用启发式，避免误切正文。
        if len(stripped) <= 80 and (stripped.endswith((":", "：")) or (len(stripped) <= 32 and not stripped.endswith(("。", ".", "!", "！", "?", "？")))):
            sections.append(Section(stripped.rstrip(":："), 1, None, order))
            order += 1
    return tuple(sections or (Section("正文", 0, None, 0),))


def _read_docx(path: Path) -> tuple[str, tuple[Section, ...]]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise ParseError(f"DOCX 读取失败：{path.name}") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ParseError(f"DOCX XML 解析失败：{path.name}") from exc

    blocks: list[tuple[str, str | None]] = []
    for paragraph in root.findall(".//w:body/w:p", _XML_NS):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", _XML_NS)]
        value = "".join(texts).strip()
        if not value:
            continue
        style = paragraph.find("./w:pPr/w:pStyle", _XML_NS)
        style_name = style.get(f"{{{_XML_NS['w']}}}val") if style is not None else None
        blocks.append((value, style_name))
    # 表格使用行列文本拼接，保留为结构化块而非悄悄丢弃。
    for table in root.findall(".//w:body/w:tbl", _XML_NS):
        rows: list[str] = []
        for row in table.findall("./w:tr", _XML_NS):
            cells = []
            for cell in row.findall("./w:tc", _XML_NS):
                cells.append(" ".join(node.text or "" for node in cell.findall(".//w:t", _XML_NS)).strip())
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(("\n".join(rows), "table"))

    lines = [value for value, _ in blocks]
    text = _normalise_lines("\n".join(lines))
    sections: list[Section] = []
    order = 0
    for value, style in blocks:
        if style and (style.lower().startswith("heading") or style.lower() in {"title", "subtitle"}):
            level_match = re.search(r"(\d+)$", style)
            level = int(level_match.group(1)) if level_match else 1
            sections.append(Section(value, level, None, order))
            order += 1
    return text, tuple(sections or (Section("正文", 0, None, 0),))


def _read_pdf(path: Path, ocr: OcrAdapter | None) -> tuple[str, tuple[Section, ...], dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParseError("解析 PDF 需要可选依赖 pypdf，请执行 pip install -e '.[documents]'") from exc
    try:
        reader = PdfReader(str(path))
        page_texts = [_normalise_lines(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf 的异常类型随版本变化，统一为领域错误。
        raise ParseError(f"PDF 读取失败：{path.name}") from exc

    nonempty_pages = [index + 1 for index, value in enumerate(page_texts) if value]
    metadata: dict[str, Any] = {"page_count": len(page_texts), "parser": "pypdf"}
    if not nonempty_pages and ocr is not None:
        ocr_text = ocr.extract(source_path=path, page_numbers=list(range(1, len(page_texts) + 1)))
        if ocr_text:
            page_texts = [_normalise_lines(ocr_text)]
            metadata["ocr_used"] = True
            metadata["ocr_pages"] = list(range(1, len(reader.pages) + 1))
    if not nonempty_pages and not any(page_texts):
        metadata["ocr_required"] = True

    lines: list[str] = []
    sections: list[Section] = []
    order = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        lines.append(page_text)
        # 页首短行作为可用标题提示，同时保留 page 字段作为权威来源。
        first_line = page_text.splitlines()[0].strip()
        title = first_line if len(first_line) <= 100 else f"第 {page_number} 页"
        sections.append(Section(title or f"第 {page_number} 页", 0, page_number, order))
        order += 1
    metadata["text_pages"] = nonempty_pages
    return _normalise_lines("\n\n".join(lines)), tuple(sections or (Section("正文", 0, None, 0),)), metadata


def parse_bytes(filename: str, content: bytes, *, ocr: OcrAdapter | None = None) -> Document:
    """从内存字节解析文档；API 入口可直接复用。"""

    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError(
            f"不支持 {suffix or '无扩展名'}；支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if not path.name or path.name in {".", ".."}:
        raise ParseError("文件名不能为空")

    metadata: dict[str, Any] = {"parser": suffix.lstrip(".")}
    if suffix == ".pdf":
        # pypdf 需要路径；内存 API 通过临时文件隔离，完成后立即删除。
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            text, sections, pdf_metadata = _read_pdf(temporary_path, ocr)
            metadata.update(pdf_metadata)
        finally:
            temporary_path.unlink(missing_ok=True)
    elif suffix == ".docx":
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            text, sections = _read_docx(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        try:
            text = _normalise_lines(content.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ParseError("文本文件必须是 UTF-8 编码") from exc
        sections = _sections_from_markdown(text) if suffix in {".md", ".markdown"} else _sections_from_plain_text(text)

    return Document(
        document_id=_document_id(path.name, content),
        filename=path.name,
        media_type=suffix.lstrip("."),
        size_bytes=len(content),
        sections=tuple(sections),
        text=text,
        metadata=metadata,
    )


def parse_file(path: Path, *, ocr: OcrAdapter | None = None, max_bytes: int = 20 * 1024 * 1024) -> Document:
    path = path.expanduser()
    if not path.is_file():
        raise ParseError(f"文件不存在：{path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ParseError(f"文件不能超过 {max_bytes} 字节")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"文件读取失败：{path.name}") from exc
    return parse_bytes(path.name, content, ocr=ocr)


def parse_text(filename: str, text: str) -> Document:
    if not isinstance(text, str):
        raise ParseError("文本内容必须是字符串")
    return parse_bytes(filename, text.encode("utf-8"))

