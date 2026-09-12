"""可选 OCR HTTP 适配器。

该模块只定义客户端边界，不包含 OCR 引擎；未设置 endpoint 时不会被工作台调用。
"""

from __future__ import annotations

from pathlib import Path

from ..errors import ConfigurationError


class HttpOcrAdapter:
    def __init__(self, endpoint: str, *, timeout: float = 15.0) -> None:
        endpoint = endpoint.strip()
        if not endpoint.startswith(("http://", "https://")):
            raise ConfigurationError("OCR endpoint 必须是 http(s) URL")
        self.endpoint = endpoint
        self.timeout = timeout

    def extract(self, *, source_path: Path, page_numbers: list[int] | None = None) -> str | None:
        try:
            import httpx
        except ImportError as exc:
            raise ConfigurationError("OCR HTTP 适配器需要 httpx，请执行 pip install -e '.[ocr]'") from exc
        with httpx.Client(timeout=self.timeout) as client:
            with source_path.open("rb") as stream:
                response = client.post(
                    self.endpoint,
                    files={"file": (source_path.name, stream, "application/pdf")},
                    data={"page_numbers": ",".join(str(item) for item in page_numbers or [])},
                )
            response.raise_for_status()
            payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) else None

