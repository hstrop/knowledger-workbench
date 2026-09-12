"""环境变量配置；默认路径与离线模式都适合本地演示。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .errors import ConfigurationError

RunMode = Literal["offline", "online"]


def _positive_int(name: str, raw: str, *, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须是整数") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} 必须大于等于 {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    mode: RunMode = "offline"
    workspace: Path = Path("knowledger_data")
    dashscope_api_key: str = field(default="", repr=False)
    llm_model: str = "qwen-plus"
    ocr_endpoint: str = ""
    host: str = "127.0.0.1"
    port: int = 8010
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 80

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            source: Mapping[str, str] = os.environ
        else:
            source = env
        mode = source.get("KNOWLEDGER_MODE", "offline").strip().lower()
        if mode not in {"offline", "online"}:
            raise ConfigurationError("KNOWLEDGER_MODE 仅支持 offline/online")
        port = _positive_int("KNOWLEDGER_PORT", source.get("KNOWLEDGER_PORT", "8010"))
        if port > 65535:
            raise ConfigurationError("KNOWLEDGER_PORT 必须不超过 65535")
        chunk_size = _positive_int(
            "KNOWLEDGER_CHUNK_SIZE", source.get("KNOWLEDGER_CHUNK_SIZE", "500"), minimum=32
        )
        overlap = _positive_int(
            "KNOWLEDGER_CHUNK_OVERLAP",
            source.get("KNOWLEDGER_CHUNK_OVERLAP", "80"),
            minimum=0,
        )
        if overlap >= chunk_size:
            raise ConfigurationError("KNOWLEDGER_CHUNK_OVERLAP 必须小于 chunk_size")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            workspace=Path(source.get("KNOWLEDGER_WORKSPACE", "knowledger_data")),
            dashscope_api_key=source.get("DASHSCOPE_API_KEY", "").strip(),
            llm_model=source.get("KNOWLEDGER_LLM_MODEL", "qwen-plus").strip() or "qwen-plus",
            ocr_endpoint=source.get("KNOWLEDGER_OCR_ENDPOINT", "").strip(),
            host=source.get("KNOWLEDGER_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=port,
            top_k=_positive_int("KNOWLEDGER_TOP_K", source.get("KNOWLEDGER_TOP_K", "5")),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    @property
    def effective_mode(self) -> RunMode:
        # online 是显式选择；没有密钥时由工作台回退到离线并在响应中说明。
        return self.mode

