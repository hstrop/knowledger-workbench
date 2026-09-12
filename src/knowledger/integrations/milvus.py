"""可选 Milvus 向量层边界。

核心离线工作台使用确定性哈希向量；本适配器仅负责把已切好的 Chunk 写入/查询
外部 Milvus，具体 collection/schema 可由部署方配置，不在示例中硬编码凭据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ConfigurationError
from ..models import Chunk


@dataclass(frozen=True, slots=True)
class MilvusSettings:
    uri: str
    collection: str = "knowledger_chunks"
    token: str = ""


class OptionalMilvusStore:
    """延迟导入 pymilvus 的最小适配器，避免离线安装和启动依赖 Milvus。"""

    def __init__(self, settings: MilvusSettings) -> None:
        if not settings.uri:
            raise ConfigurationError("Milvus URI 不能为空")
        self.settings = settings
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise ConfigurationError("Milvus 适配器需要 pymilvus，请执行 pip install -e '.[milvus]'") from exc
        kwargs: dict[str, Any] = {"uri": settings.uri}
        if settings.token:
            kwargs["token"] = settings.token
        self.client = MilvusClient(**kwargs)

    def describe(self) -> dict[str, Any]:
        return {"uri": self.settings.uri, "collection": self.settings.collection}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """保留接入点；生产 schema/embedding 应按实际部署建立。"""
        if not chunks:
            return
        raise NotImplementedError(
            "请按部署的 embedding 维度建立 Milvus schema 后实现 add_chunks；离线索引不依赖此适配器"
        )

