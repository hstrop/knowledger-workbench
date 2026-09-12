"""证据预取与回答生成。

离线回答器只根据召回证据生成可解释摘要；在线 Qwen 适配器是可选能力，
没有密钥或依赖时不会伪装成在线模型。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .config import Settings
from .errors import ConfigurationError
from .models import Evidence


def build_evidence_context(evidence: Iterable[Evidence]) -> str:
    parts: list[str] = []
    for item in evidence:
        source = item.source()
        parts.append(
            f"[来源 {source['chunk_id']}] 文件={source['filename']} "
            f"章节={source['section']} 页码={source['page'] or '-'}\n{item.chunk.text}"
        )
    return "\n\n".join(parts)


def offline_answer(query: str, evidence: tuple[Evidence, ...]) -> str:
    if not evidence:
        return "未在当前知识库召回足够相关内容。请尝试补充关键词，或先上传相关文档。"
    lines = [f"离线证据摘要（问题：{query}）："]
    for item in evidence[:3]:
        text = " ".join(item.chunk.text.split())
        if len(text) > 220:
            text = text[:220].rstrip() + "…"
        lines.append(f"- [{item.chunk.chunk_id}] {text}")
    lines.append("以上内容来自检索到的文档片段；请结合来源字段核查原文。")
    return "\n".join(lines)


@dataclass(slots=True)
class AnswerGenerator:
    settings: Settings

    @property
    def online_available(self) -> bool:
        return self.settings.mode == "online" and bool(self.settings.dashscope_api_key)

    async def generate(self, query: str, evidence: tuple[Evidence, ...]) -> tuple[str, str]:
        if not self.online_available:
            mode = "offline" if self.settings.mode == "offline" else "offline_fallback"
            return offline_answer(query, evidence), mode
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
        except ImportError as exc:
            raise ConfigurationError(
                "在线回答需要可选依赖 langchain-community，请执行 pip install -e '.[online]'"
            ) from exc

        prompt = (
            "你是企业知识库助手。只使用给定证据回答问题；如果证据不足，明确说不知道。"
            "每个关键结论后附 [来源 chunk_id]，不要虚构页码或文件名。\n\n"
            f"问题：{query}\n\n证据：\n{build_evidence_context(evidence)}"
        )
        model = ChatTongyi(
            model=self.settings.llm_model,
            dashscope_api_key=self.settings.dashscope_api_key,
            temperature=0,
        )
        try:
            result = await model.ainvoke(prompt)
            content = getattr(result, "content", result)
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            answer = str(content).strip()
            if answer:
                return answer, "online"
        except Exception:  # noqa: BLE001 - online provider failures must fall back safely
            # 对外不泄露请求内容或密钥；保留确定性结果，方便本地演示。
            return offline_answer(query, evidence), "offline_fallback"
        return offline_answer(query, evidence), "offline_fallback"
