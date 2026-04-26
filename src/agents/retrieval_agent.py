"""
检索 Agent（RAG Sub-Agent）

职责：接收用户问题，执行 HyDE + Rerank 检索，将相关文档上下文写入共享状态。

注意：当前实现为最多 2 轮顺序检索（首轮 + 一次重试改写），
并非完整的 ReAct 循环（无 action→observation→reasoning→next_action 的迭代）。
如需完整 ReAct，需要重构为真正的循环结构。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.agents.state import AgentState, NODE_SUPERVISOR


# 检索结果字符数低于此阈值时触发问题改写后重试
_MIN_CONTEXT_LENGTH = 200


class RetrievalAgent:
    """
    检索 Agent

    封装完整 RAG Pipeline（HyDE + ChromaDB + Rerank），
    结果格式化为 Markdown 写入 state.rag_context。
    """

    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: src.rag.reranker.RAGPipeline 实例
        """
        self._pipeline = rag_pipeline
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """
        检索 Agent 节点入口

        ReAct 流程：
          1. 用原始问题检索
          2. 若上下文过短 → 改写问题 → 再次检索
          3. 格式化结果写入 state.rag_context
        """
        query = state.get("user_query", "")
        print(f"[检索 Agent] 开始检索：{query[:50]}...")

        # 第一轮检索
        docs = self._pipeline.query(query, use_hyde=True)
        context = self._format_context(docs)

        # ReAct：若结果不足，改写问题重试
        if len(context) < _MIN_CONTEXT_LENGTH:
            print("[检索 Agent] 初次检索结果不足，正在改写问题后重试...")
            rewritten_query = self._rewrite_query(query)
            docs = self._pipeline.query(rewritten_query, use_hyde=True)
            context = self._format_context(docs)
            print(f"[检索 Agent] 改写后问题：{rewritten_query[:60]}")

        print(f"[检索 Agent] 检索完成，上下文长度：{len(context)} 字符")

        return {
            **state,
            "rag_context": context,
            "next_agent": NODE_SUPERVISOR,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _rewrite_query(self, original_query: str) -> str:
        """让 LLM 改写问题，使其更适合向量检索"""
        prompt = (
            "你是一位金融检索专家。请将以下用户问题改写为更适合文档检索的关键词组合，"
            "突出核心财务指标和公司名称，去掉语气词，只输出改写后的问题：\n\n"
            f"原问题：{original_query}"
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _format_context(self, docs: list[dict]) -> str:
        """将检索文档列表格式化为 Markdown 上下文块"""
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            score_str = f"（相关度：{doc.get('score', 0):.3f}）"
            source_str = f"来源：{doc.get('source', '未知')}"
            parts.append(
                f"### 参考片段 {i} {score_str}\n"
                f"> {source_str}\n\n"
                f"{doc['content']}\n"
            )
        return "\n---\n".join(parts)
