"""
财务分析 Agent

职责：基于 RAG 检索到的财报上下文，进行专业财务解读。
分析维度：盈利能力、成长性、现金流、估值水平。

注意：当前实现为单次 LLM 调用，并非完整的 ReAct 循环。
如果上下文不足，会输出警告而非触发补充检索。
如需完整的 ReAct（自我审查 + 补充检索），需要重构此模块。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.agents.state import AgentState, NODE_SUPERVISOR

# 财务分析专家系统提示词
_FINANCIAL_SYSTEM_PROMPT = """你是一位拥有 CFA 资质的资深财务分析师，擅长 A 股上市公司财报解读。

分析时请严格遵循以下框架：
1. **盈利能力**：毛利率、净利润率、ROE 的趋势变化及原因
2. **成长性**：营收、净利润的同比/环比增速，未来增长驱动力
3. **现金流**：经营性现金流与净利润的匹配程度（质量判断）
4. **估值水平**：PE/PB 历史分位，与同行业对比

输出要求：
- 语言简洁专业，数据有据可查（引用参考片段编号）
- 明确指出亮点和风险点
- 结尾给出综合评级：强烈推荐 / 推荐 / 中性 / 回避
"""


class FinancialAgent:
    """
    财务分析 Agent

    输入：state.rag_context（检索到的财报片段）
    输出：state.financial_analysis（结构化财务分析报告）
    """

    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.3,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """财务分析 Agent 节点入口"""
        query = state.get("user_query", "")
        context = state.get("rag_context", "")
        print(f"[财务分析 Agent] 开始分析：{query[:50]}...")

        if not context:
            analysis = "【警告】未获取到相关财报数据，无法进行财务分析。请先确保 PDF 已入库。"
            print("[财务分析 Agent] 无检索上下文，跳过分析")
        else:
            analysis = self._analyze(query, context)
            print(f"[财务分析 Agent] 分析完成，输出长度：{len(analysis)} 字符")

        return {
            **state,
            "financial_analysis": analysis,
            "next_agent": NODE_SUPERVISOR,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _analyze(self, query: str, context: str) -> str:
        """调用 LLM 进行财务分析"""
        user_prompt = (
            f"## 用户问题\n{query}\n\n"
            f"## 参考财报片段\n{context}\n\n"
            "请基于以上财报数据，按分析框架撰写专业财务分析。"
        )
        messages = [
            SystemMessage(content=_FINANCIAL_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()
