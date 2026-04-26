"""
报告生成模块

将各 Sub-Agent 的分析结论汇总，由 LLM 润色整合为标准 Markdown 研报格式，
并持久化保存到 reports/ 目录。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL, REPORTS_DIR
from src.agents.state import AgentState, NODE_END


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class ReportMetadata(BaseModel):
    """研报元数据"""
    title: str = Field(..., description="研报标题")
    generated_at: str = Field(..., description="生成时间（ISO 格式）")
    output_path: str = Field(..., description="保存路径")
    word_count: int = Field(..., description="报告字数")


# 研报生成系统提示词
_REPORT_SYSTEM_PROMPT = """你是一位专业的金融研报撰写专家，曾在顶级券商研究所工作。

请将以下多位专家的分析意见整合为一份完整的投资研究报告，输出标准 Markdown 格式。

研报必须包含以下章节（按顺序）：
1. **执行摘要**：50字以内的核心结论和投资建议
2. **公司/行业概况**：基本信息和近期动态（结合实时数据）
3. **财务分析**：盈利能力、成长性、现金流、估值（来自财务分析专家）
4. **风险评估**：主要风险因素及应对建议（来自风控专家）
5. **投资结论**：综合评级、目标价区间（如有数据）、建议持仓比例

写作要求：
- 语言专业、客观，避免夸大性表述
- 数据引用须注明来源（如"根据2023年报"）
- 各章节逻辑连贯，前后呼应
- 结论必须有据可查，不得凭空捏造数据
"""


class ReportGenerator:
    """
    研报生成器

    作为 LangGraph 的一个节点，汇总所有 Sub-Agent 的输出，
    调用 LLM 生成最终研报，并保存到本地文件。
    """

    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.4,  # 适度发挥，保持专业文风
        )

    def __call__(self, state: AgentState) -> AgentState:
        """
        报告生成节点入口（供 LangGraph 节点调用）

        汇总 state 中各 Agent 的输出，生成并保存最终研报。
        """
        query = state.get("user_query", "")
        print(f"[报告生成] 开始整合分析结果，生成研报...")

        # rag_contexts: list from main_enhanced.py
        # rag_context: string field in AgentState (from graph.py path)
        # Handle both cases for compatibility
        rag_contexts = state.get("rag_contexts") or state.get("rag_context") or []
        if isinstance(rag_contexts, list):
            rag_context = "\n".join(rag_contexts)
        else:
            rag_context = str(rag_contexts)

        # 汇总各 Agent 输出
        report_md = self._generate(
            query=query,
            financial_analysis=state.get("financial_analysis") or "",
            risk_assessment=state.get("risk_assessment") or "",
            realtime_data=state.get("realtime_data") or "",
            rag_context=rag_context,
        )

        # 保存到文件
        metadata = self._save(query, report_md)
        print(f"[报告生成] 研报已保存：{metadata.output_path}（{metadata.word_count} 字）")

        return {
            **state,
            "final_report": report_md,
            "next_agent": NODE_END,
        }

    def generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        报告生成入口（供 main_enhanced.py 直接调用）

        接收普通字典而非 AgentState，输出普通字典。
        """
        query = state.get("user_query", "") or state.get("query", "")
        print(f"[报告生成] 开始整合分析结果，生成研报...")

        rag_contexts = state.get("rag_contexts") or []
        if isinstance(rag_contexts, list):
            rag_context = "\n".join(rag_contexts[:3])  # 最多3个片段
        else:
            rag_context = str(rag_contexts)

        financial_analysis = state.get("financial_analysis") or ""
        if not financial_analysis and state.get("final_answer"):
            financial_analysis = state.get("final_answer")

        report_md = self._generate(
            query=query,
            financial_analysis=financial_analysis,
            risk_assessment=state.get("risk_assessment") or "",
            realtime_data=state.get("realtime_data") or "",
            rag_context=rag_context,
        )

        metadata = self._save(query, report_md)
        print(f"[报告生成] 研报已保存：{metadata.output_path}（{metadata.word_count} 字）")

        return {
            "final_report": report_md,
            "output_path": metadata.output_path,
            "word_count": metadata.word_count,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _generate(
        self,
        query: str,
        financial_analysis: str,
        risk_assessment: str,
        realtime_data: str,
        rag_context: str,
    ) -> str:
        """调用 LLM 整合各模块输出，生成完整研报"""

        # 构建用户提示词，将各模块结论结构化输入
        sections = [f"## 用户研究需求\n{query}\n"]

        if realtime_data:
            sections.append(f"## 实时市场数据\n{realtime_data}\n")
        if financial_analysis:
            sections.append(f"## 财务分析专家意见\n{financial_analysis}\n")
        if risk_assessment:
            sections.append(f"## 风控专家意见\n{risk_assessment}\n")
        if rag_context:
            # 仅传入前 2000 字符，避免超出上下文窗口
            truncated = rag_context[:2000] + ("..." if len(rag_context) > 2000 else "")
            sections.append(f"## 历史财报参考片段（节选）\n{truncated}\n")

        user_prompt = (
            "\n---\n".join(sections)
            + "\n\n请基于以上所有信息，生成完整的投资研究报告。"
        )

        messages = [
            SystemMessage(content=_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self._llm.invoke(messages)
        return response.content.strip()

    def _save(self, query: str, report_md: str) -> ReportMetadata:
        """将研报保存为 Markdown 文件"""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 用查询前 20 字生成文件名，过滤非法字符
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", query[:20]).strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_name}.md"
        output_path = REPORTS_DIR / filename

        # 在文件头部追加元数据注释
        generated_at = datetime.now().isoformat()
        header = (
            f"---\n"
            f"generated_at: {generated_at}\n"
            f"query: {query}\n"
            f"---\n\n"
        )
        output_path.write_text(header + report_md, encoding="utf-8")

        return ReportMetadata(
            title=safe_name,
            generated_at=generated_at,
            output_path=str(output_path),
            word_count=len(report_md),
        )
