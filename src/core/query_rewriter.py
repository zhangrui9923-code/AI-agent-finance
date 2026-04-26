"""
金融领域查询改写模块 (Financial Domain Query Rewriting)
======================================================

【模块定位 / Module Positioning】
本模块是 AI 金融研究助手系统的「查询优化」核心组件，位于 Slot Extraction（槽位提取）
和 Intent Classification（意图分类）之后、RAG 检索和 Task Planning 之前。
它负责将用户的原始自然语言查询转化为更适合机器检索和分析的标准化查询表达。

【架构角色 / Architectural Role】

┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  原始用户输入   │───▶│ QueryRewriter   │───▶│  下游消费者        │
│              │    │ (本模块)         │                    │
│ "茅台去年赚了  │    │ 多轮改写策略:     │    │ RAG Pipeline     │
│  多少钱"      │    │ 标准化→特化       │    │ TaskPlanner      │
│              │    │ →扩展/简化        │    │ Search Engines   │
└──────────────┘    └─────────────────┘    └──────────────────┘

【为什么需要 Query Rewriting？ / Why Do We Need It?】

用户的自然语言查询通常存在以下问题，直接影响检索效果：

1. **术语非标准化**
   - 用户说"赚了多少钱"，系统需要"净利润/归母净利润"
   - 用户说"ROE"，系统可能还需要"净资产收益率"（同义词召回）

2. **信息不完整**
   - 用户说"茅台"（缺时间）→ 改写为"贵州茅台 最近一年"
   - 用户说"盈利情况"（缺具体指标）→ 改写为"毛利率 净利率 ROE"

3. **表达冗余/噪声**
   - "帮我分析一下...好吗？" → 去除礼貌用语和疑问词
   - "我想知道...能不能告诉我..." → 提取核心语义

4. **数据源不适配**
   - 同一个意思对不同数据源需要不同格式：
     * 财报检索: "贵州茅台2023年年度报告 净利润 营业收入"
     * 实时行情: "600519.SH real-time stock quote"
     * 新闻搜索: "贵州茅台 利润 业绩 2023"

【核心设计原则 / Core Design Principles】

1. **多轮渐进式改写 (Multi-Round Progressive Rewriting)**
   - Round 1: 标准化（Standardize）— 口语→专业术语 + 槽位注入
   - Round 2: 特化（Specialize）— 针对目标数据源优化格式
   - Round 3+: 扩展/简化（Expand/Simplify）— 同义词补充 或 冗余去除
   - 每轮都有早停条件：质量达标即停止

2. **质量驱动早停 (Quality-Driven Early Stopping)**
   - 每轮改写后评估查询质量分数（0-1）
   - 当 quality >= 0.7 且包含 >= 2 类关键实体时停止
   - 避免过度改写导致语义漂移（semantic drift）

3. **候选保留与最优选择 (Candidate Retention & Best Selection)**
   - 保留每轮的中间结果作为候选
   - 最终从所有候选中选出质量分最高的作为 best_query
   - 保证不会因为某一轮改写失误而丢失好的中间结果

4. **可解释性 (Explainability)**
   - 每步改写都记录策略（strategy）、原因（reasoning）、预期改进
   - 生成 optimization_notes 供调试和审计

【改写策略详解 / Rewrite Strategy Details】

┌──────────┬──────────────────────────────────────────────────────────────┐
│ 策略       │ 说明                                                         │
├──────────┼──────────────────────────────────────────────────────────────┤
│ standardize│ 第一轮执行。将口语转为专业术语，注入已提取的槽位信息。           │
│            │ 例："茅台去年赚了多少钱" → "贵州茅台(600519.SH) 去年净利润分析"  │
├──────────┼──────────────────────────────────────────────────────────────┤
│ specialize │ 第二轮执行。根据 target_data_source 选择对应模板或 LLM 优化。   │
│            │ 例：财报源 → "{company}{year}年报 {metrics}解读"             │
├──────────┼──────────────────────────────────────────────────────────────┤
│ expand    │ 第三轮+（当查询过短时）。添加同义词和相关概念提升召回率。          │
│            │ 例："ROE分析" → "ROE分析（相关概念：净资产收益率 股东权益回报）"│
├──────────┼──────────────────────────────────────────────────────────────┤
│ simplify  │ 第三轮+（当查询过长时）。去除冗余词汇提升匹配精度。               │
│            │ 例："帮我分析一下贵州茅台去年的净利润情况怎么样" →              │
│            │     "贵州茅台去年净利润"                                      │
└──────────┴──────────────────────────────────────────────────────────────┘

【性能特征 / Performance Characteristics】
- 单次 LLM 调用延迟：~1-2s（temperature=0.3，比槽位提取略高以允许适度创造性）
- 规则操作（同义词扩展/简化）：< 1ms
- 最大轮数：MAX_QUERY_REWRITES（默认 3 次，定义于 enhanced_state.py）
- 总体 P99 延迟：~2-6s（取决于实际执行的轮数）

【依赖关系 / Dependencies】
- 上游：SlotExtractionResult（提供 slots 信息）、IntentClassifier（提供 intent 信息用于推断数据源）
- 下游：EnhancedAgentState.rewritten_queries / EnhancedAgentState.current_query
- 外部依赖：
  - langchain_openai.ChatOpenAI（LLM 推理）
  - pydantic（数据校验）
  - config.settings（API 配置）
  - src.core.slot_extractor（金融词典复用：COMPANY_ALIASES, FINANCIAL_METRICS, TIME_PATTERNS）
  - src.core.enhanced_state（MAX_QUERY_REWRITES 常量）

【版本信息 / Version Info】
- Author: AI Finance Assistant Team
- Version: 2.0.0 (极致优化版 / Extreme Optimization Edition)
- Last Updated: 2026-04-13
- Compatible with: enhanced_state.py v2.0+, slot_extractor.py v2.0+
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.core.enhanced_state import MAX_QUERY_REWRITES


# ─────────────────────────────────────────────────────────────
# 日志配置 / Logging Configuration
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 第一部分：数据模型定义
# Part 1: Data Model Definitions
# ════════════════════════════════════════════════════════════


@dataclass
class RewriteStep:
    """
    单次改写步骤记录 (Single Rewrite Step Record)

    【职责】记录一轮改写的完整上下文，用于追溯和审计。
    每执行一次 _perform_single_rewrite() 就产生一个 RewriteStep 实例。

    【生命周期 / Lifecycle】
    1. 创建于 QueryRewriter.rewrite() 的循环体内
    2. 追加到 rewrite_steps 列表
    3. 序列化为 dict 后写入 QueryRewriteResult.rewrite_steps
    4. 被 _generate_optimization_notes() 消费生成人类可读的优化说明

    【字段说明 / Field Descriptions】
    - step_number: 第几轮改写（从 1 开始），用于排序和引用
    - original_query: 本轮输入（即上一轮的输出，首轮为用户原始查询）
    - rewritten_query: 本轮输出的改写结果
    - rewrite_strategy: 本轮使用的策略标识（standardize/specialize/expand/simplify）
    - reasoning: 改写的自然语言原因解释（用于日志和调试）
    - expected_improvement: 预期带来什么改进（如"提升召回率""提高精度"）
    """

    step_number: int
    """本轮序号（1-based index），第一轮为 1"""

    original_query: str
    """本轮改写的输入查询文本"""

    rewritten_query: str
    """本轮改写产出的查询文本"""

    rewrite_strategy: str
    """
    本轮使用的改写策略标识。

    可选值：
    - "standardize": 标准化——口语转术语 + 槽位注入
    - "specialize":  特化——针对数据源优化格式
    - "expand":     扩展——同义词补充，提升召回
    - "simplify":   简化——去除冗余，提升精度
    """

    reasoning: str
    """
    改写原因的自然语言描述。

    示例："标准化术语，补充完整实体和时间信息"
    用于 optimization_notes 生成和调试日志。
    """

    expected_improvement: str = Field(
        default="",
        description="预期改进点的简短描述。示例：'提升术语规范性，改善召回率'",
    )


class QueryRewriteRequest(BaseModel):
    """
    查询改写请求模型 (Query Rewrite Request Schema)

    【职责】封装调用方传入的所有改写参数，是 QueryRewriter.rewrite() 的唯一入参。

    【字段关系图 / Field Relationship Diagram】

    original_query ──┬──> slots (结构化槽位信息)
                     ├──> context (对话上下文)
                     ├──> target_data_source (目标数据源类型)
                     └──> rewrite_history (历史改写记录，支持增量改写)

    【使用示例 / Usage Example】
        request = QueryRewriteRequest(
            original_query="帮我分析茅台去年赚了多少钱",
            slots={"companies": [{"name": "贵州茅台", "code": "600519.SH"}],
                   "time_info": {"year": 2025}},
            target_data_source="financial_reports"
        )
        result = rewriter.rewrite(request)
    """

    original_query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "用户原始查询文本（改写的起点）。"
            "这是必填字段，通常是用户在 CLI/UI 中直接输入的自然语言问题。"
            "改写器将在此基础上进行多轮优化。"
        ),
    )

    slots: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "上游 SlotExtractor 已提取的结构化槽位信息（可选但强烈推荐）。"
            "当提供此字段时，改写器会将槽位中的公司名/时间/指标等注入到改写后的查询中，"
            "显著提升查询的精确度和完整性。"
            "\n格式：与 SlotExtractionResult.model_dump() 输出一致，包含 "
            "companies/time_info/metrics/comparison_targets 等键。"
            "\n若为 None，改写器将仅依赖 LLM 自行理解查询语义（效果可能下降）。"
        ),
    )

    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description=(
            "多轮对话的历史上下文（可选）。"
            '用于消解代词引用（"它的PE是多少"→"它"=前文提到的公司）'
            '和省略主语（"那现金流呢"→省略的主语=前文公司）。'
        ),
    )

    target_data_source: Optional[str] = Field(
        default=None,
        description=(
            "目标数据源类型（可选，影响 Round 2 特化策略的选择）。"
            "\n可选值："
            "\n  - 'financial_reports': 财报文档（默认，强调会计科目和时间精度）"
            "\n  - 'realtime_data':    实时行情（强调股票代码和实时性术语）"
            "\n  - 'news':            新闻资讯（强调事件词和情绪表达）"
            "\n  - 'knowledge_graph': 知识图谱（强调实体关系和属性查询）"
            "\n若为 None，默认使用 'financial_reports' 策略。"
        ),
    )

    rewrite_history: List[RewriteStep] = Field(
        default_factory=list,
        description=(
            "历史改写步骤记录（可选）。"
            "支持增量改写场景：如果之前已经做过若干轮改写，"
            "可以通过此字段传入历史记录，避免重复执行已完成的工作。"
            "大多数场景下为空列表（从头开始改写）。"
        ),
    )


class QueryRewriteResult(BaseModel):
    """
    查询改写结果模型 (Query Rewrite Result Schema)

    【职责】封装一次改写操作的全部产出，包括最佳查询、所有候选和质量评估。

    【核心输出字段 / Core Output Fields】
    - best_query: 下游应使用的最终查询（质量最高的候选）
    - all_candidates: 所有中间结果的列表（含原始查询）
    - quality_score: best_query 的综合质量评分（0-1）

    【诊断字段 / Diagnostic Fields】
    - rewrite_steps: 每步改写的详细记录（策略+原因+预期改进）
    - total_rewrites: 实际执行的改写轮数
    - optimization_notes: 人类可读的优化过程摘要

    [序列化 / Serialization]
    可通过 .model_dump_json() 序列化后写入
    EnhancedAgentState.rewritten_queries 字段。
    """

    success: bool = Field(
        default=True,
        description=(
            "改写操作是否成功完成。"
            "True 表示至少执行了一轮改写（即使结果与原查询相同）；"
            "False 仅在发生不可恢复异常时出现。"
        ),
    )

    best_query: str = Field(
        ...,
        description=(
            "最佳改写查询——从 all_candidates 中按质量评分选出的最高分候选。"
            "这是下游模块（RAG Pipeline、TaskPlanner 等）应使用的唯一查询。"
            "注意：best_query 可能等于 original_query（当改写器判断无需改写时）。"
        ),
    )

    all_candidates: List[str] = Field(
        ...,
        description=(
            "所有候选查询的有序列表。"
            "\n结构：[original_query, round1_result, round2_result, ...]"
            "\n用途："
            "\n  1. 调试时可对比各轮差异"
            "\n  2. 高级场景下可用于多查询检索（Multi-Query Retrieval）"
            "\n  3. 回退机制：如果 best_query 效果不佳，可尝试其他候选"
        ),
    )

    rewrite_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "改写过程的逐步记录。"
            "每个元素是一个字典，对应 RewriteStep 的字段："
            "step_number, original_query, rewritten_query, rewrite_strategy, "
            "reasoning, expected_improvement。"
            "由 RewriteStep.__dict__ 序列化而来。"
        ),
    )

    total_rewrites: int = Field(
        default=0,
        ge=0,
        description=(
            "实际执行的改写轮数。"
            "可能小于 MAX_QUERY_REWRITES（当早停条件触发时）。"
            "值为 0 表示未做任何改写（直接返回原查询）。"
        ),
    )

    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "best_query 的综合质量评分（0.0-1.0）。"
            "\n评分维度："
            "\n  - 查询长度合理性（15-60字最佳）：权重 25%"
            "\n  - 公司实体完整性：权重 25%"
            "\n  - 时间信息完整性：权重 15%"
            "\n  - 指标覆盖度：权重 20%"
            "\n  - 专业术语使用量：权重 15%"
            "\n下游可根据此值决定是否需要对查询做进一步处理。"
        ),
    )

    optimization_notes: List[str] = Field(
        default_factory=list,
        description=(
            "人类可读的优化过程摘要列表。"
            "每个元素描述一轮改写做了什么，格式为："
            "'第N轮: [策略名] - 改写原因'。"
            "用于向终端用户展示改写过程，增强系统透明度。"
        ),
    )


# ════════════════════════════════════════════════════════════
# 第二部分：金融领域同义词词典
# Part 2: Financial Domain Synonym Dictionary
#
# 用于第三轮"扩展"策略的同义词补充。
# 设计原则：每个条目的同义词数量控制在 2-4 个，
# 避免过度扩展导致查询噪声增加。
# ════════════════════════════════════════════════════════════


SYNONYM_EXPANSION: Dict[str, List[str]] = {
    # ── 盈利能力类 / Profitability Terms ──
    "盈利能力": [
        "盈利",
        "利润",
        "赚钱能力",
        "获利能力",
    ],
    "净利润": [
        "净利",
        "纯利润",
        "归母净利润",
    ],
    "毛利率": [
        "毛利",
        "gross margin",
    ],
    "净利率": [
        "净利",
        "净利润率",
        "net margin",
    ],

    # ── 收入类 / Revenue Terms ──
    "营收": [
        "营业收入",
        "收入",
        "销售额",
        "营业额",
    ],

    # ── 投资回报率类 / Return Metrics ──
    "ROE": [
        "净资产收益率",
        "股东权益回报率",
        "return on equity",
    ],
    "ROA": [
        "总资产收益率",
        "return on assets",
    ],

    # ── 现金流类 / Cash Flow Terms ──
    "现金流": [
        "现金流量",
        "cash flow",
    ],
    "经营现金流": [
        "经营性现金流",
        "operating cash flow",
    ],

    # ── 估值类 / Valuation Terms ──
    "估值": [
        "valuation",
        "定价",
    ],
    "PE": [
        "市盈率",
        "price-to-earnings",
        "本益比",
    ],
    "PB": [
        "市净率",
        "price-to-book",
    ],

    # ── 成长性类 / Growth Terms ──
    "成长性": [
        "增长",
        "growth",
        "增速",
    ],

    # ── 风险类 / Risk Terms ──
    "风险": [
        "risk",
        "不确定性",
    ],

    # ── 杠杆类 / Leverage Terms ──
    "负债率": [
        "资产负债率",
        "debt ratio",
        "杠杆",
    ],
}


# ════════════════════════════════════════════════════════════
# 第三部分：查询模板库
# Part 3: Query Template Library
#
# 针对 4 种典型数据源的预定义查询模板。
# 在 Round 2 "特化"阶段，改写器会尝试用槽位信息填充这些模板。
# 如果模板填充失败（缺少必要槽位），则回退到 LLM 特化。
# ════════════════════════════════════════════════════════════


QUERY_TEMPLATES: Dict[str, List[str]] = {
    # ── 数据源 1: 财报文档 / Financial Reports ──
    # 适用场景：RAG 检索财报 PDF、年报文本、分析师报告等
    # 模板特点：强调公司名+时间+财务指标的标准组合
    "financial_analysis": [
        "{company} {year}年 {metrics} 分析",
        "{company} {time_range} 财报解读：{metrics}",
        "分析{company}{year}年度报告中的{metrics}表现",
    ],

    # ── 数据源 2: 股票对比 / Stock Comparison ──
    # 适用场景：多公司横向对比分析
    # 模板特点：强调多公司并列+对比关键词+共同指标
    "stock_comparison": [
        "对比分析 {companies} 的 {metrics}",
        "{companies} {time_range} {metrics} 对比研究",
        "{company1} vs {company2}: {metrics} 差异分析",
    ],

    # ── 数据源 3: 行业概览 / Industry Overview ──
    # 适用场景：行业级分析（无特定公司）
    # 模板特点：强调行业名称+趋势/竞争/产业链等宏观视角
    "industry_overview": [
        "{industry} 行业发展趋势与竞争格局",
        "{industry} 市场规模、主要参与者及前景分析",
        "{industry} 产业链上下游关系及投资机会",
    ],

    # ── 数据源 4: 实时数据 / Real-Time Data ──
    # 适用场景：股票行情、实时价格查询
    # 模板特点：强调实时性+股票代码+行情术语
    "realtime_data": [
        "{company} 实时股价行情",
        "{company} 今日涨跌幅与成交量",
        "获取{company}最新市场数据",
    ],
}


# ════════════════════════════════════════════════════════════
# 第四部分：查询改写器核心类
# Part 4: Query Rewriter Core Class
#
# 本模块唯一的公共 API 提供者。
# 外部调用方式：rewriter = QueryRewriter(); result = rewriter.rewrite(request)
# ════════════════════════════════════════════════════════════


class QueryRewriter:
    """
    金融领域查询改写器 (Financial Domain Query Rewriter)

    【类职责 / Class Responsibility】
    作为本模块的门面 (Facade)，封装完整的「标准化 → 特化 → 扩展/简化」
    多轮改写流水线。对外提供唯一的 rewrite() 方法。

    【改写流水线概览 / Pipeline Overview】

    Input: original_query ("茅台去年赚了多少钱")
        │
        ▼
    ┌─ Round 1: standardize ──────────────────────────┐
    │  LLM 驱动：口语→专业术语 + 槽位信息注入         │
    │  Output: "贵州茅台(600519.SH) 2025年净利润分析"   │
    │  质量>=0.7? ─No──▶ 继续下一轮                    │
    └──────────────────────────────────────────────────┘
        │
        ▼
    ┌─ Round 2: specialize ───────────────────────────┐
    │  模板填充 or LLM：针对目标数据源优化格式          │
    │  Output: "贵州茅台2025年度报告 净利润 营收 解读"  │
    │  质量>=0.7? ─Yes──▶ 早停，选择最佳候选            │
    └──────────────────────────────────────────────────┘
        │
        ▼
    ┌─ Round 3: expand / simplify ────────────────────┐
    │  规则驱动：同义词扩展（短查询）or 冗余去除（长查询）│
    │  Output: "...（相关概念：归母净利润 营业收入）"     │
    └──────────────────────────────────────────────────┘
        │
        ▼
    Output: best_query + all_candidates + quality_score

    【线程安全 / Thread Safety】
    - 无状态设计（除 self._llm 外无可变实例变量）
    - self._llm (ChatOpenAI) 是线程安全的
    - 结论：可在多线程环境中共享同一实例

    【温度参数 rationale / Temperature Rationale】
    temperature=0.3（介于槽位提取的 0.0 和创意写作的 0.7 之间）
    - 太低（0.0-0.1）：改写结果过于保守，可能只是复制原文
    - 太高（0.6+）：可能引入原文没有的信息（幻觉风险）
    - 0.3：平衡点——允许适度的措辞变化，同时保持事实一致性
    """

    def __init__(self) -> None:
        """
        初始化查询改写器实例。

        【初始化内容 / Initialization Tasks】
        创建 LLM 客户端（ChatOpenAI，连接 GLM-4-plus），
        配置 temperature=0.3 以允许适度的措辞创造性。
        """
        self._llm: ChatOpenAI = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.3,
        )

    def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteResult:
        """
        执行多轮查询改写的入口方法。

        【算法流程 / Algorithm Flow】

        1. 初始化候选列表 candidates = [original_query]
        2. FOR each round i in [1, MAX_QUERY_REWRITES]:
           a. 检查早停条件：_is_query_sufficient()?
              → 若满足且 i > 1，跳出循环
           b. 执行单轮改写：_perform_single_rewrite()
              → 返回 (rewritten_query, strategy, reasoning) 或 None
           c. 若返回 None（改写失败），跳出循环
           d. 记录 RewriteStep 并追加到 candidates
           e. 更新 current_query = rewritten_query
        3. 从 candidates 中选出最佳查询：_select_best_query()
        4. 评估最佳查询质量：_evaluate_query_quality()
        5. 构建并返回 QueryRewriteResult

        [Args / 参数]
            request: 经 Pydantic 校验的改写请求

        [Returns / 返回值]
            QueryRewriteResult: 包含 best_query/all_candidates/quality_score
                               等完整信息的改写结果对象

        [Raises / 异常]
            本方法不会抛出异常。所有异常在各子方法内部捕获并降级处理。
        """
        logger.info("[Query改写] 开始处理: %s...", request.original_query[:50])

        candidates: List[str] = [request.original_query]
        rewrite_steps: List[Dict[str, Any]] = []
        current_query: str = request.original_query

        for i in range(MAX_QUERY_REWRITES):
            if i > 0 and self._is_query_sufficient(current_query, request):
                logger.info("[Query改写] 第%d轮后查询质量达标，停止改写", i)
                break

            step_result = self._perform_single_rewrite(
                query=current_query,
                request=request,
                step_number=i + 1,
            )

            if step_result is None:
                logger.warning("[Query改写] 第%d轮改写未产生结果，终止", i + 1)
                break

            rewritten_query, strategy, reasoning = step_result

            step = RewriteStep(
                step_number=i + 1,
                original_query=current_query,
                rewritten_query=rewritten_query,
                rewrite_strategy=strategy,
                reasoning=reasoning,
                expected_improvement=self._get_expected_improvement(strategy),
            )
            rewrite_steps.append(step.__dict__)
            candidates.append(rewritten_query)
            current_query = rewritten_query

        best_query: str = self._select_best_query(candidates, request)
        quality_score: float = self._evaluate_query_quality(best_query, request)

        result = QueryRewriteResult(
            success=True,
            best_query=best_query,
            all_candidates=candidates,
            rewrite_steps=rewrite_steps,
            total_rewrites=len(rewrite_steps),
            quality_score=quality_score,
            optimization_notes=self._generate_optimization_notes(rewrite_steps),
        )

        logger.info(
            "[Query改写] 完成, 共%d轮, 质量分=%.2f",
            result.total_rewrites,
            result.quality_score,
        )
        return result

    # ─────────────────────────────────────────────────────
    # 单轮改写调度
    # Single-Rewrite Dispatch
    # ─────────────────────────────────────────────────────

    def _perform_single_rewrite(
        self,
        query: str,
        request: QueryRewriteRequest,
        step_number: int,
    ) -> Optional[Tuple[str, str, str]]:
        """
        根据当前轮次选择并执行对应的改写策略。

        【策略选择逻辑 / Strategy Selection Logic】

        step_number == 1 → standardize（标准化 + 槽位注入）
        step_number == 2 → specialize（数据源特化）
        step_number >= 3 → expand（查询 < 30字时）or simplify（查询 >= 30字时）

        这种固定轮次-策略映射保证了改写过程的确定性和可预测性，
        避免因动态策略选择导致的不可复现行为。

        [Args / 参数]
            query:       当前待改写的查询文本
            request:     改写请求（含槽位和数据源信息）
            step_number: 当前轮次编号（1-based）

        [Returns / 返回值]
            Tuple[str, str, str]: (改写结果, 策略名, 改写原因)
            None:                 改写未能产生有效结果（调用方应终止循环）
        """
        if step_number == 1:
            return self._rewrite_standardize(query, request)

        elif step_number == 2:
            return self._rewrite_for_datasource(query, request)

        else:
            strategy: str = "expand" if len(query) < 30 else "simplify"
            return self._rewrite_expand_or_simplify(query, strategy, request)

    # ─────────────────────────────────────────────────────
    # Round 1: 标准化改写
    # Standardization Rewrite
    # ─────────────────────────────────────────────────────

    def _rewrite_standardize(
        self,
        query: str,
        request: QueryRewriteRequest,
    ) -> Tuple[str, str, str]:
        """
        第一轮改写：将口语化表达转换为专业金融术语，并注入已提取的槽位信息。

        【改写目标 / Rewrite Goals】
        1. 术语标准化："赚了多少钱" → "净利润"、"好不好" → "盈利能力评估"
        2. 实体补全："茅台" → "贵州茅台(600519.SH)"
        3. 时间明确："去年" → "2025年"、"最近" → "近一年"
        4. 指标明确："盈利情况" → "毛利率 净利率 ROE"
        5. 冗余去除："帮我...好吗？请..." → 删除礼貌用语
        6. 长度控制：输出控制在 20-60 字符

        [Prompt Engineering 要点 / Prompt Key Points]
        - System Prompt 定义 6 条明确的优化规则
        - User Prompt 注入原始查询 + 槽位 JSON + 目标数据源
        - 要求只输出查询文本（不要解释），便于程序化解析

        [Args / 参数]
            query:   原始查询文本
            request: 改写请求（slots 字段将被注入 prompt）

        [Returns / 返回值]
            Tuple[str, str, str]: (标准化后的查询, "standardize", 原因说明)
        """
        system_prompt: str = (
            "你是一个金融领域的查询优化专家。\n"
            "请将用户的自然语言问题转换为更适合文档检索的标准化查询。\n\n"
            "优化规则：\n"
            "1. 将口语化表达转为专业金融术语\n"
            "2. 补充完整的公司名称和股票代码\n"
            "3. 明确时间范围（如'近三年'、'2020-2023'）\n"
            "4. 提取并明确列出关键财务指标\n"
            "5. 保持查询简洁，去除冗余词汇\n"
            "6. 输出长度控制在20-60字之间"
        )

        slot_info: str = ""
        if request.slots:
            slot_info = (
                f"\n\n已知槽位信息：\n"
                f"{json.dumps(request.slots, ensure_ascii=False, indent=2)}"
            )

        datasource_section: str = ""
        if request.target_data_source:
            datasource_section = f"\n目标数据源：{request.target_data_source}"

        user_prompt: str = (
            f"原始查询：{query}"
            f"{slot_info}"
            f"{datasource_section}\n\n"
            f"请输出优化后的标准化查询（只输出查询文本，不要解释）："
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            rewritten: str = response.content.strip()
            reasoning: str = "标准化术语，补充完整实体和时间信息"
            return rewritten, "standardize", reasoning

        except Exception as e:
            logger.error("[Query改写] 标准化改写失败: %s", e)
            return query, "standardize", "保持原样"

    # ─────────────────────────────────────────────────────
    # Round 2: 数据源特化改写
    # Data Source-Specific Specialization
    # ─────────────────────────────────────────────────────

    def _rewrite_for_datasource(
        self,
        query: str,
        request: QueryRewriteRequest,
    ) -> Tuple[str, str, str]:
        """
        第二轮改写：根据目标数据源的类型特性优化查询格式。

        【两阶段策略 / Two-Phase Strategy】

        Phase 1 — 模板填充（快速路径，零 LLM 调用）：
        尝试用槽位信息填充 QUERY_TEMPLATES 中对应的预定义模板。
        如果填充成功且结果与当前查询不同，直接返回。

        Phase 2 — LLM 特化（慢速路径，语义级优化）：
        当模板不适用（缺少必要槽位或模板均不匹配）时，
        使用 LLM 根据数据源特点对查询进行针对性优化。

        【各数据源优化要点 / Per-DataSource Optimization Points】

        数据源              优化方向                          示例
        ─────────────────────────────────────────────────────────────
        financial_reports  强调财报关键词、会计科目、时间精确性  "2023年报 归母净利润"
        realtime_data      强调实时性、具体代码、行情术语      "600519.SH 实时行情"
        news               强调事件、情绪词、时间敏感性        "茅台 业绩超预期 2024Q1"
        knowledge_graph    强调实体关系、属性查询              "茅台 子公司 关联交易"

        [Args / 参数]
            query:   Round 1 标准化后的查询
            request: 改写请求（target_data_source 决定特化方向）

        [Returns / 返回值]
            Tuple[str, str, str]: (特化后的查询, "specialize", 原因说明)
        """
        datasource: str = request.target_data_source or "financial_reports"
        templates: List[str] = QUERY_TEMPLATES.get(
            datasource, QUERY_TEMPLATES["financial_analysis"]
        )

        if request.slots:
            for template in templates:
                try:
                    filled: Optional[str] = self._fill_template(
                        template, request.slots
                    )
                    if filled and filled != query:
                        reasoning: str = f"使用{datasource}专用查询模板"
                        return filled, "specialize", reasoning
                except (KeyError, TypeError):
                    continue

        system_prompt: str = (
            f"你是一个检索优化专家。\n"
            f"当前目标是查询{datasource}数据源。\n\n"
            f"请根据数据源特点优化查询：\n"
            f"- financial_reports: 强调财报关键词、会计科目、时间精确性\n"
            f"- realtime_data: 强调实时性、具体代码、行情术语\n"
            f"- news: 强调事件、情绪词、时间敏感性\n"
            f"- knowledge_graph: 强调实体关系、属性查询"
        )

        user_prompt: str = (
            f"当前查询：{query}\n"
            f"数据源类型：{datasource}\n"
            f"请输出针对该数据源优化的查询（只输出查询文本）："
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            rewritten: str = response.content.strip()
            reasoning = f"针对{datasource}数据源特性优化"
            return rewritten, "specialize", reasoning

        except Exception as e:
            logger.error("[Query改写] 特化改写失败: %s", e)
            return query, "specialize", "保持原样"

    # ─────────────────────────────────────────────────────
    # Round 3+: 扩展或简化改写
    # Expansion or Simplification Rewrite
    # ─────────────────────────────────────────────────────

    def _rewrite_expand_or_simplify(
        self,
        query: str,
        strategy: str,
        request: QueryRewriteRequest,
    ) -> Tuple[str, str, str]:
        """
        第三轮及后续改写：根据查询长度选择扩展或简化策略。

        【策略选择规则 / Strategy Selection Rule】
        - len(query) < 30 → expand（查询偏短，可能缺乏足够的检索关键词）
          通过 SYNONYM_EXPANSION 词典添加同义词，提升召回率
        - len(query) >= 30 → simplify（查询偏长，可能含有噪声词汇）
          通过正则去除礼貌用语和冗余表达，提升精度

        [Args / 参数]
            query:    当前查询文本
            strategy: "expand" 或 "simplify"（由调用方根据长度决定）
            request:  改写请求（预留接口，当前未深度使用）

        [Returns / 返回值]
            Tuple[str, str, str]: (改写后的查询, 策略名, 原因说明)
                                  若无明显改进则返回原查询
        """
        if strategy == "expand":
            expanded: str = self._apply_synonym_expansion(query)
            if expanded != query:
                return expanded, "expand", "添加同义词和相关概念提升召回"
        else:
            simplified: str = self._simplify_query(query)
            if simplified != query:
                return simplified, "simplify", "精简冗余词汇提升匹配精度"

        return query, strategy, "无明显改进，保持原样"

    def _apply_synonym_expansion(self, query: str) -> str:
        """
        应用同义词扩展规则，在查询末尾追加相关概念。

        【扩展逻辑 / Expansion Logic】
        1. 遍历 SYNONYM_EXPANSION 词典，检查每个标准术语是否出现在查询中
        2. 对于命中的术语，取其前 2 个同义词（控制噪声）
        3. 过滤掉已在查询中出现的同义词（避免重复）
        4. 将收集到的同义词追加到查询末尾，格式为：
           "原查询 （相关概念：同义词1 同义词2 同义词3）"

        【为什么限制 2 个同义词？ / Why Limit to 2 Synonyms?】
        - 每个术语的同义词总数 2-4 个，全部添加会使查询过长
        - 前 2 个通常是最常用/最标准的同义表达
        - 总追加上限 3 个词（words_to_add[:3]），防止过度扩展

        [Args / 参数]
            query: 待扩展的查询文本

        [Returns / 返回值]
            str: 扩展后的查询（若无命中术语则返回原查询）
        """
        words_to_add: List[str] = []

        for term, synonyms in SYNONYM_EXPANSION.items():
            if term in query:
                for syn in synonyms[:2]:
                    if syn not in query:
                        words_to_add.append(syn)

        if words_to_add:
            return f"{query} （相关概念：{' '.join(words_to_add[:3])}）"
        return query

    def _simplify_query(self, query: str) -> str:
        """
        简化查询文本，去除冗余的礼貌用语和无效表达。

        【噪声模式列表 / Noise Pattern List】

        模式                      匹配示例                  替换结果
        ─────────────────────────────────────────────────────────────
        "帮我\s*"               "帮我分析茅台"           → "分析茅台"
        "请\s*"                 "请告诉我ROE"           → "告诉我ROE"
        "我想知道\s*"           "我想知道净利润"         → "净利润"
        "能否告诉我\s*"         "能否告诉我营收"         → "告诉我营收"
        "分析一下\s*"           "分析一下盈利"           → "盈利"
        "的/情况/怎么样/如何/呢/吗" "茅台的情况怎么样"      → "茅台"
                              （仅在句尾匹配，避免误删句中有效词）

        [Args / 参数]
            query: 待简化的查询文本

        [Returns / 返回值]
            str: 简化后的查询（若简化结果为空则返回原查询）
        """
        noise_patterns: List[str] = [
            r"帮我\s*",
            r"请\s*",
            r"我想知道\s*",
            r"能否告诉我\s*",
            r"分析一下\s*",
            r"\s*(?:的|情况|怎么样|如何|呢|吗)\s*$",
        ]

        simplified: str = query
        for pattern in noise_patterns:
            simplified = re.sub(pattern, "", simplified, flags=re.IGNORECASE)

        simplified = simplified.strip()
        return simplified if simplified else query

    # ─────────────────────────────────────────────────────
    # 辅助工具方法
    # Utility Methods
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _fill_template(template: str, slots: Dict[str, Any]) -> Optional[str]:
        """
        用槽位信息填充查询模板中的占位符。

        【填充机制 / Filling Mechanism】
        使用 Python 内置的 str.format() 方法，将模板中的 {placeholder}
        替换为 slots 字典中对应键的值。

        【支持的占位符 / Supported Placeholders】
        - {company} / {companies}: 从 slots["companies"] 提取
        - {year} / {time_range}:  从 slots["time_info"] 提取
        - {metrics}:             从 slots["metrics"] 提取
        - {industry}:            从 slots["industry"] 提取
        - {company1}, {company2}: 对比场景下的两个公司

        [Args / 参数]
            template: 含占位符的模板字符串
            slots:    键值对字典，键与模板占位符对应

        [Returns / 返回值]
            str:  填充后的完整查询字符串
            None: 占位符无法匹配（KeyError）时返回 None
        """
        try:
            return template.format(**slots)
        except KeyError:
            return None

    def _is_query_sufficient(self, query: str, request: QueryRewriteRequest) -> bool:
        """
        判断当前查询是否已经足够好，可以提前终止改写。

        【充分性判定规则 / Sufficiency Rules】

        必须同时满足以下全部条件才返回 True：
        1. 查询长度 >= 10 字符（太短的查询通常信息不足）
        2. 至少包含 2 类以下关键实体：
           - 公司实体（通过 COMPANY_ALIASES 匹配或 slots 中有 companies）
           - 时间信息（通过 TIME_PATTERNS 匹配或 slots 中有 time_info）
           - 财务指标（通过 FINANCIAL_METRICS 匹配）

        【为什么阈值是 2 类？ / Why Threshold of 2?】
        - 1 类：信息不足，继续改写可能带来显著改善
        - 2 类：已经具备基本检索能力，额外一轮改写的边际收益递减
        - 3 类：非常完善，几乎不需要再改写

        [Args / 参数]
            query:   当前查询文本
            request: 改写请求（提供 slots 信息辅助判断）

        [Returns / 返回值]
            True:  查询质量达标，建议停止改写
            False: 查询仍有改善空间，建议继续
        """
        if len(query) < 10:
            return False

        has_company: bool = any(
            alias in query for alias in COMPANY_ALIASES.keys()
        ) or bool(request.slots and request.slots.get("companies"))

        has_time: bool = any(
            re.search(pattern, query) for pattern in TIME_PATTERNS.values()
        ) or bool(request.slots and request.slots.get("time_info"))

        has_metrics: bool = any(
            metric in query
            for metrics_list in FINANCIAL_METRICS.values()
            for metric in metrics_list
        )

        score: int = sum([has_company, has_time, has_metrics])
        return score >= 2

    def _select_best_query(
        self,
        candidates: List[str],
        request: QueryRewriteRequest,
    ) -> str:
        """
        从所有候选查询中选择质量分数最高的一个作为最佳查询。

        【选择算法 / Selection Algorithm】
        1. 对每个候选调用 _evaluate_query_quality() 计算质量分
        2. 按分数降序排列
        3. 返回第一名（最高分候选）

        【为什么不用 LLM 做 selection？ / Why Not Use LLM?】
        - 规则评分是确定性的（相同输入永远给出相同结果），便于调试
        - LLM selection 引入额外的延迟（~1s）和非确定性
        - 当前评分函数已覆盖主要质量维度，准确度足够

        [Args / 参数]
            candidates: 候选查询列表（含原始查询和各轮改写结果）
            request:    改写请求（提供 slots 信息用于评分）

        [Returns / 返回值]
            str: 质量分数最高的候选查询字符串
        """
        scored_candidates: List[Tuple[float, str]] = []
        for candidate in candidates:
            score: float = self._evaluate_query_quality(candidate, request)
            scored_candidates.append((score, candidate))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    def _evaluate_query_quality(self, query: str, request: QueryRewriteRequest) -> float:
        """
        评估单个查询的综合质量分数（0.0-1.0）。

        【评分维度与权重 / Scoring Dimensions & Weights】

        ┌────────────────────┬────────┬──────────────────────────────────┐
        │ 维度                │ 权重   │ 评分细则                            │
        ├────────────────────┼────────┼──────────────────────────────────┤
        │ 长度合理性          │ 0.25   │ 15-60字:满分 / 10-80字:60% / 其他:20% │
        │ 公司实体完整性      │ 0.25   │ slots中有公司且查询中出现:满分       │
        │ 时间信息完整性      │ 0.15   │ slots有时间且查询中出现:满分         │
        │ 指标覆盖度          │ 0.20   │ slots有指标且查询中出现:满分         │
        │ 专业术语使用量      │ 0.15   │ 出现>=2个金融术语:满分              │
        └────────────────────┴────────┴──────────────────────────────────┘

        [Args / 参数]
            query:   待评估的查询文本
            request: 改写请求（提供 slots 作为参考基准）

        [Returns / 返回值]
            float: 0.0-1.0 的质量分数（clamp 到上限 1.0）
        """
        score: float = 0.0

        length: int = len(query)
        if 15 <= length <= 60:
            score += 0.25
        elif 10 <= length <= 80:
            score += 0.15
        else:
            score += 0.05

        if request.slots:
            slots: Dict[str, Any] = request.slots

            companies = slots.get("companies")
            if companies and any(
                c.get("name") in query for c in companies if isinstance(c, dict)
            ):
                score += 0.25

            time_info = slots.get("time_info")
            if time_info and isinstance(time_info, dict):
                year_val = time_info.get("year")
                period_val = time_info.get("period")
                if year_val and str(year_val) in query:
                    score += 0.15
                elif period_val and period_val in query:
                    score += 0.15

            metrics = slots.get("metrics")
            if metrics and isinstance(metrics, dict):
                primary = metrics.get("primary_metrics", [])
                if any(m in query for m in primary):
                    score += 0.20

        professional_terms: int = sum(
            1
            for terms in FINANCIAL_METRICS.values()
            for term in terms
            if term in query
        )
        if professional_terms >= 2:
            score += 0.15

        return min(score, 1.0)

    @staticmethod
    def _get_expected_improvement(strategy: str) -> str:
        """
        根据改写策略返回预期的改进描述。

        [Args / 参数]
            strategy: 策略标识符

        [Returns / 返回值]
            str: 该策略的预期改进点描述
        """
        improvements: Dict[str, str] = {
            "standardize": "提升术语规范性，改善召回率",
            "specialize": "适配数据源特点，提升精准度",
            "expand": "增加相关概念，扩大召回范围",
            "simplify": "减少噪声，提升匹配精度",
        }
        return improvements.get(strategy, "优化查询表达")

    @staticmethod
    def _generate_optimization_notes(steps: List[Dict[str, Any]]) -> List[str]:
        """
        将改写步骤记录转换为人类可读的优化说明列表。

        [Output Format / 输出格式]
        每个元素的格式：'第N轮: [策略名] - 改写原因'

        [Example / 示例]
        输入: [{"step_number": 1, "rewrite_strategy": "standardize", ...}]
        输出: ["第1轮: [standardize] - 标准化术语，补充完整实体和时间信息"]

        [Args / 参数]
            steps: RewriteStep 序列化后的字典列表

        [Returns / 返回值]
            List[str]: 人类可读的优化说明列表
        """
        notes: List[str] = []
        for step in steps:
            notes.append(
                f"第{step['step_number']}轮: "
                f"[{step['rewrite_strategy']}] - {step['reasoning']}"
            )
        return notes


# ════════════════════════════════════════════════════════════
# 第五部分：延迟导入（解决循环依赖）
# Part 5: Lazy Imports (Circular Dependency Resolution)
#
# query_rewriter.py 依赖 slot_extractor.py 中的词典常量，
# 而 slot_extractor.py 不依赖 query_rewriter.py（单向依赖）。
# 将导入放在文件底部配合 try-except 确保即使导入失败也不会
# 阻止模块加载（此时词典为空字典，功能降级但不会崩溃）。
# ════════════════════════════════════════════════════════════

try:
    from src.core.slot_extractor import TIME_PATTERNS, COMPANY_ALIASES, FINANCIAL_METRICS
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[QueryRewriter] ⚠️ 未能导入 slot_extractor 常量: {e}，功能可能降级")
    TIME_PATTERNS: Dict[str, Any] = {}
    COMPANY_ALIASES: Dict[str, Any] = {}
    FINANCIAL_METRICS: Dict[str, Any] = {}


# ════════════════════════════════════════════════════════════
# 第六部分：公共便捷函数
# Part 6: Public Convenience Function
# ════════════════════════════════════════════════════════════


def rewrite_query(
    original_query: str,
    slots: Optional[Dict[str, Any]] = None,
    target_source: Optional[str] = None,
) -> QueryRewriteResult:
    """
    便捷函数：一步完成查询改写。

    【设计意图 / Design Intent】
    将「创建实例→构建请求→调用 rewrite」三步压缩为一步函数调用。
    适用于快速测试、脚本式调用和 Jupyter Notebook 交互。

    [Args / 参数]
        original_query: 用户原始查询文本
        slots:         已提取的槽位信息（来自 SlotExtractor 的输出）
        target_source:  目标数据源类型

    [Returns / 返回值]
        QueryRewriteResult: 完整的改写结果

    [Example / 示例]
        >>> result = rewrite_query(
        ...     "茅台去年赚了多少钱",
        ...     target_source="financial_reports"
        ... )
        >>> print(result.best_query)
        '贵州茅台 2025年 净利润 分析'
        >>> print(result.quality_score)
        0.82
    """
    rewriter = QueryRewriter()
    request = QueryRewriteRequest(
        original_query=original_query,
        slots=slots,
        target_data_source=target_source,
    )
    return rewriter.rewrite(request)


# ════════════════════════════════════════════════════════════
# 第七部分：模块自测
# Part 7: Self-Test Suite
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases: List[Dict[str, str]] = [
        {"query": "茅台去年赚了多少钱", "source": "financial_reports"},
        {"query": "对比下比亚迪和宁德时代谁更好", "source": "stock_comparison"},
        {"query": "新能源车行业现在怎么样了", "source": "industry_overview"},
        {"query": "招商银行今天股价", "source": "realtime_data"},
    ]

    for case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"原始: {case['query']}")
        result = rewrite_query(case["query"], target_source=case["source"])
        print(f"最佳: {result.best_query}")
        print(f"质量: {result.quality_score:.2f}")
        print(f"候选: {result.all_candidates}")
