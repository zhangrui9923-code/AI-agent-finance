"""
增强版金融意图分类模块 (Enhanced Financial Intent Classification)
================================================================

【模块定位 / Module Positioning】
本模块是 AI 金融研究助手系统的「意图理解」核心组件，位于 Slot Extraction 和
Query Rewriting 之后、Task Planning 之前。它负责将用户查询映射到系统预定义的
8 大主意图类别，并进一步细分为子意图，为下游的任务规划提供路由依据。

【架构角色 / Architectural Role】

┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│  改写后查询    │───▶│ IntentClassifier    │───▶│  下游消费者        │
│              │    │ (本模块)             │                    │
│ "贵州茅台...  │    │ 8类主意图 × 12子意图  │    │ TaskPlanner      │
│  净利润 分析" │    │ 规则 + LLM + 融合     │    │ (根据意图选模板)  │
│              │    │ Top-N + 消歧          │    │ QueryRewriter    │
└──────────────┘    └─────────────────────┘    │ (调整改写策略)    │
                                               └──────────────────┘

【为什么需要独立的意图分类？/ Why Do We Need Intent Classification?】

同一个查询在不同意图下需要完全不同的处理路径：

查询："茅台的PE是多少"
├─ financial_analysis → 调用 FinancialAgent 做 PE 估值分析（深度报告）
├─ realtime_query     → 调用 MCP StockQuote 工具查实时 PE-TTM（快速回答）
└─ stock_comparison   → 需要等待用户提供对比对象后再处理

没有准确的意图分类，系统就无法选择正确的处理链路。

【意图分类体系设计 / Intent Taxonomy Design】

本系统采用「主意图(Primary) + 子意图(Secondary)」两层分类体系：

Level 1 — Primary Intent（8 类）：
决定宏观处理路径——调用哪个 Agent、使用哪套任务模板、检索哪个知识库

Level 2 — Secondary Intent（12 类）：
决定微观分析维度——在已选定的大方向下，具体关注哪些方面

【8 大主意图详解 / 8 Primary Intents Detail】

┌────┬─────────────────┬──────────────────────────────────────────┐
│ #  │ 主意图            │ 典型场景                                  │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 1  │ financial_       │ "分析茅台2023年ROE""解读比亚迪现金流量表"   │
│    │ analysis         │ → 调用 FinancialAgent + RAG 财报检索       │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 2  │ stock_comparison │ "茅台vs五粮液谁更好""对比三家银行盈利能力"   │
│    │                 │ → 多对象数据获取 + 对比矩阵生成             │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 3  │ industry_overview│ "新能源汽车行业趋势""半导体产业链分析"       │
│    │                 │ → 行业知识库检索 + 竞争格局分析              │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 4  │ realtime_query   │ "茅台今天股价""现在大盘怎么样"              │
│    │                 │ → MCP 实时行情工具 + 快速返回               │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 5  │ portfolio_       │ "我的组合风险大不""如何优化持仓"           │
│    │ analysis         │ → 组合风险模型 + 配置优化建议              │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 6  │ risk_alert       │ "这只股票会暴雷吗""监测比亚迪负面新闻"       │
│    │                 │ → RiskAgent + 新闻情感监控                 │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 7  │ strategy_backtest│ "回测均线策略""价值投资历史表现"           │
│    │                 │ → 回测引擎 + 绩效指标计算                  │
├────┼─────────────────┼──────────────────────────────────────────┤
│ 8  │ market_sentiment │ "现在市场情绪""投资者对新能源怎么看"        │
│    │                 │ → 舆情分析 + 恐慌/贪婪指数                  │
└────┴─────────────────┴──────────────────────────────────────────┘

【核心设计原则 / Core Design Principles】

1. **混合分类策略 (Hybrid Classification Strategy)**
   - Layer 1: 关键词规则匹配（< 10ms，覆盖 ~70% 高频明确表达）
   - Layer 2: LLM 语义理解（~1-2s，覆盖复杂/模糊/隐含语义）
   - Layer 3: 加权融合 + 归一化排序（确定性的最终输出）

2. **Top-N 候选机制 (Top-N Candidate Mechanism)**
   - 不只返回最佳匹配，而是返回 Top-5 候选列表
   - 当最佳候选置信度不足时（< 0.6），可查看次优候选
   - 支持消歧交互：当 top-2 分数接近时（差值 < 0.15），触发澄清问题

3. **歧义检测与消歧 (Ambiguity Detection & Resolution)**
   - 自动检测：top-2 置信度差 < 0.15 且最高分 < 0.6
   - 自动生成：基于 INTENT_DEFINITIONS 的 description 字段生成追问
   - 交互层据此决定是否暂停自动流程等待用户确认

4. **上下文感知 (Context Awareness)**
   - 支持多轮对话上下文注入（解决代词引用和省略问题）
   - 支持槽位信息辅助（已提取的公司/时间/指标帮助缩小意图范围）

【性能特征 / Performance Characteristics】
- 规则阶段：O(K*M)，K=意图类别数(8)，M=每类关键词数(~12)，通常 < 10ms
- LLM 阶段：网络 I/O 密集，典型延迟 1-2s
- 融合排序：O(N log N)，N=候选数（通常 < 10），< 1ms
- 总体 P99 延迟：~2-3s

【依赖关系 / Dependencies】
- 上游：SlotExtractionResult（slots 辅助判断）、QueryRewriter（改写后的查询）
- 下游：EnhancedAgentState.intent / EnhancedAgentState.intent_hierarchy / TaskPlanner
- 外部依赖：langchain_openai.ChatOpenAI、pydantic、config.settings

【版本信息 / Version Info】
- Author: AI Finance Assistant Team
- Version: 2.0.0 (极致优化版 / Extreme Optimization Edition)
- Last Updated: 2026-04-13
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL


# ─────────────────────────────────────────────────────────────
# 日志配置 / Logging Configuration
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 第一部分：意图类型枚举定义
# Part 1: Intent Type Enum Definitions
#
# 使用 Python Enum（而非字符串常量）的原因：
# 1. 类型安全：IDE 自动补全和静态检查
# 2. 可迭代：可直接 for intent in PrimaryIntent 遍历所有类别
# 3. 序列化友好：.value 属性获取字符串表示
# 4. 可扩展：新增意图只需添加枚举成员
# ════════════════════════════════════════════════════════════


class PrimaryIntent(str, Enum):
    """
    主意图枚举 (Primary Intent Enumeration)

    【职责】定义系统支持的所有顶层意图类别（共 8 个）。
    每个 PrimaryIntent 值对应一条主要的 Agent 处理路径。

    【枚举值与路由关系 / Enum Value & Routing】

    枚举值                      路由目标                     核心数据源
    ──────────────────────────────────────────────────────────────────
    FINANCIAL_ANALYSIS       → FinancialAgent           Alpha Vantage + 财报RAG
    STOCK_COMPARISON         → FinancialAgent (对比模式)  多公司数据聚合
    INDUSTRY_OVERVIEW        → RAG Pipeline (行业库)    行业研究报告
    REALTIME_QUERY           → RealtimeAgent            MCP StockQuote Tool
    PORTFOLIO_ANALYSIS       → 组合分析模块              用户持仓数据
    RISK_ALERT               → RiskAgent                新闻API + 财务预警
    STRATEGY_BACKTEST        → 回测引擎                  历史行情数据
    MARKET_SENTIMENT         → 舆情分析模块              社交媒体 + 新闻

    【扩展指南 / Extension Guide】
    如需新增主意图：
    1. 在此枚举中添加新成员（如 NEWS_ANALYSIS = "news_analysis"）
    2. 在 INTENT_DEFINITIONS 中添加对应的定义字典
    3. 在 TaskPlanner 中注册对应的任务模板
    4. （可选）添加对应的 SecondaryIntent 子类别
    """

    FINANCIAL_ANALYSIS = "financial_analysis"
    """
    财务分析 (Financial Analysis)
    
    最常见的主意图类型。涵盖公司财报解读、财务指标分析、经营数据评估等。
    是系统的「默认意图」——当无法明确分类时回退到此类型。
    
    触发示例：
    - "分析贵州茅台2023年年报"
    - "比亚迪的ROE和净利率怎么样"
    - "解读宁德时代的现金流量表"
    """

    STOCK_COMPARISON = "stock_comparison"
    """
    个股对比 (Stock Comparison)
    
    用于多公司横向比较场景。系统会识别所有被比较的对象，
    并为每个对象分别获取数据后生成对比矩阵。
    
    触发示例：
    - "对比茅台和五粮液的估值水平"
    - "比亚迪 vs 宁德时代，谁更值得投资？"
    - "比较三家银行的 ROE 水平"
    """

    INDUSTRY_OVERVIEW = "industry_overview"
    """
    行业综述 (Industry Overview)
    
    当用户查询聚焦于行业/赛道层面而非单一公司时触发。
    不需要具体的公司槽位，但 industry 槽位应有值。
    
    触发示例：
    - "新能源汽车行业发展趋势"
    - "半导体产业链上中下游分析"
    - "白酒行业的竞争格局和前景"
    """

    REALTIME_QUERY = "realtime_query"
    """
    实时行情 (Real-Time Market Data)
    
    时间敏感性最高的意图。用户期望获得秒级/分钟级的最新数据，
    而非基于历史数据的分析报告。应优先调用 MCP 实时工具。
    
    触发示例：
    - "茅台今天股价多少"
    - "比亚迪现在的价格"
    - "今天大盘怎么样"
    - "查询宁德时代最新成交量"
    """

    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    """
    投资组合分析 (Portfolio Analysis)
    
    针对用户持仓组合的整体性分析。需要用户提供持仓数据
    （或从外部系统导入），然后进行风险/收益/配置优化分析。
    
    触发示例：
    - "分析我的投资组合风险"
    - "如何优化我的股票持仓结构"
    - "评估这个组合的收益风险比"
    """

    RISK_ALERT = "risk_alert"
    """
    风险预警 (Risk Alert)
    
    关注负面因素和潜在风险的防御型意图。与 financial_analysis
    的区别在于视角不同：前者看"好不好"，后者看"有什么危险"。
    
    触发示例：
    - "茅台有哪些潜在风险"
    - "监测比亚迪的负面新闻"
    - "这只股票会暴雷吗"
    """

    STRATEGY_BACKTEST = "strategy_backtest"
    """
    策略回测 (Strategy Backtesting)
    
    投资研究的高级场景。用户描述一个交易策略（如"均线交叉""动量"），
    系统用历史数据模拟执行并计算绩效指标。
    
    触发示例：
    - "回测一下均线交叉策略的表现"
    - "价值投资策略在过去5年的收益率"
    - "模拟动量交易策略的最大回撤"
    """

    MARKET_SENTIMENT = "market_sentiment"
    """
    市场情绪 (Market Sentiment)
    
    从心理学和行为金融学角度分析市场参与者整体情绪状态。
    数据来源包括社交媒体舆情、资金流向、VIX 恐慌指数等。
    
    触发示例：
    - "现在市场情绪怎么样"
    - "投资者对新能源板块怎么看"
    - "分析今日 A 股市场舆情"
    """


class SecondaryIntent(str, Enum):
    """
    子意图枚举 (Secondary Intent Enumeration)

    【职责】在主意图的基础上进一步细化分析维度。
    每个 PrimaryIntent 可关联多个 SecondaryIntent（多对多关系）。

    【子意图分组 / Secondary Intent Grouping】

    ┌──────────────────┬──────────────────────────────────────────┐
    │ 分组              │ 包含的子意图                               │
    ├──────────────────┼──────────────────────────────────────────┤
    │ 财务分析维度       │ PROFITABILITY, GROWTH, CASH_FLOW, VALUATION │
    │ 分析方法维度       │ TECHNICAL, FUNDAMENTAL                   │
    │ 风险类型维度       │ CREDIT_RISK, MARKET_RISK, OPERATIONAL_RISK │
    │ 分析深度维度       │ OVERVIEW, DEEP_DIVE, TREND_PREDICTION     │
    └──────────────────┴──────────────────────────────────────────┘

    【注意 / Note】
    SecondaryIntent 是可选的——并非每次分类都必须有子意图。
    当 LLM 无法确定或用户查询过于笼统时，secondary_intent 为 None。
    """

    # ── 财务分析维度 / Financial Analysis Dimensions ──

    PROFITABILITY = "profitability_analysis"
    """盈利能力分析：毛利率、净利率、ROE、ROA 等"""

    GROWTH = "growth_analysis"
    """成长性分析：营收增速、净利润增速、EPS 增速等"""

    CASH_FLOW = "cashflow_analysis"
    """现金流分析：OCF、FCF、资本支出等"""

    VALUATION = "valuation_analysis"
    """估值分析：PE、PB、PS、EV/EBITDA、PEG 等"""

    # ── 分析方法维度 / Analysis Method Dimensions ──

    TECHNICAL = "technical_analysis"
    """技术面分析：K线形态、均线、MACD、RSI 等技术指标"""

    FUNDAMENTAL = "fundamental_analysis"
    """基本面分析：财报数据、估值模型、竞争优势等"""

    # ── 风险类型维度 / Risk Type Dimensions ──

    CREDIT_RISK = "credit_risk"
    """信用风险：违约概率、债务偿还能力"""

    MARKET_RISK = "market_risk"
    """市场风险：系统性风险、Beta、波动率"""

    OPERATIONAL_RISK = "operational_risk"
    """经营风险：管理风险、运营中断、合规风险"""

    # ── 分析深度维度 / Analysis Depth Dimensions ──

    OVERVIEW = "overview"
    """概览性分析：高层摘要，适合快速浏览"""

    DEEP_DIVE = "deep_dive"
    """深度分析：全面详尽的专业级分析"""

    TREND_PREDICTION = "trend_prediction"
    """趋势预测：基于历史数据的未来走势研判"""


# ════════════════════════════════════════════════════════════
# 第二部分：数据模型定义
# Part 2: Data Model Definitions
# ════════════════════════════════════════════════════════════


@dataclass
class IntentCandidate:
    """
    候选意图 (Intent Candidate)

    【职责】表示一个可能的意图分类结果，包含置信度和推理依据。
    在融合排序阶段会产生多个候选，最终选出最佳的一个作为 primary_result。

    【字段说明 / Field Descriptions】
    - primary: 主意图类别（必填）
    - secondary: 子意图类别（可选，LLM 可能无法确定时为 None）
    - confidence: 该候选的置信度分数（归一化后总和 ≈ 1.0）
    - reasoning: 分类依据的自然语言描述（用于调试和消歧问题生成）

    [生命周期 / Lifecycle]
    创建于 _classify_by_rules() 或 _classify_by_llm()，
    经 _merge_and_rank() 融合加权后写入最终的 candidates 列表。
    """

    primary: PrimaryIntent
    """主意图类别（来自 PrimaryIntent 枚举）"""

    secondary: Optional[SecondaryIntent] = None
    """
    子意图类别（来自 SecondaryIntent 枚举）。
    None 表示未能确定子意图（不影响主意图的有效性）。
    """

    confidence: float = 0.0
    """
    置信度分数（0.0-1.0）。
    
    计算过程：
    1. 规则阶段：base = 0.3 + match_count × 0.1（上限 0.7）
    2. LLM 阶段：直接由 LLM 输出（0-1）
    3. 融合后：加权求和（规则×0.4 + LLM×0.6），然后归一化
    
    注意：归一化后的 confidence 不再代表绝对概率，
    而是相对排名指标（各候选之和 = 1.0）。
    """

    reasoning: str = ""
    """
    分类依据的自然语言描述。
    
    示例（规则阶段）："匹配到3个关键词: ['财报', '利润', 'ROE']"
    示例（LLM阶段）："用户询问'盈利能力'且提到了具体公司，属于典型的财务分析需求"
    
    用途：调试日志、消歧问题生成、结果可解释性展示。
    """


class IntentClassificationResult(BaseModel):
    """
    意图分类完整结果模型 (Complete Intent Classification Result)

    【职责】本模块对外输出的唯一数据结构，封装了一次分类操作的全部产出。

    【核心输出字段 / Core Output Fields】
    - primary_intent: 最佳匹配的主意图（下游 TaskPlanner 用此选择任务模板）
    - secondary_intent: 细分子意图（影响分析维度的选择）
    - confidence: 最佳匹配的置信度（下游据此决定是否需要消歧）

    【诊断字段 / Diagnostic Fields】
    - candidates: Top-N 候选列表（含每个候选的详细信息）
    - intent_hierarchy: 层级化的完整意图信息
    - ambiguity_detected: 是否检测到歧义
    - disambiguation_questions: 自动生成的消歧追问

    [序列化 / Serialization]
    .model_dump_json() 后直接写入 EnhancedAgentState 的 intent 和 intent_hierarchy 字段。
    """

    success: bool = Field(
        default=True,
        description="分类操作是否成功完成",
    )

    primary_intent: PrimaryIntent = Field(
        ...,
        description=(
            "最佳匹配的主意图类别。"
            "这是下游 TaskPlanner 选择任务模板的核心依据。"
            "8 种可能值见 PrimaryIntent 枚举定义。"
        ),
    )

    secondary_intent: Optional[SecondaryIntent] = Field(
        default=None,
        description=(
            "最佳匹配的子意图类别。"
            "在主意图确定后进一步细化分析方向。"
            "None 表示未能确定子意图（下游应使用默认分析维度）。"
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "primary_intent 的归一化置信度分数。"
            "\n解读指南："
            "\n  - >= 0.7: 高置信度，直接使用"
            "\n  - 0.5-0.7: 中等，建议参考 candidates 中的其他选项"
            "\n  - < 0.5: 低置信度，应检查 ambiguity_detected 并考虑消歧"
        ),
    )

    candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Top-N 候选意图列表（通常 1-5 个）。"
            "每个元素包含 primary_intent, secondary_intent, confidence, reasoning。"
            "按置信度降序排列。用于消歧和调试。"
        ),
    )

    intent_hierarchy: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "完整的意图层级结构。"
            "包含 primary, secondary, confidence, alternatives, definition 等键。"
            "比扁平的 primary_intent+secondary_intent 携带更多上下文信息。"
            "直接映射到 EnhancedAgentState.intent_hierarchy 字段。"
        ),
    )

    classification_method: str = Field(
        ...,
        description=(
            "本次分类使用的方法标识。"
            "可选值：'rule'（仅规则）/ 'llm'（仅 LLM）/ 'hybrid'（融合，最常见）"
        ),
    )

    ambiguity_detected: bool = Field(
        default=False,
        description=(
            "是否检测到意图歧义。"
            "\n判定条件（需同时满足）："
            "\n  1. 候选数 >= 2"
            "\n  2. 第一名置信度 < 0.6"
            "\n  3. 第一名与第二名置信度差 < 0.15"
            "\n为 True 时，disambiguation_questions 必定非空。"
        ),
    )

    disambiguation_questions: List[str] = Field(
        default_factory=list,
        description=(
            "自动生成的消歧追问列表。"
            "每个问题基于候选意图的 INTENT_DEFINITIONS.description 生成。"
            "格式示例：'您是想了解[某意图的描述]吗？'"
        ),
    )

    context_aware: bool = Field(
        default=False,
        description=(
            "本次分类是否利用了对话上下文信息。"
            "True 表示 context 参数非空且对分类结果产生了影响。"
            "用于评估多轮对话场景下的分类质量提升效果。"
        ),
    )


# ════════════════════════════════════════════════════════════
# 第三部分：意图定义与示例库
# Part 3: Intent Definition & Example Library
#
# 这是规则分类阶段的"知识库"，定义了每种意图的关键词特征和典型表达。
# 结构：{PrimaryIntent: {keywords, sub_intents, examples, description}}
# ════════════════════════════════════════════════════════════


INTENT_DEFINITIONS: Dict[PrimaryIntent, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────
    # 意图 1: 财务分析 (最常见，覆盖率 ~40%)
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.FINANCIAL_ANALYSIS: {
        "description": (
            "解读公司财报、财务指标、经营数据分析。"
            "包括盈利能力、成长性、现金流、估值等多维度评估。"
        ),
        "keywords": [
            "财报", "年报", "季报", "财务",
            "利润", "营收", "ROE", "PE",
            "盈利", "现金流", "资产负债表", "利润表",
            "毛利率", "净利率", "净资产收益率",
            "归母净利润", "EBITDA", "杜邦分析",
        ],
        "sub_intents": [
            SecondaryIntent.PROFITABILITY,
            SecondaryIntent.GROWTH,
            SecondaryIntent.CASH_FLOW,
            SecondaryIntent.VALUATION,
        ],
        "examples": [
            "分析贵州茅台2023年年报",
            "比亚迪的盈利能力如何？",
            "解读宁德时代的现金流量表",
            "计算招商银行的估值水平",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 2: 个股对比
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.STOCK_COMPARISON: {
        "description": (
            "对比分析多只股票或公司的表现。"
            "支持基本面对比、估值对比、技术面对比等多种比较维度。"
        ),
        "keywords": [
            "对比", "比较", "vs", "VS",
            "哪个好", "谁更强", "优劣",
            "差异", "差异分析", "排名",
            "选哪个", "更值得",
        ],
        "sub_intents": [
            SecondaryIntent.FUNDAMENTAL,
            SecondaryIntent.TECHNICAL,
            SecondaryIntent.VALUATION,
        ],
        "examples": [
            "对比茅台和五粮液的估值",
            "比亚迪vs宁德时代，谁更值得投资？",
            "比较三家银行的ROE水平",
            "新能源车企竞争力排行",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 3: 行业综述
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.INDUSTRY_OVERVIEW: {
        "description": (
            "行业整体情况、发展趋势、竞争格局分析。"
            "适用于宏观层面的行业/赛道级别查询。"
        ),
        "keywords": [
            "行业", "产业", "市场", "赛道",
            "格局", "趋势", "发展",
            "前景", "规模", "龙头",
            "产业链", "上下游", "竞争",
        ],
        "sub_intents": [
            SecondaryIntent.OVERVIEW,
            SecondaryIntent.DEEP_DIVE,
            SecondaryIntent.TREND_PREDICTION,
        ],
        "examples": [
            "新能源汽车行业发展趋势",
            "半导体产业链分析",
            "白酒行业的竞争格局",
            "光伏市场的未来前景",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 4: 实时行情（时间敏感度最高）
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.REALTIME_QUERY: {
        "description": (
            "查询实时股价、涨跌幅、成交量等最新市场数据。"
            "时间敏感性极高，应优先调用实时数据源而非历史缓存。"
        ),
        "keywords": [
            "股价", "价格", "今天", "现在", "当前",
            "实时", "涨跌", "涨幅", "跌幅",
            "行情", "多少", "最新",
            "成交量", "成交额", "市值",
        ],
        "sub_intents": [
            SecondaryIntent.OVERVIEW,
        ],
        "examples": [
            "茅台今天股价多少",
            "比亚迪现在的价格",
            "今天大盘怎么样",
            "查询宁德时代最新行情",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 5: 投资组合分析
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.PORTFOLIO_ANALYSIS: {
        "description": (
            "投资组合分析、资产配置优化建议。"
            "需要用户提供持仓数据或从外部系统导入。"
        ),
        "keywords": [
            "组合", "持仓", "资产配置", "投资组合",
            "仓位", "分散化", "优化",
            "再平衡", "风险收益", "夏普",
        ],
        "sub_intents": [
            SecondaryIntent.FUNDAMENTAL,
            SecondaryIntent.MARKET_RISK,
        ],
        "examples": [
            "分析我的投资组合风险",
            "如何优化股票持仓结构",
            "评估这个组合的收益风险比",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 6: 风险预警
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.RISK_ALERT: {
        "description": (
            "风险识别、异常检测、压力测试、负面因素预警。"
            '防御型意图，关注"可能出什么问题"而非"表现如何"。'
        ),
        "keywords": [
            "风险", "预警", "警报",
            "暴跌", "暴雷", "异常",
            "黑天鹅", "回撤", "止损",
            "危险", "隐患", " downside",
        ],
        "sub_intents": [
            SecondaryIntent.MARKET_RISK,
            SecondaryIntent.CREDIT_RISK,
            SecondaryIntent.OPERATIONAL_RISK,
        ],
        "examples": [
            "茅台有哪些潜在风险",
            "监测比亚迪的负面新闻",
            "这个股票会暴雷吗",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 7: 策略回测
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.STRATEGY_BACKTEST: {
        "description": (
            "投资策略的历史回测与绩效验证。"
            "用户描述交易逻辑，系统用历史数据模拟执行并计算绩效指标。"
        ),
        "keywords": [
            "回测", "策略", "历史表现",
            "模拟交易", "收益率",
            "最大回撤", "夏普比率",
            "胜率", "盈亏比", "backtest",
        ],
        "sub_intents": [
            SecondaryIntent.TECHNICAL,
            SecondaryIntent.FUNDAMENTAL,
        ],
        "examples": [
            "回测均线交叉策略",
            "价值投资策略的历史表现",
            "模拟动量策略的收益",
        ],
    },

    # ─────────────────────────────────────────────────────────
    # 意图 8: 市场情绪
    # ─────────────────────────────────────────────────────────
    PrimaryIntent.MARKET_SENTIMENT: {
        "description": (
            "市场情绪分析、舆情监控、投资者心理状态评估。"
            "结合行为金融学理论，从非理性角度理解市场动态。"
        ),
        "keywords": [
            "情绪", "舆情", "sentiment",
            "看多", "看空", "做多", "做空",
            "恐慌指数", "贪婪恐惧",
            "资金流向", "北向资金", "情绪面",
        ],
        "sub_intents": [
            SecondaryIntent.OVERVIEW,
            SecondaryIntent.TREND_PREDICTION,
        ],
        "examples": [
            "现在市场情绪如何",
            "投资者对新能源怎么看",
            "分析今日市场舆情",
        ],
    },
}


# ════════════════════════════════════════════════════════════
# 第四部分：意图分类器核心类
# Part 4: Intent Classifier Core Class
#
# 本模块唯一的公共 API 提供者。
# 外部调用方式：classifier = EnhancedIntentClassifier(); result = classifier.classify(query)
# ════════════════════════════════════════════════════════════


class EnhancedIntentClassifier:
    """
    增强版金融意图分类器 (Enhanced Financial Intent Classifier)

    【类职责 / Class Responsibility】
    作为本模块的门面 (Facade)，封装完整的「规则分类 → LLM 分类 →
    融合排序 → 歧义检测」四阶段流水线。对外提供唯一的 classify() 方法。

    【流水线概览 / Pipeline Overview】

    Input: query ("分析茅台2023年的ROE")
        │
        ▼
    ┌─ Stage 1: Rule-Based ───────────────────────┐
    │  遍历 INTENT_DEFINITIONS 的 keywords          │
    │  匹配计数 → base_confidence = 0.3 + n×0.1    │
    │  Output: Top-5 rule_candidates (< 10ms)       │
    └──────────────────────────────────────────────┘
        │
        ▼
    ┌─ Stage 2: LLM-Based ────────────────────────┐
    │  GLM-4-plus 语义理解                         │
    │  注入 query + context + slots               │
    │  Output: Top-3 llm_candidates (~1-2s)        │
    └──────────────────────────────────────────────┘
        │
        ▼
    ┌─ Stage 3: Merge & Rank ─────────────────────┐
    │  加权融合：rule × 0.4 + llm × 0.6           │
    │  归一化（confidence 总和 = 1.0）             │
    │  Output: Top-5 final_candidates              │
    └──────────────────────────────────────────────┘
        │
        ▼
    ┌─ Stage 4: Build Result ─────────────────────┐
    │  取第一名作为 primary_intent                  │
    │  检测歧义（top-2 差 < 0.15 且 best < 0.6）   │
    │  生成消歧问题                                 │
    │  Output: IntentClassificationResult           │
    └──────────────────────────────────────────────┘

    【线程安全 / Thread Safety】
    - 无状态设计（self._llm 和 self._keyword_patterns 初始化后只读）
    - 可在多线程环境中共享同一实例

    [温度参数 rationale / Temperature Rationale]
    temperature=0（分类任务是确定性任务，不需要创造性）
    """

    def __init__(self) -> None:
        """
        初始化意图分类器实例。

        【初始化内容 / Initialization Tasks】
        1. 创建 LLM 客户端（ChatOpenAI，temperature=0）
        2. 预编译所有关键词正则表达式（提升规则匹配性能）
        """
        self._llm: ChatOpenAI = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )

        self._keyword_patterns: Dict[PrimaryIntent, List[re.Pattern]] = {}
        for intent, definition in INTENT_DEFINITIONS.items():
            patterns: List[re.Pattern] = [
                re.compile(re.escape(kw), re.IGNORECASE)
                for kw in definition["keywords"]
            ]
            self._keyword_patterns[intent] = patterns

    def classify(
        self,
        query: str,
        context: Optional[str] = None,
        slots: Optional[Dict[str, Any]] = None,
    ) -> IntentClassificationResult:
        """
        执行意图分类的入口方法。

        [Args / 参数]
            query:   用户查询文本（改写后的或原始查询均可）
            context: 多轮对话上下文（可选，用于代词消解和省略补全）
            slots:   上游提取的结构化槽位信息（可选，辅助缩小意图范围）

        [Returns / 返回值]
            IntentClassificationResult: 包含 primary_intent/candidates/
                                       ambiguity_detected 等完整信息的分类结果
        """
        logger.info("[意图识别] 开始处理: %s...", query[:50])

        rule_candidates: List[IntentCandidate] = self._classify_by_rules(query)
        llm_candidates: List[IntentCandidate] = self._classify_by_llm(
            query, context, slots
        )
        final_candidates: List[IntentCandidate] = self._merge_and_rank(
            rule_candidates, llm_candidates
        )
        result: IntentClassificationResult = self._build_result(
            final_candidates, query, context
        )

        logger.info(
            "[意图识别] 完成: %s (置信度: %.2f)",
            result.primary_intent.value,
            result.confidence,
        )
        if result.ambiguity_detected:
            logger.warning("[意图识别] ⚠️ 检测到歧义，需要消疑")

        return result

    # ─────────────────────────────────────────────────────
    # Stage 1: 基于规则的快速分类
    # Rule-Based Fast Classification
    # ─────────────────────────────────────────────────────

    def _classify_by_rules(self, query: str) -> List[IntentCandidate]:
        """
        通过关键词匹配执行快速的意图初筛。

        【算法 / Algorithm】
        FOR each PrimaryIntent in INTENT_DEFINITIONS:
            match_count = 0
            FOR each keyword in intent's keyword list:
                IF keyword found in query (case-insensitive):
                    match_count += 1
            IF match_count > 0:
                confidence = min(0.3 + match_count * 0.1, 0.7)
                create IntentCandidate with this confidence

        【置信度公式 rationale / Confidence Formula Rationale】
        - 基础分 0.3：只要命中 1 个关键词就给予一定的初始置信度
        - 每个额外关键词 +0.1：多关键词同时命中增强可信度
        - 上限 0.7：规则匹配本身有局限性（无法处理语义相似但词汇不同的表达），
                       保留空间给 LLM 可能给出的更高分数

        [Args / 参数]
            query: 待分类的查询文本

        [Returns / 返回值]
            List[IntentCandidate]: 按 confidence 降序排列的 Top-5 候选列表
        """
        candidates: List[IntentCandidate] = []

        for intent, patterns in self._keyword_patterns.items():
            match_count: int = 0
            matched_keywords: List[str] = []

            for pattern in patterns:
                if pattern.search(query):
                    match_count += 1
                    matched_keywords.append(pattern.pattern)

            if match_count > 0:
                base_confidence: float = min(0.3 + match_count * 0.1, 0.7)
                candidate = IntentCandidate(
                    primary=intent,
                    secondary=None,
                    confidence=base_confidence,
                    reasoning=f"匹配到{match_count}个关键词: {matched_keywords[:3]}",
                )
                candidates.append(candidate)

        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[:5]

    # ─────────────────────────────────────────────────────
    # Stage 2: 基于 LLM 的语义分类
    # LLM-Based Semantic Classification
    # ─────────────────────────────────────────────────────

    def _classify_by_llm(
        self,
        query: str,
        context: Optional[str] = None,
        slots: Optional[Dict[str, Any]] = None,
    ) -> List[IntentCandidate]:
        """
        使用 LLM 执行语义级别的意图分类。

        [Prompt Engineering 要点 / Prompt Key Points]
        - System Prompt: 定义 8 类主意图 + 12 类子意图的完整列表和含义
        - User Prompt: 注入 query + context（如有）+ slots JSON（如有）
        - Output Format: JSON 数组，每个元素包含 primary/secondary/confidence/reasoning
        - 要求返回 Top-3（平衡精度和 token 成本）

        [Args / 参数]
            query:   待分类的查询文本
            context: 对话上下文（可选）
            slots:   槽位信息（可选）

        [Returns / 返回值]
            List[IntentCandidate]: LLM 推断的 Top-3 候选列表（失败时返回空列表）
        """
        system_prompt: str = (
            "你是一个专业的金融问题意图分类器。\n"
            "请将用户问题归类为以下8种主意图之一，并选择最合适的子意图。\n\n"
            "主意图类别：\n"
            "1. financial_analysis - 财务分析（财报解读、指标分析、经营数据）\n"
            "2. stock_comparison - 个股对比（多公司比较、竞争力分析）\n"
            "3. industry_overview - 行业综述（行业趋势、竞争格局、产业链）\n"
            "4. realtime_query - 实时行情（股价查询、涨跌、最新数据）\n"
            "5. portfolio_analysis - 组合分析（持仓优化、风险评估）\n"
            "6. risk_alert - 风险预警（风险识别、异常检测）\n"
            "7. strategy_backtest - 策略回测（历史验证、绩效评估）\n"
            "8. market_sentiment - 市场情绪（舆情分析、投资者心理）\n\n"
            "子意图类别：\n"
            "- profitability_analysis - 盈利能力分析\n"
            "- growth_analysis - 成长性分析\n"
            "- cashflow_analysis - 现金流分析\n"
            "- valuation_analysis - 估值分析\n"
            "- technical_analysis - 技术面分析\n"
            "- fundamental_analysis - 基本面分析\n"
            "- overview - 概览性分析\n"
            "- deep_dive - 深度分析\n"
            "- trend_prediction - 趋势预测\n"
            "- credit_risk - 信用风险\n"
            "- market_risk - 市场风险\n"
            "- operational_risk - 经营风险\n\n"
            "输出要求：\n"
            "以JSON格式返回Top-3候选意图，包含：\n"
            "- primary_intent: 主意图标签\n"
            "- secondary_intent: 子意图标签（可为null）\n"
            "- confidence: 置信度（0-1，总和约为1.0）\n"
            "- reasoning: 判断依据"
        )

        user_prompt_parts: List[str] = [f"用户问题：{query}"]

        if context:
            user_prompt_parts.append(f"\n对话上下文：{context}")

        if slots:
            user_prompt_parts.append(
                f"\n已知实体信息：\n"
                f"{json.dumps(slots, ensure_ascii=False, indent=2)}"
            )

        user_prompt_parts.append("\n\n请输出JSON格式的分类结果（只输出JSON）：")
        user_prompt: str = "\n".join(user_prompt_parts)

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            content: str = response.content.strip()
            content = self._clean_json_response(content)
            data: Any = json.loads(content)

            return self._parse_llm_response(data)

        except json.JSONDecodeError as e:
            logger.error("[意图识别] LLM 返回的 JSON 解析失败: %s", e)
            return []
        except Exception as e:
            logger.error("[意图识别] LLM 分类失败: %s", e)
            return []

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """清理 LLM 返回中的 Markdown 代码块标记"""
        for prefix in ("```json", "```"):
            if content.startswith(prefix):
                content = content[len(prefix):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    @staticmethod
    def _parse_llm_response(data: Any) -> List[IntentCandidate]:
        """
        解析 LLM 返回的 JSON 数据为 IntentCandidate 列表。

        支持两种格式：
        1. 列表格式：[{primary_intent: ..., ...}, ...]（标准 Top-N 格式）
        2. 字典格式：{primary_intent: ..., ...}（单结果兼容格式）
        """
        candidates: List[IntentCandidate] = []

        items: Any
        if isinstance(data, list):
            items = data[:3]
        elif isinstance(data, dict):
            items = [data]
        else:
            return candidates

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                primary = PrimaryIntent(item["primary_intent"])
                secondary: Optional[SecondaryIntent] = None
                if item.get("secondary_intent"):
                    try:
                        secondary = SecondaryIntent(item["secondary_intent"])
                    except ValueError:
                        pass

                candidate = IntentCandidate(
                    primary=primary,
                    secondary=secondary,
                    confidence=float(item.get("confidence", 0)),
                    reasoning=item.get("reasoning", ""),
                )
                candidates.append(candidate)
            except (KeyError, ValueError):
                continue

        return candidates

    # ─────────────────────────────────────────────────────
    # Stage 3: 结果融合与排序
    # Result Merging & Ranking
    # ─────────────────────────────────────────────────────

    def _merge_and_rank(
        self,
        rule_candidates: List[IntentCandidate],
        llm_candidates: List[IntentCandidate],
    ) -> List[IntentCandidate]:
        """
        融合规则和 LLM 的分类结果，并进行加权排序。

        【融合权重 rationale / Merge Weight Rationale】
        - 规则权重 0.4：确定性高但泛化弱（只能匹配预定义关键词）
        - LLM 权重 0.6：泛化强但有幻觉风险（温度=0 已降低此风险）
        - 两者互补：规则覆盖高频明确表达，LLM 覆盖长尾复杂语义

        【归一化 / Normalization】
        融合后的 confidence 先截断到 [0, 1]，再做 softmax 式归一化
        （除以总和），使最终 confidence 表示相对排名而非绝对概率。

        [Args / 参数]
            rule_candidates: Stage 1 规则分类的候选列表
            llm_candidates:  Stage 2 LLM 分类的候选列表

        [Returns / 返回值]
            List[IntentCandidate]: 融合后按 confidence 降序排列的 Top-5 候选
        """
        merged: Dict[PrimaryIntent, Dict[str, Any]] = {}

        for candidate in rule_candidates:
            key = candidate.primary
            if key not in merged:
                merged[key] = {
                    "conf": candidate.confidence * 0.4,
                    "reasons": [candidate.reasoning],
                    "secondary": candidate.secondary,
                }
            else:
                merged[key]["conf"] += candidate.confidence * 0.4
                merged[key]["reasons"].append(candidate.reasoning)

        for candidate in llm_candidates:
            key = candidate.primary
            if key not in merged:
                merged[key] = {
                    "conf": candidate.confidence * 0.6,
                    "reasons": [candidate.reasoning],
                    "secondary": candidate.secondary,
                }
            else:
                merged[key]["conf"] += candidate.confidence * 0.6
                merged[key]["reasons"].append(candidate.reasoning)
                if candidate.secondary and not merged[key]["secondary"]:
                    merged[key]["secondary"] = candidate.secondary

        final_candidates: List[IntentCandidate] = []
        for intent, data in merged.items():
            c = IntentCandidate(
                primary=intent,
                secondary=data["secondary"],
                confidence=min(data["conf"], 1.0),
                reasoning="; ".join(data["reasons"][:2]),
            )
            final_candidates.append(c)

        total_conf: float = sum(c.confidence for c in final_candidates)
        if total_conf > 0:
            for c in final_candidates:
                c.confidence = c.confidence / total_conf

        final_candidates.sort(key=lambda x: x.confidence, reverse=True)
        return final_candidates[:5]

    # ─────────────────────────────────────────────────────
    # Stage 4: 结果构建与歧义检测
    # Result Building & Ambiguity Detection
    # ─────────────────────────────────────────────────────

    def _build_result(
        self,
        candidates: List[IntentCandidate],
        query: str,
        context: Optional[str] = None,
    ) -> IntentClassificationResult:
        """
        从候选列表构建最终的 IntentClassificationResult。

        【歧义检测算法 / Ambiguity Detection Algorithm】
        同时满足以下三个条件时判定为歧义：
        1. 候选数 >= 2（至少有两个可能的解释）
        2. 第一名置信度 < 0.6（最佳匹配不够确信）
        3. |第一名 - 第二名| < 0.15（前两名非常接近，难以区分）

        [Args / 参数]
            candidates: 融合排序后的候选列表
            query:      原始查询（用于日志记录）
            context:   对话上下文（用于标记 context_aware 标志）

        [Returns / 返回值]
            IntentClassificationResult: 完整的分类结果
        """
        if not candidates:
            fallback = IntentCandidate(
                primary=PrimaryIntent.FINANCIAL_ANALYSIS,
                secondary=None,
                confidence=1.0,
                reasoning="无法分类，使用默认意图（财务分析）",
            )
            candidates = [fallback]

        best: IntentCandidate = candidates[0]

        candidate_list: List[Dict[str, Any]] = [
            {
                "primary_intent": c.primary.value,
                "secondary_intent": c.secondary.value if c.secondary else None,
                "confidence": round(c.confidence, 4),
                "reasoning": c.reasoning,
            }
            for c in candidates
        ]

        hierarchy: Dict[str, Any] = {
            "primary": best.primary.value,
            "secondary": best.secondary.value if best.secondary else None,
            "confidence": round(best.confidence, 4),
            "alternatives": candidate_list[1:],
            "definition": INTENT_DEFINITIONS.get(
                best.primary, {}
            ).get("description", ""),
        }

        ambiguity_detected: bool = (
            len(candidates) >= 2
            and candidates[0].confidence < 0.6
            and abs(candidates[0].confidence - candidates[1].confidence) < 0.15
        )

        disambiguation_questions: List[str] = []
        if ambiguity_detected:
            for i, candidate in enumerate(candidates[1:3], 1):
                intent_def = INTENT_DEFINITIONS.get(candidate.primary, {})
                question = (
                    f"您是想了解"
                    f"{intent_def.get('description', '相关信息')}吗？"
                )
                disambiguation_questions.append(question)

        return IntentClassificationResult(
            success=True,
            primary_intent=best.primary,
            secondary_intent=best.secondary,
            confidence=round(best.confidence, 4),
            candidates=candidate_list,
            intent_hierarchy=hierarchy,
            classification_method="hybrid",
            ambiguity_detected=ambiguity_detected,
            disambiguation_questions=disambiguation_questions,
            context_aware=context is not None,
        )


# ════════════════════════════════════════════════════════════
# 第五部分：公共便捷函数
# Part 5: Public Convenience Function
# ════════════════════════════════════════════════════════════


# Module-level singleton to avoid creating new LLM clients on every call
_classifier_instance: Optional["EnhancedIntentClassifier"] = None


def classify_intent(
    query: str,
    context: Optional[str] = None,
    slots: Optional[Dict[str, Any]] = None,
) -> IntentClassificationResult:
    """
    便捷函数：一步完成意图分类。

    [Args / 参数]
        query:   用户查询文本
        context: 对话上下文（可选）
        slots:   槽位信息（可选）

    [Returns / 返回值]
        IntentClassificationResult: 完整的分类结果

    [Example / 示例]
        >>> result = classify_intent("分析茅台2023年ROE")
        >>> print(result.primary_intent.value)
        'financial_analysis'
        >>> print(result.confidence)
        0.92
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EnhancedIntentClassifier()
    return _classifier_instance.classify(query, context, slots)


# ════════════════════════════════════════════════════════════
# 第六部分：模块自测
# Part 6: Self-Test Suite
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_queries: List[str] = [
        "帮我分析一下贵州茅台2023年的盈利能力和现金流状况",
        "对比一下茅台和五粮液近三年的估值水平",
        "比亚迪今天的股价是多少？",
        "新能源汽车行业的发展趋势和竞争格局如何？",
        "我的投资组合风险大不大？",
        "回测一下均线交叉策略的表现",
        "现在市场情绪怎么样？",
        "这只股票会有风险吗？",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"输入: {query}")
        result = classify_intent(query)
        print(f"主意图: {result.primary_intent.value}")
        print(
            f"子意图: "
            f"{result.secondary_intent.value if result.secondary_intent else None}"
        )
        print(f"置信度: {result.confidence:.2f}")
        print(f"候选数: {len(result.candidates)}")
        if result.ambiguity_detected:
            print(f"⚠️ 歧义: {result.disambiguation_questions}")
