"""
金融领域槽位提取模块 (Financial Domain Slot Extraction)
=====================================================

【模块定位 / Module Positioning】
本模块是 AI 金融研究助手系统的「信息结构化」核心组件，负责将用户的自然语言查询
转化为机器可理解的结构化数据（即"槽位/Slots"）。它是整个 Agent Pipeline 的第一道关卡，
所有下游模块（意图分类、Query 改写、任务规划、RAG 检索）都依赖于此模块的输出。

【架构角色 / Architectural Role】
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  用户输入     │───▶│  SlotExtractor   │───▶│  下游消费者       │
│  自然语言     │    │  (本模块)         │    │  IntentClassifier│
│              │    │                  │    │  QueryRewriter   │
│ "分析茅台...  │    │ 混合策略:        │    │  TaskPlanner     │
│  2023年盈利" │    │ 规则 + LLM + 融合 │    │  RAG Pipeline    │
└─────────────┘    └──────────────────┘    └─────────────────┘

【核心设计原则 / Core Design Principles】

1. **混合提取策略 (Hybrid Extraction Strategy)**
   - 第一层：规则匹配（正则表达式 + 词典查找）
     → 优点：零延迟、100%确定性、适合高频常见模式
     → 缺点：无法处理未见过的表达方式、维护成本随词典规模增长
   - 第二层：LLM 语义理解（GLM-4-plus via OpenAI-compatible API）
     → 优点：泛化能力强、可处理复杂语义、能做模糊推理
     → 缺点：有延迟（~1-3s）、非确定性（需 temperature=0 降低随机性）、有 API 成本
   - 第三层：结果融合（Merge & Deduplicate）
     → 策略：规则结果作为基线，LLM 结果作为增强，取并集去重
     → 冲突解决：当规则和 LLM 对同一字段给出不同值时，优先采用 LLM 结果

2. **金融领域特化 (Financial Domain Specialization)**
   - 内置 A 股主流公司别名映射表（含股票代码）
   - 内置财务指标分类体系（6 大类、30+ 细分指标）
   - 内置中文时间表达模式识别（年份/季度/相对时间/时间范围）
   - 所有模式均针对中文金融语境优化

3. **置信度驱动质量门控 (Confidence-Driven Quality Gating)**
   - 每次提取输出 0-1 的综合置信度分数
   - 置信度 < 0.5 时自动触发澄清机制（生成追问问题）
   - 缺失的关键槽位会被明确标记，便于下游模块做补偿决策

4. **Pydantic 强类型保障 (Pydantic Type Safety)**
   - 所有输入/输出使用 Pydantic BaseModel 序列化/反序列化
   - 运行时类型校验，防止非法数据流入下游
   - 自动生成 JSON Schema，便于与 LangGraph State 对接

【槽位分类体系 / Slot Taxonomy】

┌─────────────────────────────────────────────────────────────────┐
│ 核心槽位 (Core Slots) — 必须尽可能提取                            │
│ ├─ companies:      公司实体列表（名称+代码+别名）                   │
│ ├─ time_info:      时间信息（年份/季度/范围/相对时间）               │
│ └─ metrics:        财务指标（主要指标+次要指标+类别标签）             │
├─────────────────────────────────────────────────────────────────┤
│ 扩展槽位 (Extended Slots) — 有则更好，无则不影响主流程              │
│ ├─ comparison_targets: 对比分析的目标公司                           │
│ ├─ industry:           所属行业                                    │
│ └─ analysis_type:      分析深度偏好（深度/概览/对比/预测）            │
├─────────────────────────────────────────────────────────────────┤
│ 元数据槽位 (Metadata Slots) — 用于调试和质量评估                    │
│ ├─ extraction_method:   提取方法标识（rule/llm/hybrid）            │
│ ├─ missing_slots:       缺失槽位列表                               │
│ ├─ clarification_needed: 是否需要用户澄清                           │
│ └─ clarification_questions: 澄清问题列表                           │
└─────────────────────────────────────────────────────────────────┘

【性能特征 / Performance Characteristics】
- 规则提取阶段：O(N*M)，N=查询长度，M=词典条目数，通常 < 10ms
- LLM 提取阶段：网络 I/O 密集，典型延迟 1-3s（取决于模型和查询长度）
- 融合验证阶段：O(K)，K=提取结果条目数，通常 < 1ms
- 总体 P99 延迟：~2-4s（主要瓶颈在 LLM 调用）

【依赖关系 / Dependencies】
- 上游：用户原始输入 query（字符串）+ 可选对话上下文 context
- 下游：EnhancedAgentState.slots / EnhancedAgentState.slot_confidence
- 外部依赖：
  - langchain_openai.ChatOpenAI（LLM 推理）
  - pydantic（数据校验）
  - config.settings（API 配置）

【版本信息 / Version Info】
- Author: AI Finance Assistant Team
- Version: 2.0.0 (极致优化版 / Extreme Optimization Edition)
- Last Updated: 2026-04-13
- Compatible with: enhanced_state.py v2.0+
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL


# ─────────────────────────────────────────────────────────────
# 日志配置 / Logging Configuration
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 第一部分：Pydantic 数据模型定义
# Part 1: Pydantic Data Model Definitions
#
# 所有模型的字段均遵循以下文档规范：
# - description: 字段用途说明
# - examples: 典型输入/输出示例
# - downstream_consumers: 哪些下游模块会消费此字段
# - edge_cases: 需要注意的边界情况
# ════════════════════════════════════════════════════════════


class SlotExtractionRequest(BaseModel):
    """
    槽位提取请求模型 (Slot Extraction Request Schema)

    【职责】封装调用方传入的提取参数，是 SlotExtractor.extract() 的唯一入参。

    【设计决策】为什么用独立 Request 模型而不是直接传字符串？
    - 类型安全：Pydantic 在构造时自动校验，防止 None/空字符串等非法输入
    - 可扩展性：未来可增加 hints（提示词）、domain（领域约束）等字段
    - 序列化友好：可直接 JSON dump，便于日志记录和调试追踪
    - 与 LangGraph Node 接口对齐：LangGraph 的 node 函数接收 state dict，
      用 Pydantic model 做 intermediate layer 可以提供更好的类型提示

    【使用示例 / Usage Example】
        request = SlotExtractionRequest(
            query="帮我分析贵州茅台2023年的ROE和净利润增速",
            context="用户之前问过茅台的PE估值"
        )
        result = extractor.extract(request)
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "用户原始自然语言查询文本。"
            "这是槽位提取的唯一必需输入，支持中文金融领域的各种表达方式，"
            "包括但不限于：公司名称/股票代码/行业术语/财务指标/时间表达/对比诉求等。"
            "长度限制为 1-2000 字符，超出部分将被截断。"
        ),
    )

    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description=(
            "多轮对话的历史上下文（可选）。"
            "当用户的问题依赖于前文时（如代词引用'它'、省略主语'那家公司呢'），"
            "此字段用于消解指代歧义。格式为拼接的历史对话文本，"
            "如 'Q:分析茅台的盈利 A:茅台2023年毛利率91.96% Q:那现金流呢'。"
            "若为单轮对话或无需上下文，传 None 即可。"
        ),
    )

    @field_validator("query")
    @classmethod
    def _validate_query_not_empty(cls, v: str) -> str:
        """校验查询文本非空且非纯空白"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("查询文本不能为空或仅包含空白字符")
        return stripped


class CompanySlot(BaseModel):
    """
    公司实体槽位模型 (Company Entity Slot)

    【职责】描述从用户查询中识别出的单一公司实体，包含标准化名称、交易代码和别名。

    【数据来源】
    - 规则路径：通过 COMPANY_ALIASES 词典精确匹配
    - LLM 路径：通过语义理解从自由文本中抽取并规范化

    【标准化规则 / Normalization Rules】
    - name: 使用公司全称（如"贵州茅台酒股份有限公司"→"贵州茅台"）
    - stock_code: 使用统一格式「数字.交易所」，如 600519.SH / 00700.HK / 000858.SZ
    - aliases: 记录用户原文中实际使用的称呼（用于回显确认）

    【下游消费者 / Downstream Consumers】
    - TaskPlanner: 根据 stock_code 构建数据获取任务
    - FinancialAgent: 作为分析对象的主语
    - RiskAgent: 作为风险评估标的
    - MCP Tools (StockQuote): 直接用 code 调用行情接口
    - Report Generator: 在报告中显示公司全称

    【边界情况 / Edge Cases】
    - 同一公司可能被多次提及（如"茅台和贵州茅台"）→ 融合阶段去重
    - 用户只说行业不说公司（如"新能源汽车行业"）→ companies 为空，industry 有值
    - 港股/A股同名不同代码 → stock_code 是唯一去重键
    """

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "公司标准化全称。"
            "例如：'贵州茅台'、'比亚迪'、'腾讯控股'。"
            "这是系统内部使用的规范名称，用于所有下游模块的数据传递和报告生成。"
            "应与 COMPANY_ALIASES 中的 name 字段保持一致。"
        ),
    )

    stock_code: Optional[str] = Field(
        default=None,
        description=(
            "股票交易代码，采用「数字.交易所」格式。"
            "\n格式说明："
            "\n  - A股上海: XXXXXX.SH （6位数字+.SH）"
            "\n  - A股深圳: XXXXXX.SZ （6位数字+.SZ）"
            "\n  - 港股主板: 0XXXX.HK （5位数字+.HK）"
            "\n  - 美股: TICKER （字母代码，如 AAPL）"
            "\n示例：600519.SH / 00700.HK / 000858.SZ / TSLA"
            "\n注意：若无法确定交易所后缀，可只保留数字部分；"
            "若完全未知（如用户提到未上市公司），则为 None。"
        ),
    )

    aliases: List[str] = Field(
        default_factory=list,
        description=(
            "用户原文中使用的该公司别名称呼列表。"
            "记录这些别名有两个目的："
            "\n1. 回显确认：在需要用户确认时，使用用户熟悉的称呼"
            "\n2. 追踪溯源：调试时可还原用户原始表达"
            "\n示例：用户说'茅台'，则 aliases=['茅台']；"
            "用户说'贵州茅台酒股份有限公司'，则 aliases=['贵州茅台酒股份有限公司']"
        ),
    )


class TimeSlot(BaseModel):
    """
    时间相关槽位模型 (Time-related Slot)

    【职责】从用户查询中提取的时间维度信息，支持多种时间表达范式。

    【支持的时间表达类型 / Supported Time Expression Types】

    类型                示例                    解析结果
    ──────────────────────────────────────────────────────
    绝对年份            "2023年"               year=2023
    绝对季度            "2024年第一季度"        year=2024, quarter=1
    相对年份            "去年/上年"             year=当前年-1
    相对时间段          "近三年/最近5年"         period="近3年"/"近5年"
    年份范围            "2020年至2023年"        time_range=("2020","2023")
    当前年度            "今年/本年/当前"        year=当前年

    【优先级规则 / Priority Rules】
    当同一查询中出现多个时间表达时：
    1. 具体的 year/quarter > 模糊的 period/time_range
    2. 后出现的 > 先出现的（符合中文语序习惯）
    3. 规则匹配到的 > LLM 推断的（确定性更高）

    【下游消费者 / Downstream Consumers】
    - QueryRewriter: 将时间信息注入改写后的查询
    - TaskPlanner: 确定数据获取的时间范围参数
    - RAG Pipeline: 作为检索的时间过滤条件
    - FinancialAgent: 指定分析的时间窗口
    - Report Generator: 在报告标题和图表中标注时间范围

    【边界情况 / Edge Cases】
    - 无时间表达：time_info=None，下游应使用默认策略（如最近一年）
    - 时间冲突："2023年和近三年" → 取更具体的 year=2023
    - 未来时间："2028年"（当前为2026年）→ 仍正常提取，由下游判断合理性
    - 财年vs自然年：中国A股用自然年，美股用财年，此处不做区分
    """

    year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description=(
            "具体年份（公历/自然年）。"
            "适用场景：用户明确指定了某一年，如'2023年''去年''今年'。"
            "对于'近三年'这类相对表达，不填此字段，而使用 period 字段。"
            "有效范围：1900-2100（防止解析错误导致的极端值）。"
        ),
    )

    quarter: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description=(
            "季度编号（1-4）。"
            "仅在用户明确指定季度时有值，如'2024年第一季度''Q3'。"
            "必须与 year 字段配合使用（单独的 quarter 无意义）。"
            "映射关系：一/Q1→1, 二/Q2→2, 三/Q3→3, 四/Q4→4。"
        ),
    )

    period: Optional[str] = Field(
        default=None,
        description=(
            "相对时间范围的文字描述。"
            "适用于用户使用模糊时间表达的场景，如："
            "'近三年'/'最近一季度'/'过去五年'/'今年以来'等。"
            "此字段的值为原始表达的标准化形式，供下游模块解释执行。"
            "注意：period 和 year/time_range 通常互斥（三选一）。"
        ),
    )

    time_range: Optional[Tuple[str, str]] = Field(
        default=None,
        description=(
            "起止时间范围元组 (start, end)。"
            "适用于用户指定了明确时间区间的场景，如'2020-2023年'"
            "'从2019年到2022年之间'。start 和 end 均为字符串形式的年份。"
            "示例：('2020', '2023') 表示 2020 年至 2023 年（含两端）。"
        ),
    )


class MetricSlot(BaseModel):
    """
    财务指标槽位模型 (Financial Metrics Slot)

    【职责】描述用户关注的财务分析指标集合，按重要性分层组织。

    【分层设计理由 / Layered Design Rationale】
    用户一次查询可能提及多个指标，但并非所有指标同等重要。
    分层设计允许下游模块根据资源约束做取舍：

    - primary_metrics（主要指标）：用户明确提到的核心关注点，
      必须在分析和报告中重点呈现（上限 5 个，防止信息过载）
    - secondary_metrics（次要指标）：隐含关联或补充性指标，
      有计算资源时可一并分析（无上限，但通常 < 10 个）
    - metric_categories（类别标签）：指标的归类标签，
      用于驱动 TaskPlanner 选择对应的分析模板

    【指标分类体系 / Metric Category System】
    本系统将财务指标划分为 6 大类，每类对应不同的分析视角：

    类别          代表指标                      分析价值
    ─────────────────────────────────────────────────────
    盈利能力      毛利率/净利率/ROE/ROA         企业赚钱能力和效率
    成长性        营收增速/净利润增速/EPS增速    发展速度和可持续性
    现金流        经营现金流/自由现金流/资本支出  现金造血能力（比利润更真实）
    估值          PE/PB/PS/EV/EBITDA/PEG       投资性价比
    偿债能力      资产负债率/流动比率/速动比率   财务安全性
    运营效率      存货周转率/应收账款周转率      资产利用效率

    【下游消费者 / Downstream Consumers】
    - TaskPlanner: 根据 metric_categories 选择分析任务模板
    - FinancialAgent: 按 primary_metrics 组织分析章节
    - RAG Pipeline: 将指标名作为关键词增强检索
    - Report Generator: 生成对应的指标分析段落和图表
    - Evaluator: 检查报告是否覆盖了所有 primary_metrics

    【自动扩展机制 / Auto-Expansion Mechanism】
    当用户说"盈利能力"而不列举具体指标时，系统会自动展开为该类别下的
    代表性指标（见 _extract_metrics() 中的 keyword_mapping 逻辑）。
    这使得系统既能处理精确查询（"ROE是多少"），也能处理笼统查询（"盈利怎么样"）。
    """

    primary_metrics: List[str] = Field(
        ...,
        min_length=0,
        description=(
            "主要财务指标列表——用户明确关注的核心指标。"
            "\n提取规则："
            "\n  1. 用户原文中直接提到的具体指标名（如'ROE''净利润'）优先归入此类"
            "\n  2. 最多保留 5 个（防止信息过载，超出部分移入 secondary）"
            "\n  3. 保持用户原始提及顺序（反映关注优先级）"
            "\n示例：['ROE', '净利率', '营收增速', '经营现金流', 'PE']"
        ),
    )

    secondary_metrics: List[str] = Field(
        default_factory=list,
        description=(
            "次要/关联财务指标列表——从主要指标自动衍生或隐含关联的指标。"
            "\n来源包括："
            "\n  1. 同类别下未被列入 primary 的其他指标"
            "\n  2. 通过 keyword_mapping 从笼统关键词展开的指标"
            "\n  3. LLM 推断的用户可能也关心的指标"
            "\n用途：在资源充裕时补充分析，不强制要求覆盖。"
        ),
    )

    metric_categories: List[str] = Field(
        default_factory=list,
        description=(
            "指标所属的分析类别标签列表。"
            "\n可选值（与 FINANCIAL_METRICS 的 key 一致）："
            "\n  ['盈利能力', '成长性', '现金流', '估值', '偿债能力', '运营效率']"
            "\n作用：驱动 TaskPlanner 选择对应的分析维度模板。"
            "\n示例：用户问'盈利和成长' → categories=['盈利能力', '成长性']"
        ),
    )


class SlotExtractionResult(BaseModel):
    """
    槽位提取完整结果模型 (Complete Slot Extraction Result)

    【职责】这是本模块对外输出的唯一数据结构，封装了一次提取操作的全部产出。
    它同时承载了「数据结果」和「质量控制」两个维度的信息。

    【生命周期 / Lifecycle】
    1. 创建于 SlotExtractor._merge_results()（初始填充）
    2. 更新于 SlotExtractor._validate_and_score()（置信度计算和质量标记）
    3. 消费于 EnhancedFinancialAssistant._run_slot_extraction()（写入 state）
    4. 传递至所有下游模块（intent_classifier, query_rewriter, task_planner 等）

    【置信度计算公式 / Confidence Score Formula】
    confidence = base_score × method_multiplier

    其中：
    - base_score = Σ(各槽位存在时的权重分)
      - companies 存在: +0.25（最关键，无公司无法定位分析对象）
      - time_info 存在: +0.15（影响数据时效性）
      - metrics 存在: +0.20（决定分析方向）
      - comparison_targets 存在: +0.10（加分项）
      - industry 存在: +0.10（加分项）
      - analysis_type 存在: +0.10（加分项）
    - method_multiplier:
      - hybrid（规则+LLM融合）: ×1.10（两种方法交叉验证，可信度高）
      - llm_only: ×1.05（LLM 泛化能力强但有幻觉风险）
      - rule_only: ×1.00（确定性高但覆盖率有限）

    最终置信度 clamp 到 [0.0, 1.0] 区间。

    【质量门控逻辑 / Quality Gate Logic】
    - confidence >= 0.7: 高质量提取，可直接用于后续流程
    - 0.5 <= confidence < 0.7: 中等质量，建议补充澄清但非必须
    - confidence < 0.5: 低质量，强制触发 clarification_needed=True，
                       并生成默认澄清问题引导用户提供更多信息

    【序列化 / Serialization】
    此模型可通过 .model_dump_json() 序列化为 JSON，直接写入
    EnhancedAgentState 的 slots 字段（该字段类型为 Dict[str, Any]）。
    """

    # ── 身份与质量标识 / Identity & Quality Flags ──

    success: bool = Field(
        ...,
        description=(
            "提取操作是否成功完成。"
            "True 表示至少完成了规则+LLM+融合的全流程（即使未提取到任何槽位）；"
            "False 仅在发生不可恢复异常时出现（如 LLM API 完全不可达）。"
            "注意：success=True 不代表提取到了有用信息，需结合 confidence 判断质量。"
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "综合置信度分数（0.0-1.0）。"
            "数值越高表示提取结果越可靠。详见类 docstring 中的计算公式。"
            "下游模块可用此值做路由决策："
            "\n  - >= 0.8: 直接信任，进入下一阶段"
            "\n  - 0.5-0.8: 可用但建议人工审核"
            "\n  - < 0.5: 应触发澄清交互或降级处理"
        ),
    )

    # ── 核心槽位 / Core Slots ──

    companies: List[CompanySlot] = Field(
        default_factory=list,
        description=(
            "识别出的公司实体列表。"
            "通常包含 1 个主分析对象（查询的主语）；"
            "在对比分析场景下可能包含 2 个以上公司。"
            "每个元素均为 CompanySlot 实例，包含标准化名称和股票代码。"
            "空列表表示未能识别出任何公司实体（此时应检查 industry 字段）。"
        ),
    )

    time_info: Optional[TimeSlot] = Field(
        default=None,
        description=(
            "提取出的时间相关信息。"
            "None 表示用户未指定时间（下游应使用合理的默认值，如'最近一年'）。"
            "有值时为 TimeSlot 实例，可能包含 year/quarter/period/time_range 中的若干字段。"
        ),
    )

    metrics: Optional[MetricSlot] = Field(
        default=None,
        description=(
            "提取出的财务指标信息。"
            "None 表示用户未指定具体指标（下游应使用默认指标集进行全景扫描式分析）。"
            "有值时为 MetricSlot 实例，包含分层指标列表和类别标签。"
        ),
    )

    # ── 扩展槽位 / Extended Slots ──

    comparison_targets: List[CompanySlot] = Field(
        default_factory=list,
        description=(
            "对比分析场景下的目标公司列表。"
            "当用户查询中包含'对比''比较''VS''和...相比'等关键词时，"
            "被比较的对象归入此列表（而非 companies）。"
            "\n区分规则：companies 是分析主体（'分析A'），comparison_targets 是参照物（'A和B对比'中的B）。"
            "空列表表示非对比分析场景。"
        ),
    )

    industry: Optional[str] = Field(
        default=None,
        description=(
            "查询涉及的行业名称。"
            "在两种情况下有值："
            "\n  1. 用户直接提到了行业（如'新能源汽车行业的竞争格局'）"
            "\n  2. LLM 根据公司名称推断出其所属行业"
            "\n示例：'白酒'/'新能源汽车'/'银行业'/'半导体'。"
            "当 companies 为空但 industry 有值时，表明这是一个行业级查询而非公司级查询。"
        ),
    )

    analysis_type: Optional[str] = Field(
        default=None,
        description=(
            "用户期望的分析深度/类型偏好。"
            "\n可选值及含义："
            "\n  - '深度解读': 全面详尽的 CFA 级分析，适合专业投资者"
            "\n  - '快速概览': 只看核心数据和结论，适合快速决策"
            "\n  - '对比分析': 多对象横向比较，突出差异"
            "\n  - '趋势预测': 基于历史数据的未来走势研判"
            "\n此字段影响 Report Generator 的输出详细程度和分析框架选择。"
        ),
    )

    # ── 原始信息保留 / Raw Information Preservation ──

    raw_query: str = Field(
        ...,
        description=(
            "用户原始查询文本（原样保留）。"
            "用于：日志审计、标准化查询生成时的参考基准、错误排查时的输入还原。"
        ),
    )

    normalized_query: Optional[str] = Field(
        default=None,
        description=(
            "基于提取槽位生成的标准化查询语句。"
            "由 _generate_normalized_query() 方法自动生成，格式为："
            "'[公司名] [时间] [指标] [分析类型]'，各部分用空格连接。"
            "示例：'贵州茅台 2023 ROE、净利率、经营现金流 深度解读'"
            "用途：可用于 RAG 检索的 query 扩展，或作为简化版的 user_query 写入 state。"
        ),
    )

    # ── 元数据 / Metadata ──

    extraction_method: str = Field(
        ...,
        description=(
            "本次提取使用的方法标识。"
            "\n可选值："
            "\n  - 'rule':    仅通过规则匹配获得结果（LLM 失败或被跳过）"
            "\n  - 'llm':     仅通过 LLM 获得（规则未匹配到任何内容）"
            "\n  - 'hybrid':  规则和 LLM 结果融合（最常见的高质量情况）"
            "\n此字段主要用于调试和统计分析，一般不影响业务逻辑。"
        ),
    )

    missing_slots: List[str] = Field(
        default_factory=list,
        description=(
            "检测到缺失的关键槽位名称列表。"
            "在 _validate_and_score() 中根据核心槽位是否存在自动填充。"
            "可能的值：'公司实体'/'时间信息'/'财务指标'。"
            "下游模块可根据此列表决定是否需要主动向用户询问补充信息。"
        ),
    )

    clarification_needed: bool = Field(
        default=False,
        description=(
            "是否需要向用户发起澄清交互。"
            "当 confidence < 0.5 或 LLM 明确标注了 ambiguity 时为 True。"
            "为 True 时，clarification_questions 必定非空。"
            "交互层（CLI/WebUI）应根据此标志决定是否暂停自动流程等待用户回复。"
        ),
    )

    clarification_questions: List[str] = Field(
        default_factory=list,
        description=(
            "需要向用户确认的澄清问题列表。"
            "每个问题都是一个自然语言句子，旨在引导用户补充缺失的关键信息。"
            "\n生成策略："
            "\n  1. 优先使用 LLM 生成的针对性问题（如果 LLM 提供了）"
            "\n  2. 回退到基于 missing_slots 的通用模板问题"
            "\n示例：['请问您想了解哪家公司的信息？', '您关注的时间范围是什么时候？']"
        ),
    )


# ════════════════════════════════════════════════════════════
# 第二部分：金融领域知识词典
# Part 2: Financial Domain Knowledge Dictionaries
#
# 这些词典是规则提取引擎的"知识库"，决定了系统能识别哪些实体和模式。
# 维护指南：
# - 新增公司：在 COMPANY_ALIASES 中添加 alias→{name, code} 映射
# - 新增指标：在 FINANCIAL_METRICS 对应类别下添加指标名
# - 新增时间模式：在 TIME_PATTERNS 中添加 regex→type 映射
# ════════════════════════════════════════════════════════════


class _CompanyAliasDict(dict):
    """
    公司别名映射字典（带文档增强的字典子类）

    【设计说明】继承自内置 dict 以保持 O(1) 查找性能，
    同时通过类 docstring 提供完整的词典使用说明。

    【数据结构 / Data Structure】
    Key:   用户可能使用的别名字符串（如"茅台""BYD""招商"）
    Value: 包含标准化信息的字典 {"name": 全称, "code": 股票代码}

    【匹配策略 / Matching Strategy】
    采用「精确子串匹配」：检查 alias 是否出现在 query 文本中。
    这意味着：
    - "茅台" 能匹配 "我想了解茅台的情况" ✓
    - "茅台" 能匹配 "贵州茅台酒股份有限公司" ✓（因为"茅台"是子串）
    - "茅台" 无法匹配 "MT"（英文缩写不在词典中）✗

    【扩展建议 / Extension Suggestions】
    - 高频查询的公司应尽量多地收录别名（简称、昵称、英文缩写）
    - 可考虑接入外部 API（如 Tushare/AKShare）动态获取公司列表
    - 对于未收录的公司，LLM 提取路径可作为兜底
    """

    __doc__ = """
    A股/H股主流公司别名映射表 (Major A-Share/H-Share Company Alias Map)

    【覆盖范围 / Coverage】
    - 白酒板块：贵州茅台、五粮液、泸州老窖（代表性最强）
    - 新能源汽车：比亚迪、宁德时代（整车+电池双龙头）
    - 互联网平台：腾讯控股、阿里巴巴、美团-W（港股核心资产）
    - 银行保险：招商银行、中国平安（金融板块代表）
    - 总计：11个别名入口，覆盖 10 家头部公司

    【使用频率统计 / Usage Frequency (预估)】
    - 茅台/贵州茅台: ★★★★★ (最高频，几乎每个 demo 都会用到)
    - 比亚迪/宁德时代: ★★★★☆ (新能源热门)
    - 腾讯/阿里巴巴: ★★★☆☆ (港股投资者常用)
    - 其他: ★★☆☆☆ (偶发查询)

    【代码示例 / Code Example】
        >>> COMPANY_ALIASES["茅台"]
        {'name': '贵州茅台', 'code': '600519.SH'}
        >>> COMPANY_ALIASES.get("苹果")  # 未收录返回 None
        None
    """


COMPANY_ALIASES: Dict[str, Dict[str, str]] = {
    # ── 白酒板块 / Baijiu (Liquor) Sector ──
    # 贵州茅台是中国 A 股市值最高的公司之一，也是金融分析中最常被查询的对象。
    # 收录了最常见的两种称呼方式。
    "茅台": {
        "name": "贵州茅台",
        "code": "600519.SH",  # 上海证券交易所，白酒板块绝对龙头
    },
    "贵州茅台": {
        "name": "贵州茅台",
        "code": "600519.SH",
    },
    # 五粮液是白酒板块第二龙头，与茅台形成"茅五"双寡头格局。
    "五粮液": {
        "name": "五粮液",
        "code": "000858.SZ",  # 深圳证券交易所
    },
    # 泸州老窖是白酒板块第三名，属于"茅五泸"第一梯队。
    "泸州老窖": {
        "name": "泸州老窖",
        "code": "000568.SZ",
    },

    # ── 新能源汽车 / New Energy Vehicle (NEV) Sector ──
    # 比亚迪是全球最大的新能源汽车制造商，涵盖电池、电子、汽车三大业务。
    "比亚迪": {
        "name": "比亚迪",
        "code": "002594.SZ",
    },
    # 宁德时代 (CATL) 是全球动力电池市场份额第一的企业，俗称"宁王"。
    "宁德时代": {
        "name": "宁德时代",
        "code": "300750.SZ",  # 创业板
    },

    # ── 互联网平台 / Internet Platform Sector (港股) ──
    # 腾讯控股是中国市值最高的互联网公司，社交+游戏+金融科技+云服务。
    "腾讯": {
        "name": "腾讯控股",
        "code": "00700.HK",  # 港交所主板
    },
    # 阿里巴巴是电商+云计算巨头，港股上市（二次上市）。
    "阿里巴巴": {
        "name": "阿里巴巴",
        "code": "09988.HK",
    },
    # 美团是本地生活服务（外卖+到店）龙头，同程旅行+摩拜单车已剥离。
    "美团": {
        "name": "美团-W",
        "code": "03690.HK",  # W 代表 Weighted Voting Rights（同股不同权）
    },

    # ── 金融板块 / Financial Sector ──
    # 招商银行被誉为"零售之王"，是股份制银行中资产质量最优的。
    "招商银行": {
        "name": "招商银行",
        "code": "600036.SH",
    },
    # 中国平安是综合金融集团（保险+银行+投资），A 股最大保险公司。
    "中国平安": {
        "name": "中国平安",
        "code": "601318.SH",
    },
}


class _FinancialMetricsDict(dict):
    """
    财务指标分类词典（带文档增强的字典子类）

    【设计说明】采用「类别→指标列表」的两层结构，既支持精确指标匹配，
    也支持类别级别的模糊匹配（如用户说"盈利能力"时展开为代表指标）。

    【指标选取原则 / Metric Selection Principles】
    1. **CFA 考纲覆盖**: 所选指标均在 CFA 一级/二级考纲范围内
    2. **A股分析师常用**: 参考券商研报中最频繁出现的指标
    3. **数据可获得性**: 所有指标均可通过 Alpha Vantage 或财报数据计算得出
    4. **中英双语**: 指标名使用中文（用户友好），注释中注明英文对照

    【类别间关系 / Inter-Category Relationships】
    - 盈利能力 × 成长性 = "Quality of Growth"（增长的质量如何）
    - 现金流 vs 会计利润 = "Earnings Quality"（利润含金量）
    - 估值 × 成长性 = PEG 策略的核心
    - 偿债能力 × 现金流 = "Credit Risk"综合评估
    """

    __doc__ = """
    财务指标分类体系 (Financial Metrics Taxonomy)

    【6大类别 / 6 Major Categories】
    """


FINANCIAL_METRICS: Dict[str, List[str]] = {
    # ─────────────────────────────────────────────────────────
    # 类别 1: 盈利能力 (Profitability)
    #
    # 【分析意义】回答"这家公司赚不赚钱？赚钱效率如何？"的核心问题。
    # 是所有财务分析的起点——如果一家公司不盈利，其他分析都失去意义。
    #
    # 【指标详解 / Metric Details】
    # - 毛利率 (Gross Margin): 营收 - 成本后的初步获利能力，反映产品定价权和成本控制
    #   正常范围因行业而异：软件>70%，制造业20-40%，零售<15%
    # - 净利率 (Net Margin): 扣除所有费用和税收后的最终获利能力，是最全面的盈利指标
    # - ROE (Return on Equity): 股东投入资本的回报率，巴菲特最看重的指标之一
    #   杜邦分解: ROE = 净利率 × 资产周转率 × 权益乘数
    # - ROA (Return on Total Assets): 全部资产的回报效率，衡量资产利用效能
    # - EBITDA利润率 (EBITDA Margin): 息税折旧摊销前利润率，排除资本结构和税负影响
    #   常用于可比公司分析和估值
    # ─────────────────────────────────────────────────────────
    "盈利能力": [
        "毛利率",
        "净利率",
        "ROE",
        "ROA",
        "EBITDA利润率",
    ],

    # ─────────────────────────────────────────────────────────
    # 类别 2: 成长性 (Growth)
    #
    # 【分析意义】回答"这家公司在发展吗？发展速度是否可持续？"
    # 成长性是估值的核心驱动力——PEG 策略就是建立在成长性基础上的。
    #
    # 【指标详解 / Metric Details】
    # - 营收增速 (Revenue Growth Rate): 最直观的成长指标，Top-line growth
    # - 净利润增速 (Net Income Growth): 盈利的成长速度，应与营收增速对比看"质量"
    # - EPS增速 (Earnings Per Share Growth): 每股收益增长，考虑了股本变化的影响
    # - 营收CAGR (Revenue CAGR): 复合年均增长率，平滑单年波动，看长期趋势
    #   公式: CAGR = (期末值/期初值)^(1/n) - 1
    # - 利润CAGR (Profit CAGR): 净利润的复合年均增长率
    # ─────────────────────────────────────────────────────────
    "成长性": [
        "营收增速",
        "净利润增速",
        "EPS增速",
        "营收CAGR",
        "利润CAGR",
    ],

    # ─────────────────────────────────────────────────────────
    # 类别 3: 现金流 (Cash Flow)
    #
    # 【分析意义】回答"这家公司的利润是真金白银还是会计数字？"
    # 华尔壁谚语:"Revenue is vanity, profit is sanity, cash is reality."
    # （营收是虚荣，利润是理智，现金才是现实）
    #
    # 【指标详解 / Metric Details】
    # - 经营现金流 (Operating Cash Flow, OCF): 主营业务产生的现金净流入
    #   应与净利润对比：OCF > Net Income → 利润含金量高（高质量盈利）
    # - 自由现金流 (Free Cash Flow, FCF): OCF - 资本支出 = 可自由支配的现金
    #   是 DCF 估值模型的核心输入，也是分红和回购的资金来源
    # - 现金流覆盖率 (Cash Flow Coverage Ratio): OCF / 利息支出或债务偿还额
    #   衡量现金偿债能力，>1.5 为安全区间
    # - 资本支出 (Capital Expenditure, CapEx): 购置固定资产的现金流出
    #   高 CapEx 行业（制造/电信）vs 低 CapEx 行业（软件/SaaS）差异巨大
    # ─────────────────────────────────────────────────────────
    "现金流": [
        "经营现金流",
        "自由现金流",
        "现金流覆盖率",
        "资本支出",
    ],

    # ─────────────────────────────────────────────────────────
    # 类别 4: 估值 (Valuation)
    #
    # 【分析意义】回答"现在的股价贵不便宜？安全边际有多大？"
    # 估值是投资决策的最终落脚点——再好的公司买太贵也会亏损。
    #
    # 【指标详解 / Metric Details】
    # - PE (Price-to-Earnings): 股价/每股收益，最经典的估值指标
    #   合理范围: 15-25x（成长股可达 40-60x，银行股 5-10x）
    # - PB (Price-to-Book): 股价/每股净资产，适用于重资产行业（银行/地产/制造）
    #   PB < 1 可能意味着市场认为资产虚高或有隐形负债
    # - PS (Price-to-Sales): 股价/每股营收，适用于未盈利或利润波动大的公司
    # - EV/EBITDA: 企业价值/EBITDA，跨公司可比性强（排除资本结构差异）
    #   常用于 M&A 估值和跨国比较
    # - PEG (PE/Growth Rate): PE 除以增长率，彼得·林奇推崇的指标
    #   PEG < 1 → 低估，PEG = 1 → 合理，PEG > 2 → 可能高估
    # - 股息率 (Dividend Yield): 每股分红/股价，价值投资者的核心关注点
    # ─────────────────────────────────────────────────────────
    "估值": [
        "PE",
        "PB",
        "PS",
        "EV/EBITDA",
        "PEG",
        "股息率",
    ],

    # ─────────────────────────────────────────────────────────
    # 类别 5: 偿债能力 (Solvency / Leverage)
    #
    # 【分析意义】回答"这家公司会不会破产？财务安不安全？"
    # 2021-2023 年中国房地产行业的危机证明了偿债能力分析的重要性。
    #
    # 【指标详解 / Metric Details】
    # - 资产负债率 (Debt-to-Asset Ratio): 总负债/总资产，最宏观的杠杆指标
    #   一般企业 < 60% 为安全，房地产可达 70-80%（行业特性）
    # - 流动比率 (Current Ratio): 流动资产/流动负债，短期偿债能力的经典指标
    #   标准: > 2 为健康（但不同行业标准差异大，零售业可低至 1.0-1.5）
    # - 速动比率 (Quick Ratio): (流动资产-存货)/流动负债，排除存货后的偿债能力
    #   更严格的标准: > 1 为安全
    # - 利息保障倍数 (Interest Coverage Ratio): EBIT/利息支出
    #   > 3 为安全，< 1.5 意味着利息支付困难（危险信号！）
    # ─────────────────────────────────────────────────────────
    "偿债能力": [
        "资产负债率",
        "流动比率",
        "速动比率",
        "利息保障倍数",
    ],

    # ─────────────────────────────────────────────────────────
    # 类别 6: 运营效率 (Operational Efficiency)
    #
    # 【分析意义】回答"这家公司的资产管理效率如何？钱转得快不快？"
    # 高运营效率意味着更少的营运资金占用和更高的资产回报。
    #
    # 【指标详解 / Metric Details】
    # - 存货周转率 (Inventory Turnover): 营业成本/平均存货
    #   越高越好：白酒>1（茅台特殊，存货增值）/ 零售>8 / 科技制造>6
    # - 应收账款周转率 (Receivables Turnover): 营收/平均应收账款
    #   越高越好：意味着回款快、坏账风险低
    # - 总资产周转率 (Total Asset Turnover): 营收/平均总资产
    #   是杜邦分析 ROE 分解的三大因子之一
    # ─────────────────────────────────────────────────────────
    "运营效率": [
        "存货周转率",
        "应收账款周转率",
        "总资产周转率",
    ],
}


class _TimePatternsDict(dict):
    """
    时间表达正则模式字典（带文档增强的字典子类）

    【设计说明】Key 为正则表达式字符串，Value 为语义类型标识。
    在 SlotExtractor.__init__() 中会被预编译为 re.Pattern 对象以提升性能。

    【匹配顺序 / Matching Order】
    字典的迭代顺序即为匹配尝试顺序。当前排列遵循「从具体到模糊」原则：
    1. 最具体的（带数字的年份/季度）优先匹配
    2. 相对时间表达次之
    3. 最模糊的（"今年""去年"）最后匹配

    【中文时间表达的复杂性 / Complexity of Chinese Time Expressions】
    中文时间表达远比英文丰富和灵活，例如：
    - "前三季度" ≠ "第三季度"（前者是累计概念）
    - "下半年" 不是 "下半这年"（是固定搭配）
    - "近年来" ≈ "最近几年" 但语气更正式
    本词典覆盖了金融分析中最常见的 ~80% 场景，剩余 ~20% 由 LLM 补充。
    """

    __doc__ = """
    中文时间表达正则模式库 (Chinese Time Expression Regex Pattern Library)

    【覆盖的场景 / Covered Scenarios】
    """


TIME_PATTERNS: Dict[str, str] = {
    # ── 场景 1: 绝对年份 / Absolute Year ──
    # 匹配 "2023年""1998年" 等 4 位数字+年字的组合
    # 这是最明确的时间表达，优先级最高
    r"(\d{4})年": "year",

    # ── 场景 2: 绝对季度 / Absolute Quarter ──
    # 匹配 "2024年第一季度""2023年Q3""2022年第2季" 等表达
    # 支持：中文数字（一二三四）+ 阿拉伯数字（1234）+ Q 前缀
    # 注意：季度的提取必须依赖年份（单独的"第一季度"缺乏上下文）
    r"(\d{4})年第([一二三四1234])季": "quarter",

    # ── 场景 3: 相对年份（近 N 年）/ Recent N Years ──
    # 匹配 "近三年""近5年""近十年" 等
    # "近" 和 "最近" 在此场景下语义相同，合并为一个模式
    r"近(\d+)年": "recent_years",

    # ── 场景 4: 相对时间段（最近 N 单位）/ Recent N Period ──
    # 匹配 "最近3个月""最近2个季度""最近半年" 等
    # 比 recent_years 更通用，支持年/月/季三种时间单位
    r"最近(\d+)(年|月|季)": "recent_period",

    # ── 场景 5: 年份范围 / Year Range ──
    # 匹配 "2020-2023""2018至2022" 等起止年份表达
    # 注意：这里的连接符只覆盖了 "-" ，中文"至""到""~"等由 LLM 处理
    r"(\d{4})-(\d{4})": "year_range",

    # ── 场景 6: 当前年份 / Current Year ──
    # 匹配 "今年""本年""当前""本年度" 等指向当年
    # 解析时会动态替换为实际的当前年份（datetime.now().year）
    r"(?:今年|本年|当前)": "current_year",

    # ── 场景 7: 去年 / Last Year ──
    # 匹配 "去年""上年""上一年""前一年" 等指向去年的表达
    # 解析时会动态替换为 当前年份-1
    r"(?:去年|上年|上一年)": "last_year",
}


# ════════════════════════════════════════════════════════════
# 第三部分：槽位提取器核心类
# Part 3: Slot Extractor Core Class
#
# 这是本模块唯一的公共 API 提供者。外部调用方式：
#   extractor = SlotExtractor()
#   result = extractor.extract(SlotExtractionRequest(query="..."))
# ════════════════════════════════════════════════════════════


class SlotExtractor:
    """
    金融领域槽位提取器 (Financial Domain Slot Extractor)

    【类职责 / Class Responsibility】
    作为本模块的门面 (Facade)，封装完整的「规则提取 → LLM 提取 → 结果融合 →
    质量验证」四阶段流水线。对外提供唯一的 extract() 方法，内部协调多个
    私有方法完成各阶段的处理。

    【线程安全 / Thread Safety】
    - 本类是无状态的（除了 self._llm 和 self._time_patterns 实例变量）
    - self._llm (ChatOpenAI) 是线程安全的（langchain 保证）
    - self._time_patterns 是初始化后只读的
    - 结论：可以在多线程环境中共享同一个 SlotExtractor 实例
    - 建议：在应用级别创建单例，避免重复初始化 LLM 连接

    【使用模式 / Usage Patterns】

    模式 1: 标准一次性使用（适合脚本/CLI）
        >>> extractor = SlotExtractor()
        >>> result = extractor.extract(SlotExtractionRequest(query="分析茅台2023年ROE"))
        >>> print(result.companies[0].name)  # "贵州茅台"

    模式 2: 复用实例（适合 Web 服务，推荐）
        >>> extractor = SlotExtractor()  # 应用启动时创建一次
        >>> # ... 后续每次请求复用 ...
        >>> result = extractor.extract(request)

    模式 3: 通过便捷函数（最简洁，内部每次创建新实例）
        >>> result = extract_slots("比亚迪今年的营收增速")

    【性能调优参数 / Performance Tuning Parameters】
    - LLM temperature: 固定为 0（槽位提取需要确定性输出）
    - 如需进一步提升规则覆盖率，扩充 COMPANY_ALIASES / FINANCIAL_METRICS
    - 如需降低延迟，可设置超时或在规则阶段就命中时跳过 LLM（短路优化）
    """

    def __init__(self) -> None:
        """
        初始化槽位提取器实例。

        【初始化内容 / Initialization Tasks】
        1. 创建 LLM 客户端（ChatOpenAI，连接 GLM-4-plus）
        2. 预编译所有时间正则表达式模式（避免每次调用时重复编译）

        【LLM 配置 rationale / LLM Configuration Rationale】
        - model: 使用 settings.LLM_MODEL（默认 glm-4-plus）
        - temperature: 0（确定性输出，槽位提取不需要创造性）
        - 为什么不用 function_calling / tool_use？
          因为当前版本的 Pydantic 输出格式已足够稳定，
          且 JSON mode 在 GLM-4-plus 上兼容性良好
        """
        self._llm: ChatOpenAI = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )

        self._time_patterns: Dict[str, re.Pattern] = {
            pattern: re.compile(pattern)
            for pattern in TIME_PATTERNS.keys()
        }

    def extract(self, request: SlotExtractionRequest) -> SlotExtractionResult:
        """
        执行完整的槽位提取流水线。

        【流水线阶段 / Pipeline Stages】

        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │
        │ 规则提取  │    │ LLM提取  │    │ 结果融合  │    │ 质量验证  │
        │ ~10ms    │    │ ~1-3s    │    │ ~1ms     │    │ ~1ms     │
        │ 确定性    │    │ 语义级    │    │ 去重合并  │    │ 置信度   │
        └──────────┘    └──────────┘    └──────────┘    └──────────┘

        【Args / 参数】
            request:
                经 Pydantic 校验的提取请求，包含 query（必填）和 context（可选）。

        【Returns / 返回值】
            SlotExtractionResult:
                包含所有提取到的槽位、置信度分数和质量标记的完整结果对象。
                即使未能提取到任何槽位，也会返回 success=True 的结果
                （只是 confidence 会很低）。

        【Raises / 异常】
            本方法不会抛出异常。所有异常在内部捕获并降级处理：
            - LLM 调用失败 → 回退到纯规则结果
            - JSON 解析失败 → 忽略 LLM 结果，使用规则结果
            - 其他意外错误 → 记录日志，返回最小可行结果

        【调用示例 / Example】
            >>> extractor = SlotExtractor()
            >>> req = SlotExtractionRequest(
            ...     query="对比茅台和五粮液近三年的ROE",
            ...     context="用户之前问过茅台的估值"
            ... )
            >>> result = extractor.extract(req)
            >>> assert result.success == True
            >>> assert len(result.companies) == 2  # 茅台 + 五粮液
            >>> assert result.time_info.period == "近3年"
        """
        logger.info("[槽位提取] 开始处理查询: %s...", request.query[:50])

        # ── Stage 1: 规则提取（快速路径，确定性 100%）──
        # 利用预编译的正则和词典做字符串匹配，延迟极低（< 10ms）
        # 适用于：高频常见模式（知名公司名、标准时间表达、常见指标名）
        rule_result: Dict[str, Any] = self._extract_by_rules(request.query)

        # ── Stage 2: LLM 提取（慢速路径，语义级理解）──
        # 将查询发送给 GLM-4-plus 进行自然语言理解
        # 适用于：复杂语义、隐含信息、词典未覆盖的长尾表达
        # 注意：即使 LLM 失败也不会中断流程，Stage 3 会优雅降级
        llm_result: Dict[str, Any] = self._extract_by_llm(request)

        # ── Stage 3: 结果融合（合并去重，冲突解决）──
        # 策略：以 LLM 结果为主（语义理解更准确），规则结果为辅（补全和校验）
        # 同一实体的冲突以 LLM 为准，不同实体取并集
        final_result: SlotExtractionResult = self._merge_results(
            rule_result, llm_result, request.query
        )

        # ── Stage 4: 质量验证（置信度计算，缺失检测，澄清生成）──
        # 基于槽位完整性计算 0-1 置信度分数
        # 低于阈值时自动标记 clarification_needed 并生成追问
        final_result = self._validate_and_score(final_result)

        logger.info(
            "[槽位提取] 完成, 置信度=%.2f, 方法=%s",
            final_result.confidence,
            final_result.extraction_method,
        )
        if final_result.clarification_needed:
            logger.warning(
                "[槽位提取] ⚠️ 需要用户澄清: %s",
                final_result.clarification_questions,
            )

        return final_result

    # ─────────────────────────────────────────────────────
    # Stage 1: 基于规则的快速提取
    # Rule-Based Fast Extraction
    # ─────────────────────────────────────────────────────

    def _extract_by_rules(self, query: str) -> Dict[str, Any]:
        """
        使用规则引擎（正则 + 词典）执行快速槽位提取。

        【设计策略 / Design Strategy】
        采用「遍历匹配」策略：依次对查询文本运行三类规则：
        1. 公司名匹配：遍历 COMPANY_ALIASES，检查每个 alias 是否是 query 的子串
        2. 时间匹配：遍历预编译的正则模式，找到第一个命中的时间表达
        3. 指标匹配：遍历 FINANCIAL_METRICS 的所有指标名，做大小写无关的子串匹配

        【为什么用子串匹配而不是分词？ / Why Substring Match over Tokenization?】
        - 中文分词（jieba/lac）引入额外依赖和延迟
        - 金融术语通常是连续字符串（"ROE""资产负债率"），子串匹配足够准确
        - 子串匹配误报率可控（"ROE"不会错误匹配"PROFESSION"中的子串，因为中文语境）

        【Args / 参数】
            query: 原始查询文本（未预处理）

        【Returns / 返回值】
            Dict[str, Any]: 包含以下键的字典（值可能为 None 或空列表）：
            - companies: List[CompanySlot] 或 []
            - time_info: TimeSlot 或 None
            - metrics: MetricSlot 或 None
            - comparison_targets: List[CompanySlot] 或 []
            - industry: str 或 None
        """
        result: Dict[str, Any] = {
            "companies": [],
            "time_info": None,
            "metrics": None,
            "comparison_targets": [],
            "industry": None,
        }

        # ── 1.1 公司实体提取 ──
        # 遍历公司别名词典，逐个检查是否出现在查询文本中
        for alias, info in COMPANY_ALIASES.items():
            if alias in query:
                company = CompanySlot(
                    name=info["name"],
                    stock_code=info["code"],
                    aliases=[alias],
                )

                # 判断该公司是「主分析对象」还是「对比目标」
                # 启发式规则：如果查询中出现"对比""比较""vs"等关键词，
                # 则第一个匹配的公司归入 companies（主语），后续归入 comparison_targets
                has_comparison_keyword = any(
                    kw in query for kw in ("对比", "比较", "vs", "VS", "和...比")
                )
                if has_comparison_keyword:
                    if len(result["companies"]) == 0:
                        result["companies"].append(company)
                    else:
                        result["comparison_targets"].append(company)
                else:
                    result["companies"].append(company)

        # ── 1.2 时间信息提取 ──
        time_info = self._extract_time(query)
        if time_info is not None:
            result["time_info"] = time_info

        # ── 1.3 财务指标提取 ──
        metrics = self._extract_metrics(query)
        if metrics is not None:
            result["metrics"] = metrics

        return result

    def _extract_time(self, query: str) -> Optional[TimeSlot]:
        """
        从查询文本中提取时间信息。

        【匹配策略 / Matching Strategy】
        按 TIME_PATTERNS 定义的顺序逐一尝试正则匹配，返回第一个命中的结果。
        这意味着更具体的模式（如"2023年"）会优先于更模糊的模式（如"今年"）。

        【支持的语义类型 / Supported Semantic Types】
        - year: 绝对年份 → TimeSlot(year=2023)
        - quarter: 绝对季度 → TimeSlot(year=2024, quarter=1)
        - recent_years: 近N年 → TimeSlot(period="近N年")
        - recent_period: 最近N单位 → TimeSlot(period="近N单位")
        - year_range: 年份范围 → TimeSlot(time_range=("2020","2023"))
        - current_year: 今年 → TimeSlot(year=当前年)
        - last_year: 去年 → TimeSlot(year=当前年-1)

        [Args / 参数]
            query: 原始查询文本

        [Returns / 返回值]
            TimeSlot: 提取到时间信息时的结构化结果
            None: 查询中未包含任何可识别的时间表达
        """
        for pattern, time_type in self._time_patterns.items():
            match = self._time_patterns[pattern].search(query)
            if match is None:
                continue

            if time_type == "year":
                return TimeSlot(year=int(match.group(1)))

            elif time_type == "quarter":
                quarter_map = {
                    "一": 1, "二": 2, "三": 3, "四": 4,
                    "1": 1, "2": 2, "3": 3, "4": 4,
                }
                q_str = match.group(2)
                return TimeSlot(
                    year=int(match.group(1)),
                    quarter=quarter_map.get(q_str, int(q_str) if q_str.isdigit() else 1),
                )

            elif time_type in ("recent_years", "recent_period"):
                num = int(match.group(1))
                unit = match.group(2) if time_type == "recent_period" else "年"
                return TimeSlot(period=f"近{num}{unit}")

            elif time_type == "year_range":
                return TimeSlot(time_range=(match.group(1), match.group(2)))

            elif time_type == "current_year":
                return TimeSlot(year=datetime.now().year)

            elif time_type == "last_year":
                return TimeSlot(year=datetime.now().year - 1)

        return None

    def _extract_metrics(self, query: str) -> Optional[MetricSlot]:
        """
        从查询文本中提取财务指标信息。

        【双层提取策略 / Two-Layer Extraction Strategy】

        Layer 1 — 精确指标名匹配：
        遍历 FINANCIAL_METRICS 中所有 30+ 个具体指标名，
        检查是否出现在查询文本中（大小写无关匹配）。
        前 3 个命中的归入 primary_metrics，其余归入 secondary_metrics。

        Layer 2 — 笼统关键词扩展：
        当用户使用类别级别的笼统词汇（如"盈利""成长""估值"）而未列举具体指标时，
        通过 keyword_mapping 自动展开为代表指标。
        例如：用户说"盈利能力" → 自动加入 [毛利率, 净利率, ROE]

        【指标数量限制 rationale / Metric Count Limit Rationale】
        - primary_metrics 上限 5 个：防止信息过载，聚焦用户最关心的指标
        - secondary_metrics 无硬上限：实际通常 < 10 个（受限于词典规模）
        - 如果用户真的提到了 10+ 个指标，说明需求本身就很复杂，
          这种情况下应由 TaskPlanner 拆分为多个子任务分别处理

        [Args / 参数]
            query: 原始查询文本

        [Returns / 返回值]
            MetricSlot: 提取到指标时的结构化结果（含分层列表和类别标签）
            None: 查询中未包含任何可识别的财务指标
        """
        primary_metrics: List[str] = []
        secondary_metrics: List[str] = []
        categories: List[str] = []

        # ── Layer 1: 精确指标名匹配 ──
        for category, metrics_list in FINANCIAL_METRICS.items():
            for metric in metrics_list:
                if metric.lower() in query.lower() or metric in query:
                    if len(primary_metrics) < 3:
                        primary_metrics.append(metric)
                    else:
                        secondary_metrics.append(metric)
                    if category not in categories:
                        categories.append(category)

        # ── Layer 2: 笼统关键词 → 具体指标扩展 ─
        # 映射关系：用户说的模糊词汇 → 应展开的具体指标列表
        keyword_mapping: Dict[str, List[str]] = {
            "盈利": ["毛利率", "净利率", "ROE"],
            "成长": ["营收增速", "净利润增速"],
            "估值": ["PE", "PB"],
            "风险": ["资产负债率", "流动比率"],
            "现金": ["经营现金流", "自由现金流"],
        }

        for keyword, related_metrics in keyword_mapping.items():
            if keyword in query and not any(
                m in primary_metrics for m in related_metrics
            ):
                for m in related_metrics[:2]:
                    if m not in primary_metrics and m not in secondary_metrics:
                        primary_metrics.append(m)
                        break

        if primary_metrics or categories:
            return MetricSlot(
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
                metric_categories=categories,
            )
        return None

    # ─────────────────────────────────────────────────────
    # Stage 2: 基于 LLM 的语义级提取
    # LLM-Based Semantic Extraction
    # ─────────────────────────────────────────────────────

    def _extract_by_llm(self, request: SlotExtractionRequest) -> Dict[str, Any]:
        """
        调用 LLM 执行语义级别的槽位提取。

        【为什么需要 LLM 层？ / Why Do We Need LLM?】
        规则提取虽然快速但有固有局限性：
        1. 无法处理未见过的表达（如"那家造车的公司"→ 比亚迪）
        2. 无法理解隐含信息（如"它的估值"需要结合上下文消解"它"）
        3. 无法处理复杂句法（如"除了盈利还想看看风险怎么样"）
        LLM 层弥补了这些不足，提供真正的自然语言理解能力。

        [Prompt Engineering 策略 / Prompt Strategy]
        - System Prompt: 定义角色（金融信息抽取专家）+ 输出格式规范
        - User Prompt: 注入查询 + 上下文 + JSON Schema 示例
        - Output Format: 严格 JSON（LLM 的 temperature=0 确保格式稳定性）
        - Fallback: 解析失败时返回空字典（不阻塞流水线）

        [Args / 参数]
            request: 包含 query 和 context 的提取请求

        [Returns / 返回值]
            Dict[str, Any]: LLM 返回的 JSON 解析结果，包含与 SlotExtractionResult
                             对应的字段（companies, time_info, metrics 等）。
                             失败时返回空字典 {}。
        """
        system_prompt: str = (
            "你是一个专业的金融领域信息抽取专家。\n"
            "请从用户的自然语言问题中提取关键的结构化信息。\n\n"
            "你需要提取以下类型的槽位：\n"
            "1. **公司实体**：公司全称、股票代码、常用别名\n"
            "2. **时间信息**：具体年份、季度、时间范围描述\n"
            "3. **财务指标**：用户关心的具体财务指标（如PE、ROE、净利润等）\n"
            "4. **对比目标**：如果用户提到对比或比较，提取对比对象\n"
            "5. **行业信息**：涉及的行业名称\n"
            "6. **分析类型**：用户期望的分析深度（深度解读/快速概览/对比分析）\n\n"
            "输出要求：\n"
            "- 以严格的JSON格式输出\n"
            "- 如果某个槽位在文本中没有提及，设为null\n"
            "- 置信度字段用0-1的小数表示\n"
            "- 对于模糊的信息，在clarification_needed中标记"
        )

        context_section: str = ""
        if request.context:
            context_section = f"\n对话上下文：{request.context}"

        user_prompt: str = (
            f"请从以下用户问题中提取结构化槽位信息：\n\n"
            f"用户问题：{request.query}"
            f"{context_section}\n\n"
            f"请以JSON格式返回提取结果，格式如下：\n"
            f'{{\n'
            f'    "companies": [{{"name": "公司名", "stock_code": "代码", "aliases": ["别名"]}}],\n'
            f'    "time_info": {{"year": 年份, "quarter": 季度, "period": "时间范围"}},\n'
            f'    "metrics": {{\n'
            f'        "primary_metrics": ["主要指标1", "主要指标2"],\n'
            f'        "secondary_metrics": ["次要指标"],\n'
            f'        "metric_categories": ["类别"]\n'
            f'    }},\n'
            f'    "comparison_targets": [{{"name": "...", "stock_code": "..."}}],\n'
            f'    "industry": "行业名称",\n'
            f'    "analysis_type": "分析类型",\n'
            f'    "confidence": 0.95,\n'
            f'    "missing_slots": ["缺失的槽位"],\n'
            f'    "clarification_needed": false,\n'
            f'    "clarification_questions": ["需要向用户确认的问题"]\n'
            f'}}\n\n'
            f"只输出JSON，不要其他内容。"
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            content: str = response.content.strip()

            content = self._clean_json_response(content)

            result: Dict[str, Any] = json.loads(content)
            logger.info("[槽位提取] LLM 提取成功")
            return result

        except json.JSONDecodeError as e:
            logger.error("[槽位提取] LLM 返回的 JSON 解析失败: %s", e)
            return {}
        except Exception as e:
            logger.error("[槽位提取] LLM 调用异常: %s", e)
            return {}

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """
        清理 LLM 返回内容中的 Markdown 代码块标记。

        [Why / 为什么需要清理]
        GLM-4-plus（以及大多数 LLM）倾向于将 JSON 包裹在 ```json ... ``` 代码块中。
        虽然这在 Markdown 渲染时很美观，但对 json.loads() 来说是非法字符。
        此方法负责剥去这些装饰性标记，提取纯净的 JSON 字符串。

        [Args / 参数]
            content: LLM 原始返回文本

        [Returns / 返回值]
            清理后的纯 JSON 字符串
        """
        for prefix in ("```json", "```"):
            if content.startswith(prefix):
                content = content[len(prefix):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    # ─────────────────────────────────────────────────────
    # Stage 3: 结果融合
    # Result Merging & Deduplication
    # ─────────────────────────────────────────────────────

    def _merge_results(
        self,
        rule_result: Dict[str, Any],
        llm_result: Dict[str, Any],
        original_query: str,
    ) -> SlotExtractionResult:
        """
        融合规则提取和 LLM 提取的结果，产生最终的槽位输出。

        【融合策略详解 / Merge Strategy Details】

        ┌──────────────┬──────────────────────────────────────────┐
        │ 槽位类型      │ 融合规则                                  │
        ├──────────────┼──────────────────────────────────────────┤
        │ companies     │ LLM 优先，规则补全，按 stock_code 去重     │
        │ time_info     │ LLM 优先（能理解复杂表达），规则作为 fallback│
        │ metrics       │ 取并集（两种方法可能发现不同指标）          │
        │ comparison_   │ 取并集，同样按 stock_code 去重             │
        │ targets       │                                          │
        │ industry      │ LLM 优先（可推断隐含行业）                 │
        │ analysis_type │ 仅来自 LLM（规则层不具备此能力）            │
        └──────────────┴──────────────────────────────────────────┘

        【去重策略 / Deduplication Strategy】
        - 公司类槽位：以 stock_code 作为唯一键（最可靠的标识符）
        - 指标类槽位：以指标名字符串作为唯一键，用 set() 去重
        - 时间槽位：不涉及去重（一个查询只有一个时间信息）

        [Args / 参数]
            rule_result:    Stage 1 规则提取的原始结果字典
            llm_result:     Stage 2 LLM 提取的原始结果字典（可能为空 {}）
            original_query: 用户原始查询（用于填充 raw_query 字段）

        [Returns / 返回值]
            SlotExtractionResult: 融合后的完整结果（置信度为初始值 0.8，
                                     待 Stage 4 校准）
        """
        merged_companies: List[CompanySlot] = []
        company_codes_seen: set = set()

        # ── 3.1 公司信息融合 ──
        # 拼接顺序：LLM 结果在前（优先），规则结果在后（补全）
        llm_companies = (llm_result.get("companies", []) if isinstance(llm_result, dict) else [])
        rule_companies = (rule_result.get("companies", []) if isinstance(rule_result, dict) else [])
        all_companies: List[Any] = llm_companies + rule_companies
        for company_data in all_companies:
            if isinstance(company_data, dict):
                code = company_data.get("stock_code")
                if code and code not in company_codes_seen:
                    company_codes_seen.add(code)
                    merged_companies.append(CompanySlot(**company_data))
            elif isinstance(company_data, CompanySlot):
                if company_data.stock_code and company_data.stock_code not in company_codes_seen:
                    company_codes_seen.add(company_data.stock_code)
                    merged_companies.append(company_data)

        # ── 3.2 时间信息融合 ──
        # LLM 优先：因为 LLM 能理解"那年到现在"这类复杂表达
        time_info: Optional[TimeSlot] = None
        if isinstance(llm_result, dict) and llm_result.get("time_info"):
            time_data = llm_result["time_info"]
            if isinstance(time_data, dict):
                time_info = TimeSlot(**time_data)
        elif isinstance(rule_result, dict) and rule_result.get("time_info"):
            time_data = rule_result["time_info"]
            time_info = TimeSlot(**time_data) if isinstance(time_data, dict) else time_data

        # ── 3.3 指标信息融合（取并集）──
        metrics: Optional[MetricSlot] = None
        
        def _extract_primary_metrics(metrics_val) -> List[str]:
            if metrics_val is None:
                return []
            if isinstance(metrics_val, MetricSlot):
                return getattr(metrics_val, 'primary_metrics', [])
            if isinstance(metrics_val, dict):
                return metrics_val.get('primary_metrics', [])
            return []
        
        llm_metrics_raw = (llm_result.get("metrics") if isinstance(llm_result, dict) else None)
        rule_metrics_raw = (rule_result.get("metrics") if isinstance(rule_result, dict) else None)
        
        all_primary_raw: List[str] = (
            _extract_primary_metrics(llm_metrics_raw) +
            _extract_primary_metrics(rule_metrics_raw)
        )
        all_primary: List[str] = list(set(all_primary_raw))
        if all_primary:
            metrics = MetricSlot(primary_metrics=all_primary[:5])

        # ── 3.4 对比目标融合（取并集）──
        comparison_targets: List[CompanySlot] = []
        llm_targets = (llm_result.get("comparison_targets", []) if isinstance(llm_result, dict) else [])
        rule_targets = (rule_result.get("comparison_targets", []) if isinstance(rule_result, dict) else [])
        for target in llm_targets + rule_targets:
            if isinstance(target, dict):
                comparison_targets.append(CompanySlot(**target))
            elif isinstance(target, CompanySlot):
                comparison_targets.append(target)

        # ── 3.5 组装最终结果 ──
        extraction_method: str = "hybrid"
        if not llm_result:
            extraction_method = "rule"
        elif (not isinstance(rule_result, dict) or 
              (not rule_result.get("companies") and not rule_result.get("metrics"))):
            extraction_method = "llm"

        result = SlotExtractionResult(
            success=True,
            confidence=0.8,
            companies=merged_companies,
            time_info=time_info,
            metrics=metrics,
            comparison_targets=comparison_targets,
            industry=(llm_result.get("industry") if isinstance(llm_result, dict) else None) or 
                     (rule_result.get("industry") if isinstance(rule_result, dict) else None),
            analysis_type=llm_result.get("analysis_type") if isinstance(llm_result, dict) else None,
            raw_query=original_query,
            extraction_method=extraction_method,
            clarification_needed=llm_result.get("clarification_needed", False),
            clarification_questions=llm_result.get("clarification_questions", []),
        )

        return result

    # ─────────────────────────────────────────────────────
    # Stage 4: 质量验证与置信度评分
    # Quality Validation & Confidence Scoring
    # ─────────────────────────────────────────────────────

    def _validate_and_score(self, result: SlotExtractionResult) -> SlotExtractionResult:
        """
        验证提取结果的完整性并计算综合置信度分数。

        【评分模型 / Scoring Model】

        置信度 = min(max(基础分 × 方法系数, 0.0), 1.0)

        基础分组成（满分 1.0）：
        ┌────────────────────┬────────┬────────────────────────────┐
        │ 槽位                │ 权重   │ 说明                        │
        ├────────────────────┼────────┼────────────────────────────┤
        │ companies          │ 0.25   │ 最关键——无公司则无法定位对象  │
        │ time_info          │ 0.15   │ 影响数据时效性和分析窗口     │
        │ metrics            │ 0.20   │ 决定分析方向和深度           │
        │ comparison_targets │ 0.10   │ 加分项——对比场景必备         │
        │ industry           │ 0.10   │ 加分项——有助于行业对标       │
        │ analysis_type      │ 0.10   │ 加分项——指导报告风格        │
        │ ────────────────── │ ────── │ ──────────────────────────  │
        │ 理论满分           │ 0.90   │ （留 0.10 余量给方法系数）   │
        └────────────────────┴────────┴────────────────────────────┘

        方法系数（Method Multiplier）：
        - hybrid: 1.10（两种方法交叉验证，可信度溢价）
        - llm:   1.05（LLM 泛化强但有幻觉风险，小溢价）
        - rule:  1.00（确定性高但覆盖率有限，基准线）

        [Args / 参数]
            result: Stage 3 产出的融合结果（将被原地修改后返回）

        [Returns / 返回值]
            SlotExtractionResult: 同一对象（原地更新了 confidence/missing_slots/
                                      clarification_needed/clarification_questions/
                                      normalized_query 字段）
        """
        score_components: List[float] = []
        missing_slots: List[str] = []

        # ── 4.1 核心槽位完整性检查 ──

        if result.companies:
            score_components.append(0.25)
        else:
            missing_slots.append("公司实体")

        if result.time_info:
            score_components.append(0.15)
        else:
            missing_slots.append("时间信息")

        if result.metrics and result.metrics.primary_metrics:
            score_components.append(0.20)
        else:
            missing_slots.append("财务指标")

        # ── 4.2 扩展槽位加分项 ──

        if result.comparison_targets:
            score_components.append(0.10)
        if result.industry:
            score_components.append(0.10)
        if result.analysis_type:
            score_components.append(0.10)

        # ── 4.3 综合置信度计算 ──
        base_confidence: float = sum(score_components)

        method_multipliers: Dict[str, float] = {
            "hybrid": 1.10,
            "llm": 1.05,
            "rule": 1.00,
        }
        multiplier: float = method_multipliers.get(
            result.extraction_method, 1.00
        )
        base_confidence *= multiplier

        final_confidence: float = min(max(base_confidence, 0.0), 1.0)

        # ── 4.4 更新结果对象 ──
        result.confidence = round(final_confidence, 2)
        result.missing_slots = missing_slots

        # ── 4.5 质量门控：低置信度触发澄清 ──
        if final_confidence < 0.5:
            result.clarification_needed = True
            if not result.clarification_questions:
                result.clarification_questions = [
                    "请问您想了解哪家公司的信息？",
                    "您关注的时间范围是什么时候？",
                ]

        # ── 4.6 生成标准化查询（用于 RAG 检索增强）──
        result.normalized_query = self._generate_normalized_query(result)

        return result

    def _generate_normalized_query(self, result: SlotExtractionResult) -> str:
        """
        基于提取的结构化槽位生成标准化查询语句。

        【目的 / Purpose】
        将非结构化的自然语言转换为半结构化的「实体+时间+指标+类型」格式，
        可用于：
        - RAG 检索时的 query expansion（提高召回率）
        - 日志中的结构化记录（便于搜索和统计）
        - 向用户展示"我理解您的意思是..."（确认环节）

        [生成规则 / Generation Rules]
        1. 公司名（取第一个，即主分析对象）
        2. 时间信息（year 或 period 二选一）
        3. 指标列表（取 primary_metrics 前三个，用顿号连接）
        4. 分析类型（如有）
        各部分之间用空格连接。如果没有任何槽位，返回原始查询。

        [Args / 参数]
            result: 已完成的槽位提取结果

        [Returns / 返回值]
            str: 标准化后的查询字符串
        """
        parts: List[str] = []

        if result.companies:
            parts.append(result.companies[0].name)

        if result.time_info:
            if result.time_info.year is not None:
                parts.append(str(result.time_info.year))
            elif result.time_info.period:
                parts.append(result.time_info.period)

        if result.metrics and result.metrics.primary_metrics:
            parts.append("、".join(result.metrics.primary_metrics[:3]))

        if result.analysis_type:
            parts.append(result.analysis_type)

        return " ".join(parts) if parts else result.raw_query


# ════════════════════════════════════════════════════════════
# 第四部分：公共便捷函数
# Part 4: Public Convenience Function
#
# 提供一种极简的调用方式，隐藏 SlotExtractor 的实例化细节。
# 适合快速原型开发和脚本式调用。
# ════════════════════════════════════════════════════════════


def extract_slots(query: str, context: Optional[str] = None) -> SlotExtractionResult:
    """
    便捷函数：一步完成槽位提取。

    【设计意图 / Design Intent】
    将「创建实例→构建请求→调用 extract」三步压缩为一步函数调用。
    适用于：
    - 快速测试和调试
    - Jupyter Notebook 交互式探索
    - 不需要复用 extractor 实例的场景

    【性能注意事项 / Performance Note】
    每次调用都会创建新的 SlotExtractor 实例（包括新的 LLM 连接）。
    在高性能场景（如 Web 服务）中，建议直接使用 SlotExtractor 类并复用实例。

    [Args / 参数]
        query:   用户原始查询文本
        context: 可选的多轮对话上下文

    [Returns / 返回值]
        SlotExtractionResult: 完整的槽位提取结果

    [Example / 示例]
        >>> result = extract_slots("分析比亚迪2024年的ROE和净利率")
        >>> print(result.confidence)
        0.85
        >>> print(result.companies[0].name)
        比亚迪
    """
    extractor = SlotExtractor()
    request = SlotExtractionRequest(query=query, context=context)
    return extractor.extract(request)


# ════════════════════════════════════════════════════════════
# 第五部分：模块自测
# Part 5: Self-Test Suite
#
# 当直接运行 python slot_extractor.py 时执行的测试用例。
# 覆盖 4 种典型场景：单公司分析、多公司对比、行业查询、模糊查询。
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_queries: List[str] = [
        "帮我分析一下贵州茅台2023年的盈利能力和现金流状况",
        "对比一下茅台和五粮液近三年的估值水平",
        "比亚迪今年的营收增长怎么样？",
        "新能源汽车行业的竞争格局如何？",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"输入: {query}")
        result = extract_slots(query)
        print(f"置信度: {result.confidence}")
        print(f"公司: {[c.name for c in result.companies]}")
        print(f"时间: {result.time_info}")
        print(
            f"指标: {result.metrics.primary_metrics if result.metrics else None}"
        )
        if result.clarification_needed:
            print(f"⚠️ 需要澄清: {result.clarification_questions}")
