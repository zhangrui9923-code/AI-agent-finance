"""
金融领域任务规划模块 (Financial Domain Task Planning)
======================================================

【模块定位 / Module Positioning】
本模块是 AI 金融研究助手系统的「任务编排」核心组件，位于 Intent Classification 之后、
Skill/Agent 执行之前。它负责将用户的抽象需求转化为可执行的结构化任务序列，
是连接"用户意图"和"具体执行"的关键桥梁。

【架构角色 / Architectural Role】

┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  意图分类结果  │───▶│  TaskPlanner     │───▶│  下游消费者        │
│              │    │ (本模块)         │                    │
│ financial_   │    │ 模板匹配→参数填充 │    │ Skill Framework  │
│ analysis     │    │ →DAG构建→拓扑排序 │    │ Financial Agent  │
│              │    │ →并行优化→风险评估 │    │ RAG Pipeline     │
└──────────────┘    └─────────────────┘    └──────────────────┘

【为什么需要任务规划？/ Why Do We Need Task Planning?】

一个看似简单的"分析茅台2023年盈利能力"请求，实际需要完成以下步骤：

1. 从知识库检索茅台2023年财报相关文档（RAG）
2. 获取茅台当前实时行情数据（MCP API）
3. 基于检索到的财报数据做深度财务分析（LLM Agent）
4. 执行独立的风险评估挑战财务分析结论（Risk Agent）
5. 精确计算关键指标数值（量化计算引擎）
6. 整合所有结果生成结构化研报（Report Generator）

这些步骤之间存在依赖关系：
- 步骤3依赖步骤1（没有数据无法分析）
- 步骤4依赖步骤3（需要先有分析结论才能挑战）
- 步骤1和步骤2可以并行（无相互依赖）

TaskPlanner 的职责就是自动完成这种「需求→任务序列」的转化。

【核心设计原则 / Core Design Principles】

1. **模板驱动 + LLM 增强 (Template-Driven + LLM Enhancement)**
   - 主路径：预定义 TASK_TEMPLATES（针对每种意图的专家级任务模板）
   - 兜底：当无匹配模板时，LLM 动态生成任务计划
   - 模板优势：确定性高、可审计、可迭代优化

2. **DAG 依赖建模 (DAG Dependency Modeling)**
   - 任务间关系用有向无环图 (DAG) 表示
   - 支持任意复杂的依赖拓扑（不仅是简单的线性链）
   - 自动检测循环依赖和悬空引用

3. **并行度最大化优化 (Maximum Parallelism Optimization)**
   - 通过拓扑排序将 DAG 分层为可并行执行的"波次"(waves)
   - 同一波次内的任务可以同时执行（如 T1 和 T2 都无前置依赖时）
   - 目标：最小化端到端延迟（wall-clock time），而非最小化总工作量

4. **风险感知规划 (Risk-Aware Planning)**
   - 规划阶段即评估计划风险（关键路径长度、外部依赖数量等）
   - 为每个任务配置重试策略和降级方案
   - 输出风险报告供决策参考

【任务生命周期 / Task Lifecycle】

  PENDING ──▶ RUNNING ──▶ COMPLETED
    │           │            ▲
    │           ▼            │
    │          FAILED ──────┘ (retry)
    │           │
    │           ▼ (max retries exceeded)
    │          SKIPPED
    │
    └────────▶ BLOCKED (依赖未就绪)

【性能特征 / Performance Characteristics】
- 模板匹配：< 1ms（字典查找）
- 参数实例化：< 5ms（字符串替换）
- DAG 验证+拓扑排序：O(V+E)，V=任务数，E=依赖数，通常 < 10ms
- 总体 P99 延迟：< 20ms（纯计算，无 LLM 调用）
- 注意：这是整个 Pipeline 中最快的阶段

【依赖关系 / Dependencies】
- 上游：IntentClassificationResult（提供 intent）、SlotExtractionResult（提供 slots）
- 下游：EnhancedAgentState.task_plan / EnhancedAgentState.selected_skills
- 外部依赖：pydantic（数据校验）、langchain_openai（LLM 动态规划兜底）

【版本信息 / Version Info】
- Author: AI Finance Assistant Team
- Version: 2.0.0 (极致优化版 / Extreme Optimization Edition)
- Last Updated: 2026-04-13
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL


# ─────────────────────────────────────────────────────────────
# 日志配置 / Logging Configuration
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 第一部分：枚举类型定义
# Part 1: Enum Type Definitions
#
# 使用 Python Enum 保证类型安全和 IDE 可发现性。
# 所有枚举值均为 str 类型（继承自 str.Enum）以便于 JSON 序列化。
# ════════════════════════════════════════════════════════════


class TaskType(str, Enum):
    """
    任务类型枚举 (Task Type Enumeration)

    【职责】定义系统中所有可能的原子任务类型（共 8 种）。
    每个 TaskType 对应一个具体的执行单元，由特定的 Agent 或 Skill 处理。

    【类型分类 / Type Categories】

    数据获取类（Data Acquisition）:
    - DATA_RETRIEVAL:    知识库/RAG 检索
    - REALTIME_DATA:    实时行情/API 数据
    - EXTERNAL_API_CALL: 第三方服务调用

    分析处理类 (Analysis & Processing):
    - FINANCIAL_ANALYSIS:  LLM 驱动的深度财务分析
    - RISK_ASSESSMENT:    独立风险评估
    - QUANTITATIVE_CALC:  数值计算（非 LLM）

    输出类 (Output):
    - REPORT_GENERATION: 结构化报告生成
    - EVALUATION:         质量/效果评估
    """

    DATA_RETRIEVAL = "data_retrieval"
    """
    数据检索 (Data Retrieval)
    
    通过 RAG Pipeline 从向量数据库/文档库中检索相关信息。
    典型实现：EnhancedRAGPipeline.retrieve()
    预估耗时：~2s | 预估 Token：~500
    """

    FINANCIAL_ANALYSIS = "financial_analysis"
    """
    财务分析 (Financial Analysis)
    
    由 FinancialAgent（CFA 级分析师 Prompt）执行深度分析。
    输入：RAG 检索结果 + 实时数据 + 用户关注指标
    输出：结构化的多维度分析文本
    预估耗时：~5s | 预估 Token：~2000
    """

    RISK_ASSESSMENT = "risk_assessment"
    """
    风险评估 (Risk Assessment)
    
    由 RiskAgent（独立风控视角）对分析结论进行挑战性审查。
    设计理念："红蓝对抗"——让两个 Agent 相互验证以提高可靠性。
    预估耗时：~4s | 预估 Token：~1500
    """

    REALTIME_DATA = "realtime_data"
    """
    实时数据获取 (Real-Time Data Fetching)
    
    通过 MCP 工具或直接 API 调用获取实时市场数据。
    典型工具：StockQuote（Alpha Vantage）、News API
    预估耗时：~1.5s | 预估 Token：~300
    """

    QUANTITATIVE_CALC = "quantitative_calc"
    """
    量化计算 (Quantitative Calculation)
    
    纯数值计算任务（不涉及 LLM 推理）。包括：
    - 财务比率精确计算（ROE = NetIncome / AvgEquity）
    - 统计指标（均值、标准差、相关系数）
    - DCF 估值模型中的数学运算
    预估耗时：~0.5s | 预估 Token：~100
    """

    REPORT_GENERATION = "report_generation"
    """
    报告生成 (Report Generation)
    
    将所有分析和计算结果整合为结构化的 Markdown 研报。
    包含：摘要、详细分析、风险提示、投资建议等章节。
    预估耗时：~6s | 预估 Token：~2500（最耗资源的任务之一）
    """

    EVALUATION = "evaluation"
    """
    质量评估 (Quality Evaluation)
    
    由 ComprehensiveEvaluator 对输出进行多维度评分。
    包括：RAG 质量、Agent 执行质量、输出文本质量。
    用于持续改进和 A/B 测试。
    预估耗时：~3s | 预估 Token：~800
    """

    EXTERNAL_API_CALL = "external_api_call"
    """
    外部 API 调用 (External API Call)
    
    调用第三方服务的通用任务类型。例如：
    - 新闻资讯 API
    - 社交媒体情感 API
    - 行业数据库查询
    预估耗时：~1.0s | 预估 Token：~200
    """


class TaskStatus(str, Enum):
    """
    任务状态枚举 (Task Status Enumeration)

    【职责】跟踪单个任务在执行过程中的状态变迁。

    [状态转换规则 / State Transition Rules]
    PENDING → RUNNING:   任务被调度器选中开始执行
    RUNNING → COMPLETED: 任务成功完成
    RUNNING → FAILED:    任务执行出错
    FAILED → RUNNING:    重试（未超过 max_retries 时）
    FAILED → SKIPPED:    放弃重试（超过 max_retries 或标记为 non-critical）
    PENDING → BLOCKED:   前置依赖未满足（由调度器设置）
    BLOCKED → PENDING:   所有依赖已完成（由调度器设置）
    """

    PENDING = "pending"
    """等待执行（初始状态，已创建但尚未调度）"""

    RUNNING = "running"
    """执行中（已被工作线程/协程领取并开始处理）"""

    COMPLETED = "completed"
    """已成功完成（result 字段包含输出数据）"""

    FAILED = "failed"
    """失败（error 字段包含错误信息，可重试）"""

    SKIPPED = "skipped"
    """跳过（因重试耗尽或优先级过低而放弃）"""

    BLOCKED = "blocked"
    """
    被阻塞（存在未完成的前置依赖任务）。
    调度器在每轮调度前检查此状态，当所有 dependencies 的状态均为
    COMPLETED/SKIPPED 时，自动将 BLOCKED 转为 PENDING。
    """


class Priority(str, Enum):
    """
    优先级枚举 (Priority Enumeration)

    【职责】定义任务的相对重要性，影响调度顺序和资源分配策略。

    [优先级语义 / Priority Semantics]

    级别       含义                  调度行为                     资源保障
    ──────────────────────────────────────────────────────────────────
    CRITICAL   核心关键任务           最先调度，抢占式执行         最高资源配额
    HIGH       重要但非阻塞           仅次于 CRITICAL             高资源配额
    MEDIUM     标准优先级（默认）      正常 FIFO 顺序             标准资源配额
    LOW        可选增强任务           资源充裕时才执行             最低资源配额
    """

    CRITICAL = "critical"
    """关键：必须成功完成，失败将导致整体流程中断"""

    HIGH = "high"
    """高：强烈建议完成，失败会显著降低输出质量"""

    MEDIUM = "medium"
    """中：标准优先级（默认值），尽力完成即可"""

    LOW = "low"
    """低：锦上添花型任务，资源紧张时可跳过"""


# ════════════════════════════════════════════════════════════
# 第二部分：核心数据模型
# Part 2: Core Data Models
# ════════════════════════════════════════════════════════════


@dataclass
class Task:
    """
    单个任务定义 (Single Task Definition)

    【职责】描述一个原子性工作单元的完整元信息。
    是 TaskPlan 的基本组成元素，也是调度器的工作单元。

    【字段分组 / Field Groups】

    身份标识: task_id, task_type, description
    执行参数: parameters, assigned_agent, assigned_skill
    依赖关系: dependencies
    状态管理: status, priority, result, error
    时间追踪: created_at, started_at, completed_at
    容错机制: retry_count, max_retries
    资源预估: estimated_tokens, estimated_time_seconds
    """

    task_id: str
    """
    任务唯一标识符。
    
    格式约定：T{序号}，如 T1, T2, ..., T10。
    序号从模板中的位置决定（1-based index），
    在同一 TaskPlan 内全局唯一。
    用途：作为依赖关系图中的节点标识。
    """

    task_type: TaskType
    """任务类型（来自 TaskType 枚举），决定了由哪个 Agent/Skill 处理"""

    description: str
    """
    人类可读的任务描述。
    
    经槽位参数填充后的完整描述，如：
    "获取贵州茅台2023年财报数据"（原始模板为"获取{company}{year}年财报数据"）
    用于日志记录和进度展示。
    """

    # ── 执行参数 / Execution Parameters ──

    parameters: Dict[str, Any] = field(default_factory=dict)
    """
    任务执行所需的键值对参数。
    
    内容取决于 task_type：
    - DATA_RETRIEVAL: {"query": "贵州茅台2023年", "use_hyde": True}
    - REALTIME_DATA: {"symbol": "600519.SH"}
    - FINANCIAL_ANALYSIS: {"analysis_type": "毛利率,净利率,ROE"}
    
    特殊保留键（以 _ 开头）：
    - _original_query: 用户原始查询（用于上下文回溯）
    - _slots: 完整槽位信息（用于运行时动态决策）
    """

    # ── 依赖关系 / Dependencies ──

    dependencies: List[str] = field(default_factory=list)
    """
    本任务的前置依赖列表（存储的是其他任务的 task_id）。
    
    语义：只有当 dependencies 中所有任务的状态为 COMPLETED 或 SKIPPED 时，
    本任务才能从 BLOCKED 转为 PENDING 并被调度执行。
    
    示例：dependencies=["T1", "T3"] 表示必须等 T1 和 T3 都完成后才能执行本任务。
    空列表 [] 表示无前置依赖（可以在第一波次立即执行）。
    """

    # ── 分配信息 / Assignment Info ──

    assigned_agent: str = ""
    """
    负责执行此任务的 Agent 名称。
    
    映射关系：
    - "financial_agent"   → src.agents.financial_agent.FinancialAgent
    - "risk_agent"        → src.agents.risk_agent.RiskAgent
    - "retrieval_agent"   → src.enhanced_rag.EnhancedRAGPipeline
    - "realtime_agent"    → src.agents.realtime_agent.RealtimeAgent
    - "report_generator"  → src.report.generator.ReportGenerator
    
    空字符串表示使用默认路由逻辑。
    """

    assigned_skill: Optional[str] = None
    """
    负责执行此任务的 Skill 名称（可选）。
    
    当 assigned_agent 和 assigned_skill 同时指定时：
    Agent 负责"编排"，Skill 提供"能力"。
    例如：assigned_agent="financial_agent", assigned_skill="comparative_analysis"
    表示由 FinancialAgent 使用对比分析 Skill 来完成任务。
    """

    # ── 状态管理 / Status Management ──

    status: TaskStatus = TaskStatus.PENDING
    """当前任务状态（初始值为 PENDING）"""

    priority: Priority = Priority.MEDIUM
    """任务优先级（初始值为 MEDIUM，可在模板中覆盖）"""

    # ── 执行结果 / Execution Results ──

    result: Optional[Any] = None
    """
    任务成功执行后的输出数据。
    
    类型取决于 task_type：
    - DATA_RETRIEVAL:     List[RetrievalResult]
    - FINANCIAL_ANALYSIS: str（Markdown 分析文本）
    - REALTIME_DATA:     dict（JSON 格式的行情数据）
    - QUANTITATIVE_CALC:  Dict[str, float]（指标名→数值）
    - REPORT_GENERATION: str（完整 Markdown 报告）
    """

    error: Optional[str] = None
    """任务失败时的错误信息（仅当 status=FAILED 时有值）"""

    # ── 时间追踪 / Time Tracking ──

    created_at: str = ""
    """任务创建时间（ISO 8601 格式 UTC 字符串）"""

    started_at: Optional[str] = None
    """任务实际开始执行的时间"""

    completed_at: Optional[str] = None
    """任务完成（成功或失败）的时间"""

    # ── 容错机制 / Fault Tolerance ──

    retry_count: int = 0
    """已重试次数（初始为 0，每次失败后 +1）"""

    max_retries: int = 2
    """
    最大允许重试次数（默认 2 次）。
    
    超过此次数后，若 priority != CRITICAL 则转为 SKIPPED；
    若 priority == CRITICAL 则保持 FAILED（导致整体流程中断）。
    """

    # ── 资源预估 / Resource Estimation ──

    estimated_tokens: int = 0
    """
    预估的 LLM Token 消耗量。
    
    用于：
    1. 成本预算控制（单次查询不超过 N tokens）
    2. 资源分配决策（Token 密集型任务分配到高性能节点）
    3. 进度估算（剩余 Token 量 → 预计剩余时间）
    """

    estimated_time_seconds: float = Field(
        default=0.0,
        description="预估的执行耗时（秒，P50中位数）。用于超时保护、进度展示、SLA检查",
    )


@dataclass
class TaskPlan:
    """
    完整任务计划 (Complete Task Plan)

    【职责】封装一次规划操作产出的所有任务及其组织结构。
    是 TaskPlanner.plan() 的主要输出，传递给下游的调度器和执行引擎。

    【统计字段含义 / Statistics Fields Meaning】
    - total_tasks: 计划中的任务总数
    - critical_path_length: 关键路径上的任务数（= 拓扑排序层数 = 最短可能执行轮次）
    - parallel_groups: 包含多个任务的并行组数量（>1 表示有并行机会）
    """

    plan_id: str
    """
    计划唯一标识符。
    
    格式：UUID 的前 8 位（如 "a3f7b2c1"）。
    用于日志追踪和调试关联。
    """

    user_query: str
    """触发此计划的用户原始查询（用于审计和展示）"""

    intent: str
    """此计划对应的主意图（来自 IntentClassificationResult.primary_intent.value）"""

    tasks: List[Task] = field(default_factory=list)
    """有序任务列表（按模板定义顺序排列）"""

    # ── 统计信息 / Statistics ──

    total_tasks: int = 0
    """任务总数 (= len(tasks)，冗余存储便于快速访问）"""

    critical_path_length: int = 0
    """
    关键路径长度（= 拓扑排序的层数 / 波次数）。
    
    这是理论上最短的执行轮次数（假设无限并行资源）。
    例如：如果 critical_path_length=4，则至少需要 4 个串行步骤才能完成全部任务。
    """

    parallel_groups: int = 1
    """
    可并行执行的分组数量。
    
    值为 1 表示完全串行（所有任务都在各自的独立波次中）。
    值 > 1 表示存在可并行的波次（同波次内多个任务可同时执行）。
    此值越大，并行优化的空间越大。
    """

    # ── 元数据 / Metadata ──

    created_at: str = ""
    """计划创建时间（ISO 8601 格式）"""

    planning_strategy: str = ""
    """
    使用的规划策略标识。
    
    格式："{method}_{intent}"，如 "template_based_financial_analysis"
    可能的方法：template_based（模板驱动）/ llm_generated（LLM 动态生成）/ hybrid（混合）
    """

    optimization_notes: List[str] = field(default_factory=list)
    """
    规划过程中的优化说明列表。
    
    示例：
    - "T1 与 T2 无依赖，已合并至第1并行组"
    - "T4 依赖 T3，移至第3波次"
    - "检测到 T5 的外部依赖，增加超时保护"
    """


class TaskPlanningRequest(BaseModel):
    """
    任务规划请求模型 (Task Planning Request Schema)

    【职责】封装调用方传入的规划参数，是 TaskPlanner.plan() 的唯一入参。
    """

    query: str = Field(
        ...,
        min_length=1,
        description="用户原始查询文本",
    )

    intent: str = Field(
        ...,
        description=(
            "上游 IntentClassifier 识别出的主意图标签。"
            "来自 PrimaryIntent 枚举的 .value 属性，"
            "如 'financial_analysis'、'stock_comparison' 等。"
            "TaskPlanner 用此值在 TASK_TEMPLATES 中查找对应的任务模板。"
        ),
    )

    slots: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "上游 SlotExtractor 提取的结构化槽位信息。"
            "用于填充任务模板中的占位符（如 {company}、{year} 等）。"
            "格式与 SlotExtractionResult.model_dump() 一致。"
        ),
    )

    intent_hierarchy: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "完整的意图层级结构（含子意图和候选列表）。"
            "可用于更精细的模板选择（如根据子意图选择不同变体模板）。"
        ),
    )

    context: Optional[str] = Field(
        default=None,
        description="多轮对话上下文（可选）",
    )


class TaskPlanningResult(BaseModel):
    """
    任务规划结果模型 (Task Planning Result Schema)

    【职责】封装一次规划操作的完整产出，是本模块对外输出的唯一数据结构。
    """

    success: bool = Field(
        default=True,
        description="规划操作是否成功",
    )

    plan: Optional[TaskPlan] = Field(
        default=None,
        description="生成的完整任务计划对象",
    )

    execution_order: List[List[str]] = Field(
        default_factory=list,
        description=(
            "优化后的执行顺序（支持并行的二维数组）。"
            "\n结构说明：外层列表 = 波次（wave），内层列表 = 该波次的任务 ID 列表。"
            "\n示例：[['T1', 'T2'], ['T3'], ['T4', 'T5', 'T6']] "
            "表示第1波次 T1/T2 并行，第2波次 T3，第3波次 T4/T5/T6 并行。"
            "\n下游调度器据此安排并发执行。"
        ),
    )

    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "任务依赖关系的邻接表表示。"
            "\n格式：{task_id: [dependency_task_ids]}"
            "\n示例：{'T3': ['T1'], 'T4': ['T3'], 'T6': ['T3', 'T4', 'T5']}"
            "\n用于可视化和调试（如用 Graphviz 渲染 DAG 图）。"
        ),
    )

    planning_time_ms: float = Field(
        default=0.0,
        description="规划耗时（毫秒），用于性能监控",
    )

    total_estimated_cost: float = Field(
        default=0.0,
        description="预估总 Token 消耗（所有任务之和），用于成本控制",
    )

    risk_assessment: str = Field(
        default="",
        description="计划级别的风险评估摘要（低/中/高风险 + 具体原因）",
    )


# ════════════════════════════════════════════════════════════
# 第三部分：任务模板库
# Part 3: Task Template Library
#
# 这是规划器的"知识库"。每种意图对应一组预定义的任务模板，
# 定义了该意图下需要执行的所有步骤、它们之间的依赖关系、
# 以及每个步骤的执行参数。
#
# 模板设计原则：
# 1. 幂等性：同一输入永远产生相同输出
# 2. 最小依赖：尽量减少任务间的依赖以提升并行度
# 3. 容错性：CRITICAL 任务有明确的重试策略
# ════════════════════════════════════════════════════════════

TASK_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    # ─────────────────────────────────────────────────────────
    # 模板集 1: 财务分析 (financial_analysis) — 6 个任务
    #
    # [DAG 拓扑]
    #   Wave 1: T1(数据检索) + T2(实时行情)  ← 并行 ✓
    #   Wave 2: T3(财务分析) + T5(量化计算)   ← 并行 ✓（都依赖 T1）
    #   Wave 3: T4(风险评估)                 ← 依赖 T3
    #   Wave 4: T6(报告生成)                 ← 依赖 T3+T4+T5
    #
    # 关键路径: T1 → T3 → T4 → T6 (长度 4)
    # 并行机会: 2 组（Wave 1 和 Wave 2 各有并行）
    # ─────────────────────────────────────────────────────────
    "financial_analysis": [
        {
            "task_type": TaskType.DATA_RETRIEVAL,
            "description": "获取{company}{year}年财报数据",
            "agent": "retrieval_agent",
            "skill": "rag_retrieval",
            "priority": Priority.CRITICAL,
            "dependencies": [],
            "parameters": {"query": "{query}", "use_hyde": True},
        },
        {
            "task_type": TaskType.REALTIME_DATA,
            "description": "获取{company}实时行情数据",
            "agent": "realtime_agent",
            "skill": "stock_quote",
            "priority": Priority.HIGH,
            "dependencies": [],
            "parameters": {"symbol": "{stock_code}"},
        },
        {
            "task_type": TaskType.FINANCIAL_ANALYSIS,
            "description": "基于财报数据进行深度财务分析",
            "agent": "financial_agent",
            "skill": "financial_analysis",
            "priority": Priority.CRITICAL,
            "dependencies": ["T1"],
            "parameters": {"analysis_type": "{metrics}"},
        },
        {
            "task_type": TaskType.RISK_ASSESSMENT,
            "description": "执行独立风险评估",
            "agent": "risk_agent",
            "skill": "risk_assessment",
            "priority": Priority.HIGH,
            "dependencies": ["T3"],
            "parameters": {},
        },
        {
            "task_type": TaskType.QUANTITATIVE_CALC,
            "description": "精确计算关键财务指标",
            "agent": "quantitative_agent",
            "skill": "metric_calculator",
            "priority": Priority.MEDIUM,
            "dependencies": ["T1"],
            "parameters": {"metrics": "{primary_metrics}"},
        },
        {
            "task_type": TaskType.REPORT_GENERATION,
            "description": "整合所有分析结果，生成研报",
            "agent": "report_generator",
            "skill": "report_generation",
            "priority": Priority.CRITICAL,
            "dependencies": ["T3", "T4", "T5"],
            "parameters": {},
        },
    ],

    # ─────────────────────────────────────────────────────────
    # 模板集 2: 个股对比 (stock_comparison) — 5 个任务
    #
    # 特点：两家公司的数据检索互相独立，天然并行
    # ─────────────────────────────────────────────────────────
    "stock_comparison": [
        {
            "task_type": TaskType.DATA_RETRIEVAL,
            "description": "获取{company1}财报数据",
            "agent": "retrieval_agent",
            "skill": "rag_retrieval",
            "priority": Priority.CRITICAL,
            "dependencies": [],
            "parameters": {"query": "{company1}", "target": "company1"},
        },
        {
            "task_type": TaskType.DATA_RETRIEVAL,
            "description": "获取{company2}财报数据",
            "agent": "retrieval_agent",
            "skill": "rag_retrieval",
            "priority": Priority.CRITICAL,
            "dependencies": [],
            "parameters": {"query": "{company2}", "target": "company2"},
        },
        {
            "task_type": TaskType.REALTIME_DATA,
            "description": "获取对比公司实时行情",
            "agent": "realtime_agent",
            "skill": "multi_stock_quote",
            "priority": Priority.HIGH,
            "dependencies": [],
            "parameters": {"symbols": ["{code1}", "{code2}"]},
        },
        {
            "task_type": TaskType.FINANCIAL_ANALYSIS,
            "description": "对比分析两家公司的财务表现",
            "agent": "financial_agent",
            "skill": "comparative_analysis",
            "priority": Priority.CRITICAL,
            "dependencies": ["T1", "T2"],
            "parameters": {},
        },
        {
            "task_type": TaskType.REPORT_GENERATION,
            "description": "生成对比分析报告",
            "agent": "report_generator",
            "skill": "comparison_report",
            "priority": Priority.CRITICAL,
            "dependencies": ["T3", "T4"],
            "parameters": {},
        },
    ],

    # ─────────────────────────────────────────────────────────
    # 模板集 3: 实时查询 (realtime_query) — 3 个任务
    #
    # 特点：轻量级快速响应，减少不必要的分析步骤
    # ─────────────────────────────────────────────────────────
    "realtime_query": [
        {
            "task_type": TaskType.REALTIME_DATA,
            "description": "获取实时行情数据",
            "agent": "realtime_agent",
            "skill": "stock_quote",
            "priority": Priority.CRITICAL,
            "dependencies": [],
            "parameters": {"symbol": "{stock_code}"},
        },
        {
            "task_type": TaskType.EXTERNAL_API_CALL,
            "description": "获取最新新闻资讯",
            "agent": "realtime_agent",
            "skill": "news_retrieval",
            "priority": Priority.MEDIUM,
            "dependencies": [],
            "parameters": {"query": "{company}"},
        },
        {
            "task_type": TaskType.REPORT_GENERATION,
            "description": "生成行情简报",
            "agent": "report_generator",
            "skill": "quick_report",
            "priority": Priority.CRITICAL,
            "dependencies": ["T1", "T2"],
            "parameters": {},
        },
    ],
}


# ════════════════════════════════════════════════════════════
# 第四部分：任务规划器核心类
# Part 4: Task Planner Core Class
# ════════════════════════════════════════════════════════════


class TaskPlanner:
    """
    金融领域任务规划器 (Financial Domain Task Planner)

    【类职责 / Class Responsibility】
    作为本模块的门面 (Facade)，封装完整的「模板选择 → 参数实例化 →
    DAG 验证 → 拓扑排序 → 风险评估」五阶段规划流水线。

    【流水线概览 / Pipeline Overview】

    Input: TaskPlanningRequest(query, intent, slots)
        │
        ▼
    ┌─ Step 1: Template Selection ─────────────┐
    │  根据 intent 在 TASK_TEMPLATES 中查找    │
    │  支持直接匹配 / 子意图映射 / LLM 兜底    │
    │  Output: template (List[Dict]) or None    │
    └──────────────────────────────────────────┘
        │
        ▼
    ┌─ Step 2: Task Instantiation ──────────────┐
    │  用 slots 填充模板中的占位符               │
    │  创建 Task 对象列表                        │
    │  Output: List[Task]                        │
    └──────────────────────────────────────────┘
        │
        ▼
    ┌─ Step 3: DAG Validation ──────────────────┐
    │  检查依赖合法性（无悬空引用/自依赖/循环）  │
    │  自动修复可修复的问题                      │
    └──────────────────────────────────────────┘
        │
        ▼
    ┌─ Step 4: Topological Sort ────────────────┐
    │  Kahn's 算法分层                           │
    │  Output: List[List[Task]] (可并行的波次)   │
    └──────────────────────────────────────────┘
        │
        ▼
    ┌─ Step 5: Plan Assembly ───────────────────┐
    │  构建 TaskPlan + 统计信息 + 风险评估        │
    │  Output: TaskPlanningResult                │
    └──────────────────────────────────────────┘

    [温度参数 rationale / Temperature Rationale]
    temperature=0（规划需要确定性和可重复性——相同输入应产生相同计划）
    """

    def __init__(self) -> None:
        """
        初始化任务规划器实例。

        [初始化内容 / Initialization Tasks]
        创建 LLM 客户端（用于无匹配模板时的动态规划兜底）。
        大多数情况下不会用到（因为 TASK_TEMPLATES 已覆盖主流场景）。
        """
        self._llm: ChatOpenAI = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )

    def plan(self, request: TaskPlanningRequest) -> TaskPlanningResult:
        """
        执行任务规划的入口方法。

        [Args / 参数]
            request: 经 Pydantic 校验的规划请求

        [Returns / 返回值]
            TaskPlanningResult: 包含 plan/execution_order/risk_assessment 等完整信息
        """
        start_time: datetime = datetime.now()

        logger.info("[任务规划] 开始处理: %s...", request.query[:50])
        logger.info("[任务规划] 意图: %s", request.intent)

        # Step 1: 选择任务模板
        template: Optional[List[Dict[str, Any]]] = self._select_template(request.intent)
        if not template:
            logger.warning("[任务规划] ⚠️ 未找到匹配的模板，使用默认模板")
            template = TASK_TEMPLATES.get("financial_analysis", [])

        # Step 2: 实例化任务（填充参数）
        tasks: List[Task] = self._instantiate_tasks(
            template, request.slots or {}, request.query
        )

        # Step 3: 构建依赖图并验证
        self._validate_dependencies(tasks)

        # Step 4: 优化执行顺序（拓扑排序）
        execution_order: List[List[Task]] = self._optimize_execution_order(tasks)

        # Step 5: 构建完整计划
        plan: TaskPlan = TaskPlan(
            plan_id=str(uuid.uuid4())[:8],
            user_query=request.query,
            intent=request.intent,
            tasks=tasks,
            total_tasks=len(tasks),
            created_at=start_time.isoformat(),
            planning_strategy=f"template_based_{request.intent}",
        )
        plan.critical_path_length = len(execution_order)
        plan.parallel_groups = sum(1 for group in execution_order if len(group) > 1)

        total_cost: float = sum(t.estimated_tokens for t in tasks)
        end_time: datetime = datetime.now()
        planning_time_ms: float = (end_time - start_time).total_seconds() * 1000

        result = TaskPlanningResult(
            success=True,
            plan=plan,
            execution_order=[[t.task_id for t in group] for group in execution_order],
            dependency_graph={t.task_id: t.dependencies for t in tasks},
            planning_time_ms=planning_time_ms,
            total_estimated_cost=total_cost,
            risk_assessment=self._assess_plan_risk(plan),
        )

        logger.info(
            "[任务规划] 完成: %d个任务, %d个执行阶段, 耗时%.0fms",
            len(tasks),
            len(execution_order),
            planning_time_ms,
        )
        return result

    def _select_template(self, intent: str) -> Optional[List[Dict[str, Any]]]:
        """
        根据意图选择任务模板。

        [匹配策略 / Matching Strategy - 优先级从高到低]
        1. 直接匹配：intent 键直接存在于 TASK_TEMPLATES
        2. 子意图映射：通过 INTENT_MAPPING 将子意图映射到主意图
        3. 返回 None：交由 LLM 动态生成（未来扩展点）
        """
        if intent in TASK_TEMPLATES:
            return TASK_TEMPLATES[intent]

        intent_mapping: Dict[str, str] = {
            "profitability_analysis": "financial_analysis",
            "growth_analysis": "financial_analysis",
            "cashflow_analysis": "financial_analysis",
            "valuation_analysis": "financial_analysis",
            "technical_analysis": "financial_analysis",
            "fundamental_analysis": "financial_analysis",
            "overview": "industry_overview",
            "deep_dive": "industry_overview",
            "trend_prediction": "industry_overview",
        }

        mapped_intent: Optional[str] = intent_mapping.get(intent)
        if mapped_intent and mapped_intent in TASK_TEMPLATES:
            return TASK_TEMPLATES[mapped_intent]

        return None

    def _instantiate_tasks(
        self,
        template: List[Dict[str, Any]],
        slots: Dict[str, Any],
        query: str,
    ) -> List[Task]:
        """
        用槽位信息实例化任务模板，创建 Task 对象列表。

        [实例化过程 / Instantiation Process]
        FOR each task_template in template:
            1. 生成 task_id = f"T{i}"（i 从 1 开始）
            2. 用 slots 填充 description 中的占位符
            3. 用 slots 填充 parameters 中的占位符
            4. 注入 _original_query 和 _slots 到 parameters
            5. 预估 token 消耗和耗时
            6. 创建 Task 对象并追加到列表
        """
        tasks: List[Task] = []

        for i, task_template in enumerate(template, 1):
            task_id: str = f"T{i}"

            description: str = self._fill_placeholders(
                task_template["description"], slots
            )

            parameters: Dict[str, Any] = {}
            if "parameters" in task_template:
                for key, value_template in task_template["parameters"].items():
                    if isinstance(value_template, str):
                        parameters[key] = self._fill_placeholders(
                            value_template, slots
                        )
                    else:
                        parameters[key] = value_template

            task = Task(
                task_id=task_id,
                task_type=task_template["task_type"],
                description=description,
                parameters={
                    **parameters,
                    "_original_query": query,
                    "_slots": slots,
                },
                dependencies=task_template.get("dependencies", []),
                assigned_agent=task_template.get("agent", ""),
                assigned_skill=task_template.get("skill"),
                priority=task_template.get("priority", Priority.MEDIUM),
                created_at=datetime.now().isoformat(),
                estimated_tokens=self._estimate_tokens(task_template["task_type"]),
                estimated_time_seconds=self._estimate_time(task_template["task_type"]),
            )
            tasks.append(task)

        return tasks

    def _fill_placeholders(self, text: str, slots: Dict[str, Any]) -> str:
        """
        替换文本中的占位符为实际的槽位值。

        [支持的占位符 / Supported Placeholders]

        占位符              来源                          缺失时替换为
        ─────────────────────────────────────────────────────────────
        {company}           slots.companies[0].name     "[待补充]"
        {stock_code}        slots.companies[0].stock_code ""
        {year}              slots.time_info.year          ""
        {time_range}        slots.time_info.period        ""
        {query}             slots.raw_query               ""
        {metrics}           slots.metrics.primary_metrics  ""
        {primary_metrics}   slots.metrics.primary_metrics  ""
        {company1}          slots.companies[0] 或 comparison_targets[0]
        {company2}          slots.companies[1] 或 comparison_targets[1]
        {code1}             第一个公司的 stock_code
        {code2}             第二个公司的 stock_code
        """
        if not slots:
            return text

        result: str = text

        placeholder_map: Dict[str, str] = {
            "{company}": (
                slots.get("companies", [{}])[0].get("name", "")
                if slots.get("companies") else ""
            ),
            "{stock_code}": (
                slots.get("companies", [{}])[0].get("stock_code", "")
                if slots.get("companies") else ""
            ),
            "{year}": (
                str(slots.get("time_info", {}).get("year", ""))
                if slots.get("time_info") else ""
            ),
            "{time_range}": (
                slots.get("time_info", {}).get("period", "")
                if slots.get("time_info") else ""
            ),
            "{query}": slots.get("raw_query", ""),
            "{metrics}": (
                ", ".join(slots.get("metrics", {}).get("primary_metrics", []))
                if slots.get("metrics") else ""
            ),
            "{primary_metrics}": (
                ", ".join(slots.get("metrics", {}).get("primary_metrics", []))
                if slots.get("metrics") else ""
            ),
        }

        companies = slots.get("companies", [])
        comparison_targets = slots.get("comparison_targets", [])
        all_companies: List[Any] = companies + comparison_targets

        if len(all_companies) >= 1:
            c0 = all_companies[0]
            placeholder_map["{company1}"] = (
                c0.get("name", "") if isinstance(c0, dict) else str(c0)
            )
            placeholder_map["{code1}"] = (
                c0.get("stock_code", "") if isinstance(c0, dict) else ""
            )
        if len(all_companies) >= 2:
            c1 = all_companies[1]
            placeholder_map["{company2}"] = (
                c1.get("name", "") if isinstance(c1, dict) else str(c1)
            )
            placeholder_map["{code2}"] = (
                c1.get("stock_code", "") if isinstance(c1, dict) else ""
            )

        for placeholder, value in placeholder_map.items():
            result = result.replace(placeholder, value if value else "[待补充]")

        return result

    def _validate_dependencies(self, tasks: List[Task]) -> None:
        """
        验证任务依赖关系的合法性。

        [检查项 / Validation Checks]
        1. 悬空引用：依赖的 task_id 不存在于任务列表中 → 移除该依赖
        2. 自依赖：任务依赖自身 → 移除该依赖（循环依赖的最简形式）
        
        注意：完整的循环检测（A→B→C→A）在拓扑排序阶段隐式处理
        （Kahn's 算法在无法找到入度为 0 的节点时会检测到循环）
        """
        task_ids: Set[str] = {t.task_id for t in tasks}

        for task in tasks:
            invalid_deps: List[str] = [
                dep_id for dep_id in task.dependencies if dep_id not in task_ids
            ]
            for dep_id in invalid_deps:
                logger.warning(
                    "[任务规划] ⚠️ 任务%s依赖不存在的任务%s", task.task_id, dep_id
                )
                task.dependencies.remove(dep_id)

            if task.task_id in task.dependencies:
                logger.warning("[任务规划] ❌ 任务%s存在自依赖", task.task_id)
                task.dependencies.remove(task.task_id)

    def _optimize_execution_order(self, tasks: List[Task]) -> List[List[Task]]:
        """
        通过拓扑排序优化执行顺序，最大化并行度。

        [算法 / Algorithm: Kahn's BFS Topological Sort with Layering]

        1. 计算每个节点的入度（incoming edge count）
        2. 初始层：所有入度为 0 的节点
        3. FOR each layer:
           a. 将入度为 0 的节点归入当前层
           b. 移除这些节点，更新剩余节点的入度
           c. 如果剩余节点中没有入度为 0 的节点 → 存在循环
              （强制加入剩余节点作为降级处理）
        4. 返回层的列表（每层内的节点可并行）

        [时间复杂度 / Time Complexity]
        O(V + E)，其中 V = 任务数，E = 依赖边数
        对于典型的金融分析模板（V ≤ 10, E ≤ 15），< 1ms

        [Returns / 返回值]
        List[List[Task]]: 分层后的任务列表（外层=波次，内层=该波次任务）
        """
        in_degree: Dict[str, int] = {
            t.task_id: len(t.dependencies) for t in tasks
        }

        layers: List[List[Task]] = []
        remaining_tasks: Dict[str, Task] = {t.task_id: t for t in tasks}

        while remaining_tasks:
            current_layer: List[Task] = [
                task
                for task in remaining_tasks.values()
                if in_degree[task.task_id] == 0
            ]

            if not current_layer:
                logger.warning("[任务规划] ⚠️ 检测到可能的循环依赖，强制添加剩余任务")
                current_layer = list(remaining_tasks.values())

            layers.append(current_layer)

            for task in current_layer:
                del remaining_tasks[task.task_id]
                for other_task in remaining_tasks.values():
                    if task.task_id in other_task.dependencies:
                        in_degree[other_task.task_id] -= 1

        return layers

    @staticmethod
    def _estimate_tokens(task_type: TaskType) -> int:
        """预估任务的 LLM Token 消耗量"""
        estimates: Dict[TaskType, int] = {
            TaskType.DATA_RETRIEVAL: 500,
            TaskType.FINANCIAL_ANALYSIS: 2000,
            TaskType.RISK_ASSESSMENT: 1500,
            TaskType.REALTIME_DATA: 300,
            TaskType.QUANTITATIVE_CALC: 100,
            TaskType.REPORT_GENERATION: 2500,
            TaskType.EVALUATION: 800,
            TaskType.EXTERNAL_API_CALL: 200,
        }
        return estimates.get(task_type, 500)

    @staticmethod
    def _estimate_time(task_type: TaskType) -> float:
        """预估任务执行耗时（秒，P50 中位数）"""
        estimates: Dict[TaskType, float] = {
            TaskType.DATA_RETRIEVAL: 2.0,
            TaskType.FINANCIAL_ANALYSIS: 5.0,
            TaskType.RISK_ASSESSMENT: 4.0,
            TaskType.REALTIME_DATA: 1.5,
            TaskType.QUANTITATIVE_CALC: 0.5,
            TaskType.REPORT_GENERATION: 6.0,
            TaskType.EVALUATION: 3.0,
            TaskType.EXTERNAL_API_CALL: 1.0,
        }
        return estimates.get(task_type, 2.0)

    def _assess_plan_risk(self, plan: TaskPlan) -> str:
        """
        评估任务计划的整体风险等级。

        [评估维度 / Assessment Dimensions]

        维度                   条件                              风险贡献
        ─────────────────────────────────────────────────────────────
        关键路径长度           critical_tasks > 3                "+单点故障风险高"
        外部依赖密度           external_deps > 2                 "+网络延迟/不可用风险"
        并行度不足             parallel_groups == 1              "+整体耗时较长"

        [输出格式 / Output Format]
        - 0 个风险因子 → "低风险 - 计划结构合理"
        - 1-2 个风险因子 → "中等风险 - {原因1}; {原因2}"
        - 3+ 个风险因子 → "高风险 - {原因1}; {原因2}; ..."
        """
        risks: List[str] = []

        critical_tasks: List[Task] = [
            t for t in plan.tasks if t.priority == Priority.CRITICAL
        ]
        if len(critical_tasks) > 3:
            risks.append("关键路径较长，单点故障风险较高")

        external_deps: List[Task] = [
            t
            for t in plan.tasks
            if t.task_type in (TaskType.REALTIME_DATA, TaskType.EXTERNAL_API_CALL)
        ]
        if len(external_deps) > 2:
            risks.append("多个外部API调用，网络延迟或服务不可用风险")

        if plan.parallel_groups == 1:
            risks.append("任务串行执行，整体耗时较长")

        if not risks:
            return "低风险 - 计划结构合理"
        elif len(risks) <= 2:
            return f"中等风险 - {'; '.join(risks)}"
        else:
            return f"高风险 - {'; '.join(risks)}"


# ════════════════════════════════════════════════════════════
# 第五部分：公共便捷函数
# Part 5: Public Convenience Function
# ════════════════════════════════════════════════════════════


def create_task_plan(
    query: str,
    intent: str,
    slots: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> TaskPlanningResult:
    """
    便捷函数：一步创建任务计划。

    [Args / 参数]
        query:   用户查询文本
        intent:  主意图标签
        slots:   槽位信息（可选）
        context: 对话上下文（可选）

    [Returns / 返回值]
        TaskPlanningResult: 完整的任务规划结果
    """
    planner = TaskPlanner()
    request = TaskPlanningRequest(
        query=query,
        intent=intent,
        slots=slots,
        context=context,
    )
    return planner.plan(request)


# ════════════════════════════════════════════════════════════
# 第六部分：模块自测
# Part 6: Self-Test Suite
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases: List[Dict[str, Any]] = [
        {
            "query": "分析贵州茅台2023年的盈利能力和现金流",
            "intent": "financial_analysis",
            "slots": {
                "companies": [{"name": "贵州茅台", "stock_code": "600519.SH"}],
                "time_info": {"year": 2023},
                "metrics": {
                    "primary_metrics": ["毛利率", "净利率", "ROE", "经营现金流"],
                },
            },
        },
        {
            "query": "对比茅台和五粮液的估值水平",
            "intent": "stock_comparison",
            "slots": {
                "companies": [{"name": "贵州茅台", "stock_code": "600519.SH"}],
                "comparison_targets": [{"name": "五粮液", "stock_code": "000858.SZ"}],
                "metrics": {"primary_metrics": ["PE", "PB"]},
            },
        },
        {
            "query": "比亚迪今天股价多少",
            "intent": "realtime_query",
            "slots": {
                "companies": [{"name": "比亚迪", "stock_code": "002594.SZ"}],
            },
        },
    ]

    for case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"查询: {case['query']}")
        result = create_task_plan(**case)

        print(f"\n📋 任务计划 ID: {result.plan.plan_id}")
        print(f"总任务数: {result.plan.total_tasks}")
        print(f"执行阶段: {result.plan.critical_path_length}")
        print(f"预估成本: {result.total_estimated_cost} tokens")
        print(f"风险评估: {result.risk_assessment}")

        print(f"\n📊 任务列表:")
        for task in result.plan.tasks:
            icon = "⏳" if task.status == TaskStatus.PENDING else "✅"
            deps = (
                f" (依赖: {', '.join(task.dependencies)})"
                if task.dependencies
                else ""
            )
            print(f"  {icon} [{task.task_id}] {task.description}{deps}")

        print(f"\n⚡ 执行顺序:")
        for i, group in enumerate(result.execution_order, 1):
            mark = " [可并行]" if len(group) > 1 else ""
            print(f"  第{i}阶段{mark}: {', '.join(group)}")
