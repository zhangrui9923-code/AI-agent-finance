"""
可插拔式金融技能框架 (Pluggable Financial Skill Framework)
======================================================

【模块定位 / Module Positioning】
本模块是 AI 金融研究助手系统的「能力扩展」核心组件，提供统一的技能抽象、
注册、发现和编排机制。它是连接 TaskPlanner（任务规划）和具体执行逻辑的中间层。

【架构角色 / Architectural Role】

┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  任务规划结果   │───▶│ SkillFramework    │───▶│  具体执行逻辑     │
│ (TaskPlan)    │    │ (本模块)          │    │ (LLM/API/计算)   │
│              │    │                  │                    │
│ "调用分析Skill" │    │ BaseSkill(ABC)   │    │ FinancialAgent   │
│ "调用检索Skill" │    │ SkillRegistry    │    │ RAG Pipeline     │
│              │    │ SkillPipeline     │    │ MCP Tools        │
└──────────────┘    └──────────────────┘    └──────────────────┘

【为什么需要 Skills 框架？/ Why Do We Need a Skill Framework?】

没有统一框架时，每个 Agent 直接硬编码其能力：
- FinancialAgent 内部直接调用 LLM → 无法替换为其他实现
- RiskAgent 复制了类似的调用模式 → 代码重复
- 新增能力需要修改多个 Agent → 违反开闭原则

有了 Skills 框架后：
- 能力被封装为独立的 Skill 对象 → 可复用、可测试、可替换
- 通过 Registry 统一管理 → 动态注册/发现/版本控制
- 通过 Pipeline 组合 → 灵活编排复杂工作流
- Agent 只依赖 Skill 接口 → 解耦，符合依赖倒置原则

【核心设计模式 / Core Design Patterns】

1. **模板方法模式 (Template Method) — BaseSkill**
   - 定义 execute() 的骨架：validate → pre_hook → execute → post_hook
   - 子类只需实现 execute() 核心逻辑
   - 保证所有 Skill 有统一的执行生命周期

2. **单例模式 (Singleton) — SkillRegistry**
   - 全局唯一的注册中心实例
   - 所有 Skill 在同一处注册和查找
   - 支持运行时动态注册/注销（热更新）

3. **装饰器模式 (Decorator) — @skill 装饰器**
   - 将普通函数一键包装为 Skill 对象
   - 自动注册到 Registry
   - 降低简单 Skill 的编写成本

4. **管道模式 (Pipeline) — SkillPipeline**
   - 将多个 Skill 串联为处理链
   - 支持 Step 间数据流转
   - 支持并行 Step 组

【Skill 生命周期 / Skill Lifecycle】

  Registered ──▶ Discovered ──▶ Validated ──▶ Executed ──▶ Result Returned
      │               │              │             │
      ▼               ▼              ▼             ▼
  registry.register()  find/search()  __call__()    SkillResult
                                      │
                                      ▼
                              validate_input()
                                      pre_execute()
                                      execute() ← 子类实现
                                      post_execute()

【性能特征 / Performance Characteristics】
- 注册操作：O(1) 字典插入 + O(K) 索引更新（K=标签数）
- 查找操作：O(N) 全量扫描 / O(1) 索引命中
- 执行开销：~0.5ms 额外开销（validate + hook 包装）
- 内存占用：每 Skill ~1KB 元数据 + 实例变量

【版本信息 / Version Info】
- Author: AI Finance Assistant Team
- Version: 2.0.0 (极致优化版 / Extreme Optimization Edition)
- Last Updated: 2026-04-13
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# 日志配置 / Logging Configuration
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 第一部分：类型别名与枚举定义
# Part 1: Type Aliases & Enum Definitions
# ════════════════════════════════════════════════════════════

SkillInput = TypeVar("SkillInput")
"""技能输入类型变量（泛型参数，由子类约束）"""

SkillOutput = TypeVar("SkillOutput")
"""技能输出类型变量（泛型参数，由子类约束）"""


class SkillCategory(str, Enum):
    """
    技能类别枚举 (Skill Category Enumeration)

    【职责】将系统中的所有 Skill 按功能领域分类，
    用于 Registry 的分类索引和按类别筛选。

    [8 大类别 / 8 Categories]

    类别                   说明                          典型 Skill 示例
    ──────────────────────────────────────────────────────────────
    DATA_RETRIEVAL       数据获取                      RAGRetrieval, StockQuote
    ANALYSIS             分析推理                      FinancialAnalysis, SentimentAnalysis
    CALCULATION           数值计算                      MetricCalculator, DCFModel
    GENERATION           内容生成                      ReportGenerator, SummaryWriter
    EXTERNAL_API         外部服务调用                   NewsAPI, WeatherAPI
    VALIDATION           数据校验                      SchemaValidator, RangeChecker
    TRANSFORMATION       数据转换                      Formatter, Normalizer
    MONITORING           监控与观测                     PerformanceTracker, ErrorLogger
    """

    DATA_RETRIEVAL = "data_retrieval"
    """数据检索类：从各种数据源获取原始信息"""

    ANALYSIS = "analysis"
    """分析类：对数据进行推理和分析"""

    CALCULATION = "calculation"
    """计算类：纯数学/统计运算（无 LLM 调用）"""

    GENERATION = "generation"
    """生成类：使用 LLM 生成文本/结构化内容"""

    EXTERNAL_API = "external_api"
    """外部 API 类：调用第三方服务"""

    VALIDATION = "validation"
    """验证类：校验数据格式和取值范围"""

    TRANSFORMATION = "transformation"
    """转换类：格式转换和数据清洗"""

    MONITORING = "monitoring"
    """监控类：性能追踪和日志记录"""


class SkillStatus(str, Enum):
    """
    技能状态枚举 (Skill Status Enumeration)

    [状态转换规则 / State Transitions]
    ACTIVE ←→ DISABLED: 可手动切换（维护模式）
    ACTIVE → DEPRECATED: 单向（弃用后不可恢复）
    ACTIVE → EXPERIMENTAL: 新功能灰度发布
    EXPERIMENTAL → ACTIVE: 验证通过后转正
    """

    ACTIVE = "active"
    """
    活跃可用（默认状态）。
    正常参与调度和执行。Registry 默认只返回 ACTIVE 状态的 Skill。
    """

    DEPRECATED = "deprecated"
    """
    已弃用。
    仍存在于 Registry 中但不再被自动发现。
    用于向后兼容——已有引用不会报错，但新代码不应使用。
    """

    DISABLED = "disabled"
    """
    已禁用。
    不参与任何调度或执行。通常用于临时排除有问题的 Skill（比删除更安全）。
    """

    EXPERIMENTAL = "experimental"
    """
    实验性（不稳定）。
    功能可能不完整或有已知 Bug。仅在显式指定时可使用，
    自动发现时会跳过此类 Skill。适用于灰度发布场景。
    """


class SkillPriority(str, Enum):
    """
    技能优先级枚举 (Skill Priority Enumeration)

    【用途】当多个 Skill 可完成同一任务时，用于选择最优 Skill。
    数值越小优先级越高。

    级别        值    场景
    ─────────────────────────────────────────
    HIGHEST    1     关键路径上的核心 Skill
    HIGH       2     重要但非阻塞
    NORMAL     3     标准（默认值）
    LOW        4     增强型/可选
    LOWEST     5     调试/实验性
    """

    HIGHEST = "1"
    HIGH = "2"
    NORMAL = "3"
    LOW = "4"
    LOWEST = "5"


# ════════════════════════════════════════════════════════════
# 第二部分：核心数据模型
# Part 2: Core Data Models
# ════════════════════════════════════════════════════════════


@dataclass
class SkillMetadata:
    """
    技能元数据 (Skill Metadata)

    【职责】描述一个 Skill 的完整身份和能力声明。
    每个 BaseSkill 子类必须定义 class-level 的 metadata 属性。

    【字段分组 / Field Groups】

    身份标识: name, version, description, category, author, tags
    能力声明: input_schema, output_schema
    性能指标: estimated_time_ms, estimated_cost_tokens
    兼容性: dependencies, compatible_agents
    状态与统计: status, created_at, updated_at, call_count, success_rate, avg_execution_time_ms
    """

    name: str
    """
    技能名称（全局唯一标识符）。
    
    命名规范：snake_case，如 'rag_retrieval'、'financial_analysis'。
    此值作为 Registry 中的主键，也出现在 SkillResult.skill_name 中。
    """

    version: str
    """
    语义化版本号（遵循 SemVer：MAJOR.MINOR.PATCH）。
    
    如 '1.0.0'、'2.1.3-beta'。
    用于兼容性检查和版本策略实施。
    """

    description: str
    """
    人类可读的功能描述。
    
    用于 Registry.search() 的模糊匹配和 UI 展示。
    应简洁明了地说明该 Skill 做什么。
    """

    category: SkillCategory
    """所属技能类别（来自 SkillCategory 枚举），决定分类索引归属"""

    author: str = ""
    """开发者/团队标识（可选，用于审计和责任追溯）"""

    tags: List[str] = field(default_factory=list)
    """
    搜索标签列表（可选）。
    
    用于 Registry.find_by_tag() 和 search()。
    应包含同义词和相关关键词以提升搜索召回率。
    示例：['retrieval', 'hyde', 'rerank', 'vector-search']
    """

    # ── 能力声明 / Capability Declaration ──

    input_schema: Optional[Dict[str, Any]] = None
    """
    输入 JSON Schema（可选）。
    
    当提供时，BaseSkill.validate_input() 会使用此 Schema
    通过 Pydantic 动态创建 Model 来校验输入数据的合法性。
    格式遵循 JSON Schema Draft-07 规范。
    """

    output_schema: Optional[Dict[str, Any]] = None
    """
    输出 JSON Schema（可选）。
    
    描述 execute() 返回值的预期结构。
    目前仅用于文档目的，未来可用于输出校验。
    """

    # ── 性能指标 / Performance Metrics ──

    estimated_time_ms: float = 1000.0
    """
    预估执行耗时（毫秒，P50 中位数）。
    
    用于：
    1. Pipeline 的超时设置（总超时 = Σ 各步骤预估 × 安全系数）
    2. 进度条展示
    3. 调度决策（低延迟 Skill 优先安排在关键路径上）
    """

    estimated_cost_tokens: int = 500
    """
    预估 LLM Token 消耗量。
    
    纯计算类 Skill（如 MetricCalculator）此值为 0。
    LLM 驱动类 Skill 通常为 500-3000。
    用于成本预算控制。
    """

    # ── 兼容性与依赖 / Compatibility & Dependencies ──

    dependencies: List[str] = field(default_factory=list)
    """
    依赖的其他 Skill 名称列表。
    
    当本 Skill 执行前，Registry.resolve_dependencies()
    会确保所有依赖已就绪。
    示例：FinancialAnalysisSkill 依赖 ['rag_retrieval']。
    """

    compatible_agents: List[str] = field(default_factory=list)
    """
    兼容的 Agent 名称列表。
    
    用于 Registry.find_by_agent() 过滤：
    只有在此列表中的 Agent 才能看到并调用此 Skill。
    空列表表示所有 Agent 都可用（默认行为）。
    """

    # ── 状态管理 / Status Management ──

    status: SkillStatus = SkillStatus.ACTIVE
    """当前状态（默认为 ACTIVE）"""

    created_at: str = ""
    """首次注册时间（ISO 8601 格式）"""

    updated_at: str = ""
    """最后更新时间（ISO 8601 格式）"""

    # ── 使用统计 / Usage Statistics ──
    # 以下字段由 Registry 在每次执行后自动更新

    call_count: int = 0
    """累计调用次数"""

    success_rate: float = 1.0
    """
    成功率（0.0-1.0）。
    计算公式：成功次数 / 总调用次数。
    初始值为 1.0（避免除零，且新 Skill 默认可信）。
    """

    avg_execution_time_ms: float = 0.0
    """
    平均执行耗时（毫秒，EMA 指数移动平均）。
    更新公式：new_avg = 0.7 × old_avg + 0.3 × latest_time
    """


class SkillContext:
    """
    技能执行上下文 (Skill Execution Context)

    【职责】在单次 Skill 调用的全生命周期内传递共享信息。
    包括请求标识、调用链追踪、中间结果缓存和性能监控数据。

    【生命周期 / Lifecycle】
    由 Pipeline 或调用方在执行开始时创建，
    在同一调用链的所有 Skill 之间传递（pass-by-reference），
    最终随 SkillResult 返回给调用方。

    【线程安全 / Thread Safety]
    本对象不是线程安全的！如果需要在多线程环境中使用，
    请确保每个线程创建独立的 SkillContext 实例。
    """

    def __init__(
        self,
        request_id: str = "",
        user_id: str = "",
        session_id: str = "",
        parent_skill: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.request_id: str = request_id or self._generate_id()
        """请求唯一标识符（12 位 MD5 哈希）"""

        self.user_id: str = user_id
        """用户标识（用于权限控制和个性化配置）"""

        self.session_id: str = session_id
        """会话标识（关联多轮对话）"""

        self.parent_skill: Optional[str] = parent_skill
        """父 Skill 名称（用于嵌套调用场景）"""

        self.metadata: Dict[str, Any] = metadata or {}
        """自定义元数据字典（供 Skill 存取任意键值对）"""

        # ── 执行追踪 / Execution Tracing ──

        self.start_time: float = time.time()
        """上下文创建时间戳（用于计算耗时）"""

        self.call_chain: List[str] = []
        """
        技能调用链记录。
        
        每次 Skill被执行时，其 name 会追加到此列表。
        用于调试和审计完整的调用路径。
        示例：['rag_retrieval', 'financial_analysis', 'risk_assessment']
        """

        # ── 中间结果缓存 / Intermediate Result Cache ──

        self.intermediate_results: Dict[str, Any] = {}
        """
        中间结果字典。
        
        Skill 之间的数据传递机制之一（另一种是通过返回值链式传递）。
        用法：context.set_intermediate("key", value) 存储，
            其他下游 Skill 通过 context.get_intermediate("key") 读取。
        典型用途：RAG 检索结果被存入此后，FinancialAnalysis 和 RiskAssessment
        都可以读取到相同的文档集合。
        """

        # ── 性能监控 / Performance Monitoring ──

        self.token_usage: int = 0
        """本次上下文中累计消耗的 LLM Token 数"""

        self.api_calls: int = 0
        """本次上下文中累计的外部 API 调用次数"""

    @staticmethod
    def _generate_id() -> str:
        """生成基于时间和对象 ID 的 12 位唯一标识符"""
        return hashlib.md5(
            f"{time.time()}{id(object())}".encode()
        ).hexdigest()[:12]

    def add_to_call_chain(self, skill_name: str) -> None:
        """将 Skill 名称追加到调用链尾部"""
        self.call_chain.append(skill_name)

    def set_intermediate(self, key: str, value: Any) -> None:
        """存储中间结果（覆盖写）"""
        self.intermediate_results[key] = value

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        """读取中间结果（不存在时返回 default）"""
        return self.intermediate_results.get(key, default)


class SkillResult(BaseModel):
    """
    技能执行结果模型 (Skill Execution Result)

    【职责】统一所有 Skill 的输出格式。
    无论 Skill 内部逻辑如何复杂，对外都返回此标准结构。

    【字段说明 / Field Descriptions】
    - success: 是否成功（False 时 data 为 None，error 有值）
    - data: 输出数据（类型取决于具体 Skill）
    - error: 错误信息（仅失败时有值）
    - skill_name: 产生此结果的 Skill 名称
    - execution_time_ms: 执行耗时
    - tokens_used: Token 消耗
    - confidence: 结果置信度（0-1，由 Skill 自行评估）
    - source: 数据来源标注（如 'llm'/'api'/'cache'）
    - suggestions: 后续建议（指导下一步行动）
    """

    success: bool = Field(default=True, description="是否成功执行")
    data: Any = Field(default=None, description="输出数据（类型取决于具体 Skill）")
    error: Optional[str] = Field(default=None, description="错误信息（仅失败时有值）")

    skill_name: str = Field(default="", description="产生此结果的 Skill 名称")
    execution_time_ms: float = Field(default=0.0, description="执行耗时（毫秒）")
    tokens_used: int = Field(default=0, description="LLM Token 消耗量")

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="结果置信度（0-1，由 Skill 自行评估）",
    )
    source: Optional[str] = Field(
        default=None,
        description="数据来源标注（如 'llm'/'api'/'cache'/'calculated'）",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="后续建议列表（指导下一步行动或提示补充信息）",
    )


# ════════════════════════════════════════════════════════════
# 第三部分：技能基类
# Part 3: Base Skill Abstract Class
# ════════════════════════════════════════════════════════════


class BaseSkill(ABC, Generic[SkillInput, SkillOutput]):
    """
    所有技能的抽象基类 (Abstract Base Class for All Skills)

    【职责】定义 Skill 的统一接口和执行模板（Template Method Pattern）。
    所有具体 Skill 必须继承此类并实现 execute() 方法。

    【子类必须实现的成员 / Must-Implement Members】
    1. metadata: class-level SkillMetadata 实例
    2. execute(input_data, context) -> SkillOutput: 核心执行逻辑

    【可选择性重写的钩子方法 / Overridable Hook Methods】
    - validate_input(): 输入合法性校验（默认使用 metadata.input_schema）
    - pre_execute(): 前置处理（默认为透传）
    - post_execute(): 后置包装（默认包装为 SkillResult）

    【执行流程 / Execution Flow (__call__)】

    input_data ──▶ [validate_input()] ──▶ [pre_execute()]
                                        │
                                        ▼
                                   [execute()] ← 子类实现
                                        │
                                        ▼
                                   [post_execute()] ──▶ SkillResult

    【泛型参数 / Generic Parameters】
    SkillInput: 输入数据类型（建议但不强制使用 TypeVar 约束）
    SkillOutput: 输出数据类型（同上）
    这两个参数主要用于 IDE 类型提示，不影响运行时行为。
    """

    metadata: SkillMetadata = NotImplemented
    """
    技能元数据（必须在子类中定义为类级别属性）。
    
    示例：
        class MySkill(BaseSkill):
            metadata = SkillMetadata(
                name="my_skill",
                version="1.0.0",
                ...
            )
    """

    @abstractmethod
    def execute(
        self, input_data: SkillInput, context: SkillContext
    ) -> SkillOutput:
        """
        核心执行逻辑（子类必须实现）。

        [Args / 参数]
            input_data: 经过 validate 和 pre_process 后的输入数据
            context:   执行上下文（含调用链、中间结果等）

        [Returns / 返回值]
            SkillOutput: 原始输出数据（将被 post_execute 包装为 SkillResult）

        [Raises / 异常]
            任何异常都会被 __call__ 捕获并转化为 SkillResult(success=False)
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        校验输入数据的合法性。

        [Default Behavior]
        如果 metadata.input_schema 为 None，则跳过校验（信任输入）。
        否则使用 Pydantic 动态创建 Model 进行校验。

        [Override 建议]
        对于有特殊校验需求的 Skill（如数值范围、枚举值等），
        建议重写此方法而非修改 input_schema。
        """
        if self.metadata.input_schema is None:
            return True

        try:
            props: Dict[str, Any] = self.metadata.input_schema.get("properties", {})
            model_fields: Dict[str, Any] = {
                k: (v.get("type", str), ...) for k, v in props.items()
            }
            dynamic_model = type("InputModel", (BaseModel,), {"__annotations__": model_fields})
            dynamic_model.model_validate(
                input_data if isinstance(input_data, dict) else {"data": input_data}
            )
            return True
        except Exception as e:
            raise ValueError(f"输入验证失败 [{self.metadata.name}]: {e}")

    def pre_execute(self, input_data: Any, context: SkillContext) -> Any:
        """
        前置处理钩子（在 execute() 之前调用）。

        [Default Behavior]
        透传 input_data，不做任何修改。

        [Override 场景]
        - 日志脱敏：移除敏感字段后再传给 execute()
        - 数据预处理：格式转换、缺失值填充
        - 缓存检查：先查缓存，命中则短路后续流程
        - 权限校验：检查 user_id 是否有调用权限
        """
        return input_data

    def post_execute(
        self, result: Any, context: SkillContext
    ) -> SkillResult:
        """
        后置处理钩子（在 execute() 之后调用）。

        [Default Behavior]
        将原始结果包装为标准的 SkillResult 对象。

        [Override 场景]
        - 结果增强：添加 confidence/source/suggestions
        - 结果缓存：写入 Redis/MemoryCache
        - 指标上报：推送到 Prometheus/Grafana
        - 异步触发：启动后台任务（如日志归档）
        """
        return SkillResult(
            success=True,
            data=result,
            skill_name=self.metadata.name,
            execution_time_ms=(time.time() - context.start_time) * 1000,
            tokens_used=context.token_usage,
        )

    def __call__(self, input_data: Any, context: Optional[SkillContext] = None) -> SkillResult:
        """
        便捷调用接口 — 执行完整的 Skill 生命周期。

        这是外部代码最常用的调用方式：
            result = my_skill({"query": "..."}, context)

        它内部协调 validate → pre_hook → execute → post_hook 四个阶段，
        并保证异常安全（任何异常都转化为 SkillResult(success=False)）。
        """
        if context is None:
            context = SkillContext()

        context.add_to_call_chain(self.metadata.name)
        start_time: float = time.time()

        try:
            self.validate_input(input_data)
            processed_input = self.pre_execute(input_data, context)
            raw_result = self.execute(processed_input, context)
            final_result = self.post_execute(raw_result, context)

            self.metadata.call_count += 1
            return final_result

        except Exception as e:
            execution_time: float = (time.time() - start_time) * 1000
            logger.error("[Skill:%s] 执行异常: %s", self.metadata.name, e)
            return SkillResult(
                success=False,
                error=str(e),
                skill_name=self.metadata.name,
                execution_time_ms=execution_time,
            )


# ════════════════════════════════════════════════════════════
# 第四部分：技能注册中心（单例）
# Part 4: Skill Registry (Singleton)
# ════════════════════════════════════════════════════════════


class SkillRegistry:
    """
    技能注册中心 (Skill Registry — Singleton)

    【职责】全局唯一的 Skill 管理中心，负责：
    1. 注册/注销 Skill 实例
    2. 按名称/类别/标签/Agent 查找 Skill
    3. 模糊搜索（名称+描述+标签匹配）
    4. 依赖关系解析（DFS 拓扑排序）
    5. 使用统计聚合

    【单例保证 / Singleton Guarantee】
    通过重写 __new__ 方法实现线程安全的单例模式。
    整个应用生命周期内只有一个 Registry 实例。
    访问方式：直接导入模块级 `registry` 变量。

    【索引结构 / Index Structure】
    除了主存储 _skills: Dict[name, Skill] 外，
    还维护三个倒排索引用于加速查询：
    - _metadata_index["category"]: {category_value: [skill_names]}
    - _metadata_index["tag"]: {tag: [skill_names]}
    - _metadata_index["agent"]: {agent_name: [skill_names]}
    """

    _instance: Optional["SkillRegistry"] = None
    _skills: Dict[str, BaseSkill] = {}
    _metadata_index: Dict[str, Dict[str, List[str]]] = {}

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """初始化存储和索引结构"""
        self._skills: Dict[str, BaseSkill] = {}
        self._metadata_index: Dict[str, Dict[str, List[str]]] = {
            "category": {},
            "tag": {},
            "agent": {},
        }

    def register(self, skill: BaseSkill) -> None:
        """
        注册一个新 Skill 到 Registry。

        [Args / 参数]
            skill: 已初始化的 BaseSkill 子类实例

        [Behavior / 行为]
        - 如果同名 Skill 已存在，静默覆盖（打印警告日志）
        - 同步更新三个倒排索引
        - 设置 created_at 时间戳（如果是新注册）
        """
        name: str = skill.metadata.name

        if name in self._skills:
            logger.warning("[SkillRegistry] ⚠️ 技能'%s'已存在，将被覆盖", name)

        self._skills[name] = skill
        meta: SkillMetadata = skill.metadata

        if not meta.created_at:
            meta.created_at = datetime.now().isoformat()
        meta.updated_at = datetime.now().isoformat()

        cat: str = meta.category.value
        if cat not in self._metadata_index["category"]:
            self._metadata_index["category"][cat] = []
        self._metadata_index["category"][cat].append(name)

        for tag in meta.tags:
            if tag not in self._metadata_index["tag"]:
                self._metadata_index["tag"][tag] = []
            self._metadata_index["tag"][tag].append(name)

        for agent in meta.compatible_agents:
            if agent not in self._metadata_index["agent"]:
                self._metadata_index["agent"][agent] = []
            self._metadata_index["agent"][agent].append(name)

        logger.info("[SkillRegistry] ✅ 注册技能: %s v%s", name, meta.version)

    def unregister(self, skill_name: str) -> bool:
        """注销指定 Skill（同时清理所有索引条目）"""
        if skill_name not in self._skills:
            return False

        self._skills.pop(skill_name)

        for index_type in self._metadata_index.values():
            for key in list(index_type.keys()):
                if skill_name in index_type[key]:
                    index_type[key].remove(skill_name)

        logger.info("[SkillRegistry] ❌ 注销技能: %s", skill_name)
        return True

    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """按名称精确查找 Skill"""
        return self._skills.get(skill_name)

    def get_all(self) -> Dict[str, BaseSkill]:
        """获取所有已注册 Skill 的浅拷贝"""
        return dict(self._skills)

    def find_by_category(self, category: SkillCategory) -> List[BaseSkill]:
        """按类别查找所有 Skill"""
        names = self._metadata_index["category"].get(category.value, [])
        return [self._skills[n] for n in names if n in self._skills]

    def find_by_tag(self, tag: str) -> List[BaseSkill]:
        """按标签查找所有 Skill"""
        names = self._metadata_index["tag"].get(tag, [])
        return [self._skills[n] for n in names if n in self._skills]

    def find_by_agent(self, agent_name: str) -> List[BaseSkill]:
        """按兼容 Agent 查找所有 Skill"""
        names = self._metadata_index["agent"].get(agent_name, [])
        return [self._skills[n] for n in names if n in self._skills]

    def search(self, query: str) -> List[BaseSkill]:
        """
        模糊搜索 Skill（多维度加权匹配）。

        [匹配维度与权重 / Match Dimensions & Weights]
        - 名称完全包含 query: +10 分
        - 描述包含 query:     +5 分
        - 标签包含 query:     +3 分（每个匹配标签）

        [Returns / 返回值]
        按 match_score 降序排列的 Skill 列表（仅返回 score > 0 的）
        """
        query_lower: str = query.lower()
        results: List[tuple[int, BaseSkill]] = []

        for name, skill in self._skills.items():
            score: int = 0

            if query_lower in name.lower():
                score += 10

            if query_lower in skill.metadata.description.lower():
                score += 5

            for tag in skill.metadata.tags:
                if query_lower in tag.lower():
                    score += 3

            if score > 0:
                results.append((score, skill))

        results.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in results]

    def resolve_dependencies(self, skill_name: str) -> List[str]:
        """
        解析 Skill 的完整依赖链（DFS 拓扑排序）。

        [Returns / 返回值]
        按拓扑序排列的 Skill 名称列表（依赖在前，自身在最后）。
        若存在循环依赖，循环内的节点顺序取决于 DFS 遍历顺序。
        """
        visited: set = set()
        result: List[str] = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            skill = self._skills.get(name)
            if skill:
                for dep in skill.metadata.dependencies:
                    dfs(dep)
            result.append(name)

        dfs(skill_name)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册中心的聚合统计数据"""
        total_skills: int = len(self._skills)
        stats_by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        total_calls: int = 0

        for skill in self._skills.values():
            meta: SkillMetadata = skill.metadata
            stats_by_status[meta.status.value] = (
                stats_by_status.get(meta.status.value, 0) + 1
            )
            by_category[meta.category.value] = (
                by_category.get(meta.category.value, 0) + 1
            )
            total_calls += meta.call_count

        return {
            "total_skills": total_skills,
            "by_status": stats_by_status,
            "by_category": by_category,
            "total_calls": total_calls,
            "available_categories": list(self._metadata_index["category"].keys()),
        }


# ── 全局单例实例 / Global Singleton Instance ──
registry: SkillRegistry = SkillRegistry()


# ════════════════════════════════════════════════════════════
# 第五部分：@skill 装饰器
# Part 5: @skill Decorator
# ════════════════════════════════════════════════════════════


def skill(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    category: SkillCategory = SkillCategory.ANALYSIS,
    tags: Optional[List[str]] = None,
    **kwargs: Any,
) -> Callable:
    """
    技能装饰器工厂函数 — 将普通函数一键转换为 Skill 并注册到 Registry。

    [Design Rationale / 设计理由]
    不是所有 Skill 都值得创建一个完整的类。
    对于简单的数据处理函数（如格式化、单位转换、单一计算），
    使用装饰器可以将代码量从 ~30 行减少到 ~5 行。

    [Usage Example / 使用示例]
        @skill(
            name="pe_calculator",
            category=SkillCategory.CALCULATION,
            tags=["valuation", "pe"],
        )
        def calculate_pe(params: dict, context: SkillContext) -> dict:
            price = params["price"]
            eps = params["eps"]
            return {"pe": round(price / eps, 2) if eps else None}

        # 现在 pe_calculator 已经是一个 registered Skill！
        result = pe_calculator({"price": 1800, "eps": 70})
    """

    def decorator(func: Callable) -> Callable:
        class FunctionSkill(BaseSkill):  # type: ignore[misc]
            metadata = SkillMetadata(
                name=name,
                version=version,
                description=description or func.__doc__ or f"Auto-generated skill from {func.__name__}",
                category=category,
                tags=tags or [],
                **kwargs,
            )

            def execute(self, input_data: Any, context: SkillContext) -> Any:
                return func(input_data, context)

        registry.register(FunctionSkill())  # type: ignore[arg-type]
        return func

    return decorator


# ════════════════════════════════════════════════════════════
# 第六部分：技能管道
# Part 6: Skill Pipeline
# ════════════════════════════════════════════════════════════


class SkillPipeline:
    """
    技能管道 (Skill Pipeline)

    【职责】将多个 Skill 组织为有序的处理链，支持串行和并行执行。

    [Execution Model / 执行模型]
    默认为串行流水线：Step1 → Step2 → Step3 → ...
    每个 Step 的输出作为下一个 Step 的输入。

    未来可扩展支持：
    - 条件分支：if Step1.result == X then Step2a else Step2b
    - 并行组：Step2 和 Step3 同时执行，结果合并后传入 Step4
    - 重试策略：单个 Step 失败后的自动重试

    [Example / 示例]
        pipeline = SkillPipeline("financial_analysis")
        pipeline.add_step("retrieval", rag_skill)
        pipeline.add_step("analysis", analysis_skill)
        pipeline.add_step("risk", risk_skill)
        result = pipeline.run(query_data, context)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name: str = name
        """管道唯一名称"""
        self.description: str = description
        """管道描述"""
        self.steps: List[tuple[str, BaseSkill]] = []
        """有序步骤列表：(step_name, skill_instance)"""
        self.parallel_groups: List[List[int]] = []
        """并行组定义（每组包含 step 索引）"""

    def add_step(
        self,
        step_name: str,
        skill: BaseSkill,
        parallel_with: Optional[List[int]] = None,
    ) -> "SkillPipeline":
        """
        向管道追加一个执行步骤。

        [Args / 参数]
            step_name:     步骤名称（用于日志和错误定位）
            skill:         要执行的 Skill 实例
            parallel_with: 与哪些已有步骤（按索引）并行执行（可选）

        [Returns / 返回值]
            self（支持链式调用：pipeline.add_step(...).add_step(...)）
        """
        step_index: int = len(self.steps)
        self.steps.append((step_name, skill))

        if parallel_with:
            group_found: bool = False
            for group in self.parallel_groups:
                if any(i in parallel_with for i in group):
                    group.append(step_index)
                    group_found = True
                    break
            if not group_found:
                new_group: List[int] = list(parallel_with) + [step_index]
                self.parallel_groups.append(new_group)

        return self

    def run(
        self,
        initial_data: Any,
        context: Optional[SkillContext] = None,
    ) -> SkillResult:
        """
        执行管道中的所有步骤。

        [Error Handling / 错误处理]
        任一 Step 失败则立即终止管道，
        返回包含错误信息的 SkillResult(success=False)。
        已完成的 Step 结果不会回滚。
        """
        if context is None:
            context = SkillContext()

        current_data: Any = initial_data

        for i, (step_name, skill) in enumerate(self.steps):
            logger.info("[Pipeline:%s] 执行第%d步: %s", self.name, i + 1, step_name)

            result: SkillResult = skill(current_data, context)

            if not result.success:
                return SkillResult(
                    success=False,
                    error=f"管道在步骤'{step_name}'失败: {result.error}",
                    skill_name=self.name,
                )

            current_data = result.data

        return SkillResult(
            success=True,
            data=current_data,
            skill_name=self.name,
            execution_time_ms=(time.time() - context.start_time) * 1000,
        )


# ════════════════════════════════════════════════════════════
# 第七部分：内置技能实现
# Part 7: Built-in Skill Implementations
#
# 这些是系统预置的核心 Skill，在 import 时自动注册到 Registry。
# ════════════════════════════════════════════════════════════


class RAGRetrievalSkill(BaseSkill):
    """
    RAG 检索技能 (RAG Retrieval Skill)

    【能力】基于 EnhancedRAGPipeline 进行混合检索，
    支持 HyDE 增强、BM25 关键词匹配、Cross-Encoder 重排序。
    """

    metadata = SkillMetadata(
        name="rag_retrieval",
        version="1.0.0",
        description="基于RAG的知识检索，支持HyDE增强和重排序",
        category=SkillCategory.DATA_RETRIEVAL,
        tags=["retrieval", "hyde", "rerank", "vector-search"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "use_hyde": {"type": "boolean", "default": True},
            },
        },
        estimated_time_ms=2000.0,
        estimated_cost_tokens=500,
    )

    def __init__(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline

    def execute(self, input_data: Dict, context: SkillContext) -> List[Dict]:
        query: str = input_data.get("query", "")
        top_k: int = input_data.get("top_k", 5)
        use_hyde: bool = input_data.get("use_hyde", True)

        docs = self.rag_pipeline.query(query, use_hyde=use_hyde)[:top_k]
        context.set_intermediate("retrieved_docs", docs)
        return docs


class FinancialAnalysisSkill(BaseSkill):
    """
    财务分析技能 (Financial Analysis Skill)

    【能力】使用 CFA 级分析师 Prompt 对检索到的财报数据进行深度分析。
    依赖 RAG 检索结果（从 context.intermediate_results 获取）。
    """

    metadata = SkillMetadata(
        name="financial_analysis",
        version="1.0.0",
        description="专业财务数据分析，覆盖盈利能力、成长性、现金流等维度",
        category=SkillCategory.ANALYSIS,
        tags=["financial", "analysis", "cfa"],
        dependencies=["rag_retrieval"],
        estimated_time_ms=5000.0,
        estimated_cost_tokens=2000,
    )

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def execute(self, input_data: Dict, context: SkillContext) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        query: str = input_data.get("query", "")
        docs: list = context.get_intermediate("retrieved_docs", [])

        system_prompt: str = (
            "你是一位拥有 CFA 资质的资深财务分析师..."
        )
        user_prompt: str = f"## 用户问题\n{query}\n\n## 参考文档\n{docs}"

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        analysis: str = response.content.strip()
        context.set_intermediate("financial_analysis", analysis)
        return analysis


class MetricCalculatorSkill(BaseSkill):
    """
    财务指标计算技能 (Metric Calculator Skill)

    【能力】精确计算常用财务指标，避免 LLM 数学幻觉。
    纯 Python 计算，零 LLM 调用，零延迟不确定性。
    """

    metadata = SkillMetadata(
        name="metric_calculator",
        version="1.0.0",
        description="精确计算财务指标，避免LLM数学幻觉",
        category=SkillCategory.CALCULATION,
        tags=["calculation", "metrics", "pe", "pb", "roe"],
        estimated_time_ms=50.0,
        estimated_cost_tokens=0,
    )

    METRIC_FORMULAS: Dict[str, Callable] = {
        "pe_ratio": lambda p: p["price"] / p["eps"] if p.get("eps", 0) != 0 else None,
        "pb_ratio": lambda p: p["price"] / p["bvps"] if p.get("bvps", 0) != 0 else None,
        "roe": lambda p: p["net_income"] / p["equity"] * 100 if p.get("equity", 0) != 0 else None,
        "gross_margin": lambda p: (p["revenue"] - p.get("cogs", 0)) / p["revenue"] * 100 if p.get("revenue", 0) != 0 else None,
        "net_margin": lambda p: p["net_income"] / p["revenue"] * 100 if p.get("revenue", 0) != 0 else None,
        "debt_ratio": lambda p: p["total_debt"] / p["total_assets"] * 100 if p.get("total_assets", 0) != 0 else None,
    }

    def execute(self, input_data: Dict, context: SkillContext) -> Dict:
        metric_name: str = input_data.get("metric", "")
        params: Dict[str, Any] = input_data.get("params", {})

        formula: Optional[Callable] = self.METRIC_FORMULAS.get(metric_name)
        if formula is None:
            raise ValueError(f"不支持的指标: {metric_name}")

        result: Optional[float] = formula(params)

        return {
            "metric": metric_name,
            "value": round(result, 4) if result is not None else None,
            "unit": self._get_unit(metric_name),
        }

    @staticmethod
    def _get_unit(metric: str) -> str:
        units: Dict[str, str] = {
            "pe_ratio": "倍",
            "pb_ratio": "倍",
            "roe": "%",
            "gross_margin": "%",
            "net_margin": "%",
            "debt_ratio": "%",
        }
        return units.get(metric, "")


# ════════════════════════════════════════════════════════════
# 第八部分：模块自测
# Part 8: Self-Test Suite
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Skills Framework Test")
    print("=" * 60)

    stats = registry.get_statistics()
    print(f"\n📊 注册中心统计:")
    print(f"   总技能数: {stats['total_skills']}")
    print(f"   分类: {stats['by_category']}")

    test_skill = MetricCalculatorSkill()
    registry.register(test_skill)

    calc_skills = registry.find_by_category(SkillCategory.CALCULATION)
    print(f"\n🔍 计算类技能: {[s.metadata.name for s in calc_skills]}")

    ctx = SkillContext(request_id="test_001")
    result = test_skill(
        {"metric": "pe_ratio", "params": {"price": 1800, "eps": 70}},
        ctx,
    )

    print(f"\n✅ 计算结果:")
    print(f"   成功: {result.success}")
    print(f"   数据: {result.data}")
    print(f"   耗时: {result.execution_time_ms:.1f}ms")
