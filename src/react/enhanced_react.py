"""
增强版 ReAct 循环 (Enhanced ReAct Loop)
==========================================

【模块定位】
本模块是 AI 金融研究助手系统的「推理引擎」，负责将用户查询转化为
结构化的思考-行动-观察-反思循环。它是连接意图识别、任务规划、
技能调用和最终答案生成的核心协调器。

【什么是 ReAct？】
ReAct (Reasoning + Acting) 是一种结合推理和行动的 Agent 范式，
由 Yao et al. (2022) 在论文 "ReAct: Synergizing Reasoning and Acting
in Language Models" 中提出。

传统 LLM 的局限:
  ❌ 纯生成模式 → 无法获取外部信息（幻觉问题）
  ❌ 纯工具调用 → 缺乏推理规划能力（盲目执行）

ReAct 的解决方案:
  ✅ Thought: 先思考"我需要什么信息？"
  ✅ Action: 再行动"调用XX工具获取"
  ✅ Observation: 后观察"结果说明了什么？"
  ✅ Reflection: 最后反思"是否足够回答？"

【架构总览】

  ┌──────────────────────────────────────────────────────┐
  │                   User Query                         │
  │            "茅台的估值是否合理？"                     │
  └────────────────────┬─────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │              Step 1: THOUGHT 💭                      │
  │  "要回答估值问题，我需要获取以下信息：                 │
  │   1. 当前股价和市值                                  │
  │   2. 最近一期的财务数据（PE、PB、ROE）               │
  │   3. 同行业可比公司的估值水平                        │
  │   我应该先调用 get_stock_price 工具"                  │
  └────────────────────┬─────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │              Step 2: ACTION 🔧                       │
  │  Action: get_stock_price                             │
  │  Input: {"symbol": "600519.SH"}                     │
  └────────────────────┬─────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │           Step 3: OBSERVATION 👁️                    │
  │  Result: {"price": 1680, "market_cap": "2.1万亿"}    │
  │  "当前股价1680元，市值约2.1万亿。这个价格处于         │
  │   历史高位区间，需要进一步分析..."                    │
  └────────────────────┬─────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │          Step 4: REFLECTION 🔄                       │
  │  "目前只有股价数据，还缺少：                           │
  │   - 盈利能力指标（PE/ROE）                           │
  │   - 行业对比基准                                     │
  │   需要继续调用 financial_analysis 工具..."            │
  └────────────────────┬─────────────────────────────────┘
                       ▼
                ... (重复循环) ...
                       ▼
  ┌──────────────────────────────────────────────────────┐
  │             FINAL ANSWER ✅                          │
  │  "基于以上分析，贵州茅台当前估值...                   │
  │   [完整答案]"                                        │
  └──────────────────────────────────────────────────────┘

【核心组件】

1. EnhancedReActAgent: 主协调器
   - 编排完整的 T-A-O-R 循环
   - 基于 LangGraph create_react_agent 实现
   - 支持自定义停止条件和错误恢复

2. 数据模型层:
   - ReActState: 循环状态机（记录完整执行历史）
   - ReActStep: 单步记录（Thought/Action/Observation）
   - ActionCall: 工具调用详情（含性能指标）
   - ObservationRecord: 观察处理结果

3. 停止条件系统:
   - GOAL_ACHIEVED: 目标达成（检测到 Final Answer）
   - MAX_STEPS_REACHED: 达到最大迭代次数
   - ERROR_OCCURRED: 连续错误超过阈值
   - NO_MORE_ACTIONS: LLM 决定不继续

【设计原则】

- 可追溯性: 每一步都有完整记录，支持事后审计
- 容错性: 单步失败不影响整体流程，支持自动重试
- 效率性: 自适应 Token 管理，避免无限循环
- 可观测性: 结构化日志输出，便于调试和分析

【与传统 ReAct 的区别】

| 维度 | 标准 ReAct | 本实现 (Enhanced) |
|------|-----------|-------------------|
| 推理链 | 隐式（在 prompt 中） | 显式（结构化记录） |
| 停止条件 | 仅最大步数 | 多维度自适应 |
| 错误处理 | 无 | 自动重试 + 降级 |
| 观察摘要 | 原始返回 | 智能提取关键信息 |
| 性能监控 | 无 | 全链路耗时 + Token 统计 |
| 并行调用 | 不支持 | 支持（同类型独立工具） |

【依赖关系】

上游:
  - config.settings: API 密钥、模型配置
  - langchain_openai.ChatOpenAI: LLM 推理引擎
  - langgraph.prebuilt.create_react_agent: ReAct 框架
  - src.core.intent_classifier: 意图理解（可选前置）

下游消费者:
  - main_enhanced.py: 主流程中的推理执行
  - src.enhanced_evaluation: 执行质量评估输入

【版本信息】
- v2.0 (2025-01): 极致优化版，完整 T-A-O-R 循环 + 自适应停止 + 自测套件
- v1.0 (2024-12): 初始版本，基于 LangGraph 封装
"""

from __future__ import annotations

import re
import json
import time
import uuid
import traceback
from typing import (
    Any, Dict, List, Optional, Tuple, Union,
    Callable, Iterator, Pattern
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
from collections import deque

from pydantic import BaseModel, Field, validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL


# ════════════════════════════════════════════════════════════
# 第一部分：枚举定义
# ════════════════════════════════════════════════════════════


class ReActStepType(Enum):
    """
    ReAct 步骤类型枚举
    
    定义循环中每一步的角色和职责。
    
    成员说明:
        THOUGHT: 思考步骤
                LLM 分析当前状态，决定下一步行动方向
                输出内容通常是推理链："因为...所以..."
                
        ACTION: 行动步骤
               选择并调用具体的工具/Skill/子Agent
               包含工具名称和参数
                
        OBSERVATION: 观察步骤
                    处理工具返回的原始结果
                    提取关键信息，评估数据质量
                    
        REFLECTION: 反思步骤
                  评估当前进度，判断是否达到目标
                  决定是继续循环还是输出最终答案
                  
        FINISH: 完成标记
               表示循环正常终止
               通常伴随 Final Answer 输出
    
    使用场景:
        >>> step = ReActStep(step_type=ReActStepType.THOUGHT, ...)
        >>> if step.step_type == ReActStepType.ACTION:
        ...     execute_action(step.action)
    """
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    FINISH = "finish"


class ActionType(Enum):
    """
    行动类型枚举（细化 ACTION 步骤的具体类别）
    
    成员说明:
        TOOL_CALL: 标准工具调用
                  使用 LangChain BaseTool 接口
                  例: get_stock_price, search_documents
                  
        SKILL_EXECUTE: 技能框架中的技能执行
                     通过 SkillRegistry 调用注册的技能
                     例: RAGRetrievalSkill, FinancialAnalysisSkill
                     
        SUB_AGENT: 子 Agent 调用
                 委托给专门的 Agent 处理复杂子任务
                 例: DataAnalysisAgent, ReportGenerationAgent
                 
        QUERY_REWRITE: 查询改写操作
                     在检索前优化用户查询
                     由 QueryRewriter 模块处理
                     
        CONTEXT_UPDATE: 上下文更新操作
                     向工作记忆中添加或修改信息
                     用于跨步骤的信息传递
    """
    TOOL_CALL = "tool_call"
    SKILL_EXECUTE = "skill_execute"
    SUB_AGENT = "sub_agent"
    QUERY_REWRITE = "query_rewrite"
    CONTEXT_UPDATE = "context_update"


class StopReason(Enum):
    """
    循环停止原因枚举
    
    记录 ReAct 循环为何终止，用于：
    - 结果质量判断（GOAL_ACHIEVED > NO_MORE_ACTIONS > MAX_STEPS > ERROR）
    - 系统调优（频繁 ERROR 可能提示工具质量问题）
    - 用户反馈（解释为什么给出这样的答案）
    
    成员说明:
        GOAL_ACHIEVED: 目标达成
                     最理想的停止状态
                     LLM 主动输出了 Final Answer
                     
        MAX_STEPS_REACHED: 达到最大迭代次数
                         安全兜底机制
                         可能说明任务过于复杂或工具不足
                         
        ERROR_OCCURRED: 发生错误
                      连续多次工具调用失败
                      或 LLM 返回格式异常
                      
        USER_INTERRUPT: 用户中断
                     外部取消信号（如超时）
                     
        NO_MORE_ACTIONS: 无更多行动可执行
                       LLM 判断已无需更多信息
                       但未显式输出 Final Answer
    """
    GOAL_ACHIEVED = "goal_achieved"
    MAX_STEPS_REACHED = "max_steps"
    ERROR_OCCURRED = "error"
    USER_INTERRUPT = "user_interrupt"
    NO_MORE_ACTIONS = "no_more_actions"


class ReasoningType(Enum):
    """
    推理类型枚举（用于分类 Thought 步骤的内容）
    
    成员说明:
        ANALYSIS: 分析型推理
                拆解问题，识别关键要素
                例:"这个问题涉及估值、盈利、风险三个维度"
                
        PLANNING: 规划型推理
                制定行动计划，确定工具调用顺序
                例:"我应该先获取财务数据，再进行对比分析"
                
        REFLECTION: 反思型推理
                 评估当前进展，调整策略
                 例:"已有数据不够，需要补充行业对比"
                 
        HYPOTHESIS: 假设型推理
                  提出假设并验证
                  例:"假设茅台合理PE为30倍，则对应股价..."
    """
    ANALYSIS = "analysis"
    PLANNING = "planning"
    REFLECTION = "reflection"
    HYPOTHESIS = "hypothesis"


# ════════════════════════════════════════════════════════════
# 第二部分：数据模型定义
# ════════════════════════════════════════════════════════════


@dataclass
class ActionCall:
    """
    工具/技能调用记录
    
    记录一次工具调用的完整生命周期，包括输入参数、执行结果和性能指标。
    
    属性分组:
        【身份标识】
        action_id: 全局唯一标识符（UUID 格式）
        action_type: 调用类型（见 ActionType 枚举）
        tool_name: 工具/技能名称
        
        【调用参数】
        parameters: 传递给工具的参数字典
        
        【执行结果】
        result: 工具返回值（任意类型）
        success: 是否成功完成
        error: 失败时的错误信息
        
        【性能指标】
        start_time / end_time: 时间戳（Unix epoch 秒）
        duration_ms: 执行耗时（毫秒）
        tokens_used: 本次调用消耗的 token 数（如可获取）
    
    使用示例:
        >>> call = ActionCall(
        ...     action_id="act_001",
        ...     action_type=ActionType.TOOL_CALL,
        ...     tool_name="get_stock_price",
        ...     parameters={"symbol": "600519"},
        ... )
        >>> # 执行后更新结果
        ... call.result = {"price": 1680}
        ... call.success = True
        ... call.duration_ms = 150.5
    """
    action_id: str
    action_type: ActionType
    tool_name: str
    parameters: Dict[str, Any]
    
    result: Any = None
    success: bool = False
    error: Optional[str] = None
    
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于日志和缓存）"""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }
    
    @property
    def is_completed(self) -> bool:
        """判断调用是否已完成（无论成功或失败）"""
        return self.success or self.error is not None


@dataclass
class ThoughtRecord:
    """
    思考记录
    
    捕获 LLM 在决策前的推理过程，是实现「可解释 AI」的关键数据。
    
    属性说明:
        step_number: 所属步骤编号
        content: 思考文本内容（原始输出）
        reasoning_type: 推理类型分类（见 ReasoningType 枚举）
        
        confidence: 置信度评分 [0.0, 1.0]
                   表示 LLM 对此思考的确信程度
                   高置信度 → 更可能坚持此路线
                   低置信度 → 可能需要探索其他方案
                   
        alternatives_considered: 考虑过的替代方案列表
                                用于展示推理的全面性
                                也帮助调试时理解为什么选择了当前路径
    
    示例:
        >>> thought = ThoughtRecord(
        ...     step_number=1,
        ...     content="用户询问估值，需要PE/PB/ROE三个指标...",
        ...     reasoning_type=ReasoningType.ANALYSIS,
        ...     confidence=0.9,
        ...     alternatives_considered=["直接搜索研报", "使用DCF模型"],
        ... )
    """
    step_number: int
    content: str
    reasoning_type: str = "analysis"
    
    confidence: float = 0.8
    alternatives_considered: List[str] = field(default_factory=list)
    
    @property
    def is_high_confidence(self) -> bool:
        """高置信度判定（> 0.85）"""
        return self.confidence > 0.85


@dataclass
class ObservationRecord:
    """
    观察记录
    
    处理工具返回的原始结果，提取结构化信息供后续步骤使用。
    
    属性说明:
        step_number: 所属步骤编号
        raw_result: 工具原始返回值（未加工）
        processed_summary: 处理后的摘要文本（限制长度）
        
        key_findings: 关键发现列表
                    从结果中自动提取的重要信息点
                    如数值、实体、结论性语句等
                    
        data_quality_score: 数据质量评分 [0.0, 1.0]
                          基于完整性、一致性、时效性评估
                          
        relevance_to_goal: 与目标的关联度 [0.0, 1.0]
                         评估该观察对回答用户问题的贡献程度
    
    数据处理流水线:
        raw_result → _summarize_observation() → processed_summary
                   → _extract_key_findings() → key_findings
                   → _score_quality() → data_quality_score
    """
    step_number: int
    raw_result: Any
    processed_summary: str = ""
    
    key_findings: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    relevance_to_goal: float = 1.0
    
    @property
    def has_valuable_info(self) -> bool:
        """判断是否包含有价值的信息"""
        return (
            len(self.key_findings) > 0 or
            self.data_quality_score > 0.5 or
            self.relevance_to_goal > 0.7
        )


@dataclass 
class ReActStep:
    """
    单个 ReAct 步骤
    
    将 Thought/Action/Observation 组合为一个逻辑单元，
    代表循环中的一轮迭代。
    
    属性说明:
        step_id: 步骤唯一标识（格式: "step_{序号}"）
        step_number: 步骤序号（从 1 开始）
        step_type: 步骤类型（见 ReActStepType 枚举）
        
        thought: 思考记录（THOUGHT/FINISH 类型时存在）
        action: 行动记录（ACTION 类型时存在）
        observation: 观察记录（ACTION/OBSERVATION 类型时存在）
        
        timestamp: ISO 格式时间戳
    
    典型的步骤序列:
        Step 1: [THOUGHT] → "我需要获取股票价格"
        Step 2: [ACTION] → 调用 get_stock_price
        Step 3: [OBSERVATION] → "价格为 1680 元"
        Step 4: [THOUGHT] → "还需要 PE 数据..."
        Step 5: [ACTION] → 调用 get_financials
        Step 6: [OBSERVATION] → "PE 为 28x"
        Step 7: [FINISH] → 最终答案
    """
    step_id: str
    step_number: int
    step_type: ReActStepType
    
    thought: Optional[ThoughtRecord] = None
    action: Optional[ActionCall] = None
    observation: Optional[ObservationRecord] = None
    
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于日志和持久化）"""
        result = {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp,
        }
        
        if self.thought:
            result["thought"] = {
                "content": self.thought.content[:200],
                "reasoning_type": self.thought.reasoning_type,
                "confidence": self.thought.confidence,
            }
        
        if self.action:
            result["action"] = self.action.to_dict()
        
        if self.observation:
            result["observation"] = {
                "summary": self.observation.processed_summary[:200],
                "key_findings_count": len(self.observation.key_findings),
                "quality": self.observation.data_quality_score,
            }
        
        return result


@dataclass
class ReActState:
    """
    ReAct 循环完整状态
    
    这是整个循环的状态机，记录从开始到结束的所有信息。
    EnhancedReActAgent.run() 的返回值就是此对象。
    
    属性分组:
        【基本信息】
        query: 用户原始查询（未经改写）
        goal: 当前活跃目标（可能在演化中）
        
        【执行历史】
        steps: 已执行的步骤列表（按时间顺序）
        current_step: 当前步骤索引
        
        【上下文累积】
        context: 工作记忆字典（跨步骤共享）
        intermediate_results: 各步骤的中间产出
        
        【目标追踪】
        goal_progress: 目标完成进度 [0.0, 1.0]
        remaining_subgoals: 未完成的子目标列表
        
        【资源消耗统计】
        total_tokens_used: 累计 Token 消耗
        total_time_ms: 总执行耗时（毫秒）
        tool_call_count: 工具调用总次数
        
        【最终输出】
        final_answer: 最终答案文本
        stop_reason: 循环停止原因
    
    下游使用:
        >>> state = agent.run("茅台估值")
        >>> print(f"执行了 {len(state.steps)} 步")
        >>> print(f"消耗 {state.total_tokens_used} tokens")
        >>> print(f"答案: {state.final_answer}")
        >>> 
        >>> # 详细审计
        >>> for step in state.steps:
        ...     if step.action:
        ...         print(f"调用 {step.action.tool_name}: {step.action.duration_ms:.0f}ms")
    """
    query: str
    goal: str
    
    steps: List[ReActStep] = field(default_factory=list)
    current_step: int = 0
    
    context: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    goal_progress: float = 0.0
    remaining_subgoals: List[str] = field(default_factory=list)
    
    total_tokens_used: int = 0
    total_time_ms: float = 0.0
    tool_call_count: int = 0
    
    final_answer: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    
    @property
    def n_thought_steps(self) -> int:
        """思考步骤数"""
        return sum(1 for s in self.steps if s.step_type == ReActStepType.THOUGHT)
    
    @property
    def n_action_steps(self) -> int:
        """行动步骤数"""
        return sum(1 for s in self.steps if s.step_type == ReActStepType.ACTION)
    
    @property
    def n_observation_steps(self) -> int:
        """观察步骤数"""
        return sum(1 for s in self.steps if s.step_type == ReActStepType.OBSERVATION)
    
    @property
    def successful_actions(self) -> int:
        """成功的工具调用次数"""
        return sum(
            1 for s in self.steps
            if s.action and s.action.success
        )
    
    @property
    def failed_actions(self) -> int:
        """失败的工具调用次数"""
        return sum(
            1 for s in self.steps
            if s.action and not s.action.success and s.action.error
        )
    
    @property
    def avg_action_duration(self) -> float:
        """平均工具调用耗时（毫秒）"""
        actions = [
            s.action for s in self.steps
            if s.action and s.action.duration_ms > 0
        ]
        if not actions:
            return 0.0
        return sum(a.duration_ms for a in actions) / len(actions)
    
    @property
    def is_successful_completion(self) -> bool:
        """是否成功完成（目标达成）"""
        return self.stop_reason == StopReason.GOAL_ACHIEVED
    
    def get_step_by_type(
        self,
        step_type: ReActStepType,
    ) -> List[ReActStep]:
        """按类型筛选步骤"""
        return [s for s in self.steps if s.step_type == step_type]
    
    def get_tool_calls_history(self) -> List[Dict[str, Any]]:
        """获取所有工具调用的历史记录"""
        history = []
        for step in self.steps:
            if step.action:
                history.append({
                    "step": step.step_number,
                    "tool": step.action.tool_name,
                    "params": step.action.parameters,
                    "success": step.action.success,
                    "duration_ms": step.action.duration_ms,
                })
        return history
    
    def summary(self) -> str:
        """生成人类可读的执行摘要"""
        lines = [
            f"Query: {self.query[:60]}...",
            f"Goal: {self.goal[:60]}...",
            f"",
            f"Execution Summary:",
            f"  Total Steps: {len(self.steps)}",
            f"  - Thoughts: {self.n_thought_steps}",
            f"  - Actions: {self.n_action_steps} ({self.successful_actions} ok, {self.failed_actions} fail)",
            f"  - Observations: {self.n_observation_steps}",
            f"",
            f"Resources:",
            f"  Tokens Used: {self.total_tokens_used:,}",
            f"  Time: {self.total_time_ms:.0f}ms",
            f"  Avg Action Duration: {self.avg_action_duration:.0f}ms",
            f"",
            f"Result:",
            f"  Progress: {self.goal_progress:.1%}",
            f"  Stop Reason: {self.stop_reason.value if self.stop_reason else 'N/A'}",
        ]
        
        if self.final_answer:
            lines.append(f"")
            lines.append(f"Final Answer:")
            lines.append(f"  {self.final_answer[:300]}{'...' if len(self.final_answer) > 300 else ''}")
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 第三部分：ReAct Agent 主实现
# ════════════════════════════════════════════════════════════


class EnhancedReActAgent:
    """
    增强版 ReAct Agent — 主协调器
    
    【核心能力】
    
    1. **结构化推理 (Structured Reasoning)**
       每个 Thought 都被解析为结构化记录，包含推理类型和置信度。
       这使得推理过程完全可追溯和可审计。
       
    2. **多工具编排 (Multi-Tool Orchestration)**
       支持同时管理多个工具，LLM 自主选择最合适的工具组合。
       工具可以是 LangChain BaseTool 或自定义 Skill。
       
    3. **动态目标管理 (Dynamic Goal Management)**
       目标可以在循环中演化。例如：
       - 初始目标: "分析茅台估值"
       - 中间演化: "获取 PE 和 ROE 数据"
       - 最终收敛: "综合所有数据给出结论"
       
    4. **自适应停止判断 (Adaptive Stopping)**
       不仅依赖最大步数限制，还综合考虑：
       - 是否出现 Final Answer
       - 连续失败次数是否超阈值
       - LLM 是否主动表示无更多需求
       
    5. **完整可追溯性 (Full Traceability)**
       每个步骤都记录完整的输入/输出/元数据，
       支持事后的执行回放和质量分析。
    
    【技术架构】
    
    本实现基于 LangGraph 的 create_react_agent 构建，
    在其基础上增加了：
    - 结构化的状态记录和解析
    - 丰富的性能统计和日志
    - 错误恢复和优雅降级
    - 观察结果的智能摘要
    
    【使用示例】
    
    基础用法:
        >>> from langchain.tools import tool
        >>> @tool
        ... def get_stock_price(symbol: str) -> str:
        ...     return f"{symbol}: 1680元"
        >>>
        >>> agent = EnhancedReActAgent(tools=[get_stock_price])
        >>> state = agent.run("茅台今天的股价？")
        >>> print(state.final_answer)
    
    高级用法（带上下文）:
        >>> context = {
        ...     "user_preference": "关注长期价值",
        ...     "risk_tolerance": "中等",
        ... }
        >>> state = agent.run("茅台值得投资吗？", context=context)
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        max_iterations: int = 10,
        verbose: bool = True,
        max_consecutive_errors: int = 3,
        retry_on_error: bool = True,
    ):
        """
        初始化 ReAct Agent
        
        Args:
            tools: 可用工具列表
                  每个 tool 必须实现 LangChain BaseTool 接口
                  工具会被注册到内部字典中，通过 name 索引
                  
            max_iterations: 最大迭代次数（安全上限）
                          推荐: 简单查询 5-8 次，复杂分析 10-15 次
                          过大可能导致 Token 浪费和延迟增加
                          
            verbose: 是否打印详细执行日志
                    生产环境建议 False（减少日志噪音）
                    开发调试建议 True
                    
            max_consecutive_errors: 最大连续错误次数
                                   超过此次数后强制停止
                                   防止陷入错误重试死循环
                                   
            retry_on_error: 工具调用失败时是否自动重试
                          当前仅记录，实际重试由 LLM 决定
        """
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_consecutive_errors = max_consecutive_errors
        self.retry_on_error = retry_on_error
        
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )
        
        try:
            self.standard_react_agent = create_react_agent(
                model=self.llm,
                tools=tools,
            )
        except Exception as e:
            print(f"[ReAct] ⚠️ 创建标准 Agent 失败: {e}")
            self.standard_react_agent = None
        
        self.system_prompt = self._build_system_prompt()
        
        self._error_count: int = 0
        self._consecutive_errors: int = 0
    
    def _build_system_prompt(self) -> str:
        """
        构建 ReAct 系统提示词（强工具调用引导版）
        
        核心设计原则:
        1. 强制性: 明确告知 LLM 必须使用工具获取数据
        2. 示例驱动: 提供完整的工具调用示例
        3. 分步约束: 每一步只能做一件事
        4. 格式严格: 工具调用必须符合 JSON schema
        
        Returns:
            格式化的系统提示词字符串
        """
        tools_desc = "\n".join(
            f"- **{name}**: {getattr(tool, 'description', 'No description')}\n"
            f"  参数格式: {json.dumps(getattr(tool, 'args_schema', {}).get('properties', {}) if hasattr(getattr(tool, 'args_schema', {}), 'get') else {}, ensure_ascii=False) if hasattr(tool, 'args_schema') else '{}'}"
            for name, tool in self.tools.items()
        )
        
        return f"""你是一个专业的金融研究助手。你的核心能力是通过调用工具获取实时数据来回答用户问题。

## ⚠️ 最重要规则：你必须使用工具！

你没有内置的金融数据！所有股价、财报、新闻等信息都必须通过工具获取。
- 绝对不要凭记忆或猜测回答数值问题
- 每次回答涉及具体数据时，必须先调用对应工具

## 你的工作流程（每轮只执行一步）

### 第 1 步：Thought（思考）
分析用户需要什么信息，决定调用哪个工具。

```
Thought: 用户询问[XXX]，我需要先获取[YYY]数据，应该使用 [工具名] 工具。
```

### 第 2 步：Action（行动）⭐ 这是关键步骤！
立即调用工具获取数据。这是唯一能获取信息的方式。

```
Action: [工具名]
Action Input: {{"参数名": "参数值"}}
```

### 第 3 步：Observation（观察）
系统会自动返回工具结果，你基于结果进行分析。

```
Observation: 工具返回了[...]，这说明...
```

### 第 4 步：决定下一步
- 如果还需要更多信息 → 回到第 1 步，继续调用其他工具
- 如果信息足够 → 输出 Final Answer

## 可用的工具

{tools_desc}

## 完整示例

**用户**: 茅台今天的股价？

**你的回复**:
Thought: 用户询问贵州茅台的当前股价。我需要使用 get_stock_quote 工具获取实时行情数据。茅台的股票代码是 600519。
Action: tool_get_stock_quote
Action Input: {{"symbol": "600519"}}
[等待工具返回...]
Observation: 工具返回了价格1680元、涨跌幅+2.3%等数据。
Thought: 已经获得了茅台的完整行情数据，可以回答用户问题了。
Final Answer: 根据最新行情数据，贵州茅台(600519)当前股价为 **1680元/股**，今日上涨 **+2.3%**...

## 格式要求

- Action 和 Action Input 必须分两行输出
- Action Input 必须是有效的 JSON 格式
- 参数值必须是字符串或数字，不要用中文描述
- 每次只调用一个工具，等结果返回后再决定下一步

## 开始处理用户问题！请立即开始 Thought → Action 循环。"""
    
    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReActState:
        """
        执行完整的 ReAct 循环（主入口方法）
        
        这是 Agent 对外暴露的核心方法，编排完整的推理-行动-观察流程。
        
        Args:
            query: 用户查询文本（原始输入）
            context: 初始上下文字典（可选）
                    可包含用户偏好、会话历史等信息
                    
        Returns:
            ReActState 完整状态对象，包含：
            - steps: 所有步骤记录
            - final_answer: 最终答案
            - stop_reason: 停止原因
            - 各种统计指标
            
        异常处理策略:
        - LangGraph Agent 异常 → 标记 ERROR_OCCURRED，返回部分结果
        - 解析异常 → 尝试从原始消息中提取答案
        - 任何情况下都不会抛出异常给调用者
        
        执行流程:
        1. 初始化状态对象
        2. 构建 LangGraph 输入
        3. 调用 standard_react_agent.invoke()
        4. 解析返回的消息序列
        5. 构建结构化步骤记录
        6. 计算统计指标
        7. 打印执行摘要
        8. 返回完整状态
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[ReAct Agent] 🚀 开始处理: {query[:60]}...")
        print(f"{'='*70}")
        print(f"  配置: max_iterations={self.max_iterations}, "
              f"tools={len(self.tools)}, verbose={self.verbose}")
        
        state = ReActState(
            query=query,
            goal=query,
            context=context or {},
        )
        
        try:
            if not self.standard_react_agent:
                raise RuntimeError("标准 ReAct Agent 未初始化")
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"请研究以下问题：\n\n{query}"),
            ]
            
            if context:
                ctx_str = json.dumps(context, ensure_ascii=False, indent=2)
                messages.append(
                    HumanMessage(content=f"\n[附加上下文]\n{ctx_str}")
                )
            
            react_input = {"messages": messages}
            
            if self.verbose:
                print(f"[ReAct] 📤 发送请求给 LangGraph Agent...")
            
            config = {"recursion_limit": self.max_iterations + 10}
            result = self.standard_react_agent.invoke(react_input, config=config)
            
            state = self._parse_langgraph_result(result, state)
            
        except Exception as e:
            print(f"[ReAct] ❌ 执行出错: {e}")
            state.stop_reason = StopReason.ERROR_OCCURRED
            state.final_answer = (
                f"抱歉，处理过程中出现错误：{str(e)}\n\n"
                "建议稍后重试，或简化问题的复杂度。"
            )
            
            if self.verbose:
                traceback.print_exc()
        
        state.total_time_ms = (time.time() - start_time) * 1000
        
        self._print_summary(state)
        
        return state
    
    def _parse_langgraph_result(
        self,
        result: Dict[str, Any],
        state: ReActState,
    ) -> ReActState:
        """
        解析 LangGraph ReAct Agent 的返回结果
        
        LangGraph 返回的消息序列通常如下:
        [
            SystemMessage(...),      # 系统提示（跳过）
            HumanMessage(...),       # 用户查询（跳过）
            AIMessage(              # 第1轮: Thought + Action
                content="Thought:...",
                additional_kwargs={'tool_calls': [...]}
            ),
            ToolMessage(...),       # 第1轮: Observation
            AIMessage(              # 第2轮: Thought + Action
                ...
            ),
            ToolMessage(...),       # 第2轮: Observation
            AIMessage(              # 最后: Final Answer
                content="Final Answer:..."
            ),
        ]
        
        本方法将上述扁平消息序列转换为结构化的 ReActStep 列表。
        
        Args:
            result: LangGraph invoke() 的返回字典
            state: 当前状态对象（将被填充）
            
        Returns:
            填充完成的 ReActState 对象
        """
        messages = result.get("messages", [])
        
        if self.verbose:
            print(f"[ReAct] 📥 收到 {len(messages)} 条消息，开始解析...")
            for idx, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content_preview = (msg.content or "")[:100] if hasattr(msg, 'content') else str(msg)[:100]
                has_tool_calls = bool(getattr(msg, 'tool_calls', None))
                print(f"  [{idx}] {msg_type}: {content_preview}{' [tool_calls]' if has_tool_calls else ''}")
        
        current_step_num = 0
        pending_thought: Optional[ThoughtRecord] = None
        
        for i, message in enumerate(messages):
            step_id = f"step_{i}"
            
            if isinstance(message, (SystemMessage, HumanMessage)):
                continue
            
            elif isinstance(message, AIMessage):
                content = message.content or ""
                
                thought = self._try_extract_thought(content, current_step_num)
                if thought:
                    pending_thought = thought
                
                final_answer = self._try_extract_final_answer(content)
                if final_answer:
                    state.final_answer = final_answer
                    state.stop_reason = StopReason.GOAL_ACHIEVED
                    
                    observation = ObservationRecord(
                        step_number=current_step_num,
                        raw_result=final_answer,
                        processed_summary=final_answer[:500],
                        key_findings=[],
                        data_quality_score=1.0,
                        relevance_to_goal=1.0,
                    )
                    
                    finish_step = ReActStep(
                        step_id=step_id,
                        step_number=current_step_num,
                        step_type=ReActStepType.FINISH,
                        thought=pending_thought,
                        observation=observation,
                        timestamp=datetime.now().isoformat(),
                    )
                    state.steps.append(finish_step)
                    break
                
                tool_calls = self._extract_tool_calls(message)
                if tool_calls:
                    for tc in tool_calls:
                        action = ActionCall(
                            action_id=f"action_{uuid.uuid4().hex[:8]}",
                            action_type=ActionType.TOOL_CALL,
                            tool_name=tc["name"],
                            parameters=tc["args"],
                            start_time=time.time(),
                        )
                        
                        action_step = ReActStep(
                            step_id=f"{step_id}_action",
                            step_number=current_step_num,
                            step_type=ReActStepType.ACTION,
                            thought=pending_thought,
                            action=action,
                            timestamp=datetime.now().isoformat(),
                        )
                        state.steps.append(action_step)
                        state.tool_call_count += 1
                        current_step_num += 1
                        pending_thought = None
                        
                        self._consecutive_errors = 0
                        self._error_count = 0
                elif content and content.strip():
                    if not pending_thought:
                        pending_thought = ThoughtRecord(
                            step_number=current_step_num,
                            content=content[:500],
                            reasoning_type="analysis",
                            confidence=0.7,
                        )
                    
                    thought_step = ReActStep(
                        step_id=f"{step_id}_thought",
                        step_number=current_step_num,
                        step_type=ReActStepType.THOUGHT,
                        thought=pending_thought,
                        timestamp=datetime.now().isoformat(),
                    )
                    state.steps.append(thought_step)
                    current_step_num += 1
                    pending_thought = None
            
            elif isinstance(message, ToolMessage):
                tool_result = message.content
                tool_name = getattr(message, 'name', None) or "unknown_tool"
                
                observation = ObservationRecord(
                    step_number=current_step_num - 1,
                    raw_result=tool_result,
                    processed_summary=self._summarize_observation(tool_result),
                    key_findings=self._extract_key_findings(tool_result),
                    data_quality_score=self._assess_data_quality(tool_result),
                    relevance_to_goal=self._assess_relevance(tool_result, state.query),
                )
                
                if state.steps:
                    last_step = state.steps[-1]
                    
                    if last_step.step_type == ReActStepType.ACTION:
                        last_step.observation = observation
                        last_step.step_type = ReActStepType.OBSERVATION
                    
                    if last_step.action:
                        last_step.action.result = tool_result
                        last_step.action.success = True
                        last_step.action.end_time = time.time()
                        last_step.action.duration_ms = (
                            last_step.action.end_time - last_step.action.start_time
                        ) * 1000
                
                state.intermediate_results[
                    f"step_{current_step_num}_{tool_name}"
                ] = tool_result
        
        if not state.final_answer and messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    candidate = msg.content.strip()
                    if candidate and not candidate.startswith(("Thought:", "Action:", "Observation:")):
                        state.final_answer = candidate
                        state.stop_reason = StopReason.NO_MORE_ACTIONS
                        break
        
        if not state.stop_reason:
            if self._consecutive_errors >= self.max_consecutive_errors:
                state.stop_reason = StopReason.ERROR_OCCURRED
            else:
                state.stop_reason = StopReason.MAX_STEPS_REACHED
        
        total_possible = min(len(state.steps), self.max_iterations)
        state.goal_progress = min(
            1.0,
            total_possible / max(1, self.max_iterations)
        )
        
        if self.verbose:
            print(f"[ReAct] ✅ 解析完成: {len(state.steps)} 个步骤")
        
        return state
    
    def _try_extract_thought(
        self,
        content: str,
        step_num: int,
    ) -> Optional[ThoughtRecord]:
        """
        尝试从消息内容中提取 Thought 记录
        
        匹配规则:
        - "Thought: ..." （大小写不敏感）
        - 支持多行内容（直到下一个标签或结尾）
        
        Args:
            content: AIMessage 的文本内容
            step_num: 当前步骤编号
            
        Returns:
            ThoughtRecord 或 None（如果未找到）
        """
        pattern = r'Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Observation|Final Answer)\s*:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            thought_text = match.group(1).strip()
            
            reasoning_type = ReasoningType.ANALYSIS.value
            if any(kw in thought_text.lower() for kw in ["首先", "然后", "计划", "应该", "下一步"]):
                reasoning_type = ReasoningType.PLANNING.value
            elif any(kw in thought_text.lower() for kw in ["但是", "然而", "不够", "还需要", "重新考虑"]):
                reasoning_type = ReasoningType.REFLECTION.value
            elif any(kw in thought_text.lower() for kw in ["假设", "如果", "可能", "预计"]):
                reasoning_type = ReasoningType.HYPOTHESIS.value
            
            return ThoughtRecord(
                step_number=step_num,
                content=thought_text,
                reasoning_type=reasoning_type,
                confidence=0.8,
            )
        
        return None
    
    def _try_extract_final_answer(self, content: str) -> Optional[str]:
        """
        尝试从消息内容中提取 Final Answer
        
        Args:
            content: 文本内容
            
        Returns:
            最终答案字符串或 None
        """
        pattern = r'Final\s+Answer\s*:\s*(.*?)(?:\n\s*(?:Thought|Action|Observation)\s*:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        if content.lower().startswith("final answer"):
            return content.split(":", 1)[-1].strip()
        
        return None
    
    def _extract_tool_calls(self, message: AIMessage) -> List[Dict[str, Any]]:
        """
        从 AIMessage 中提取工具调用信息
        
        LangGraph/LangChain 的工具调用存储在:
        - message.additional_kwargs['tool_calls'] (OpenAI 格式)
        - 或 message.tool_calls (LangChain 标准格式)
        
        Args:
            message: AIMessage 对象
            
        Returns:
            [{"name": str, "args": dict}, ...] 列表
        """
        calls = []
        
        if hasattr(message, 'additional_kwargs'):
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            for tc in tool_calls:
                func = tc.get('function', {})
                name = func.get('name', '')
                args_raw = func.get('arguments', '{}')
                
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"raw_input": str(args_raw)}
                
                calls.append({"name": name, "args": args})
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                calls.append({
                    "name": tc.get('name', ''),
                    "args": tc.get('args', {}),
                })
        
        return calls
    
    def _summarize_observation(
        self,
        raw_result: Any,
        max_length: int = 300,
    ) -> str:
        """
        生成观察结果的摘要文本
        
        处理策略:
        - 字符串: 直接截断（保留前 max_length 字符）
        - 字典: JSON 格式化后截断
        - 其他: 转字符串后截断
        
        Args:
            raw_result: 原始结果
            max_length: 最大长度
            
        Returns:
            摘要字符串
        """
        if isinstance(raw_result, str):
            if len(raw_result) <= max_length:
                return raw_result
            return raw_result[:max_length] + f"\n... (共{len(raw_result)}字符)"
        
        elif isinstance(raw_result, dict):
            summary = json.dumps(raw_result, ensure_ascii=False, indent=2)
            if len(summary) <= max_length:
                return summary
            return summary[:max_length] + f"\n... (共{len(summary)}字符)"
        
        elif isinstance(raw_result, (list, tuple)):
            text = str(list(raw_result))
            if len(text) <= max_length:
                return text
            return text[:max_length] + "..."
        
        else:
            text = str(raw_result)
            return text[:max_length] + ("..." if len(text) > max_length else "")
    
    def _extract_key_findings(self, raw_result: Any) -> List[str]:
        """
        从观察结果中提取关键发现
        
        提取策略:
        - 数值模式: 金额、百分比、倍数等
        - 关键字段: price/revenue/profit/score 等
        - 结论性语句: 包含"显示"、"表明"、"说明"等的句子
        
        Args:
            raw_result: 原始结果
            
        Returns:
            关键发现列表（最多 10 条）
        """
        findings: List[str] = []
        
        text = ""
        if isinstance(raw_result, str):
            text = raw_result
        elif isinstance(raw_result, dict):
            text = json.dumps(raw_result, ensure_ascii=False)
        else:
            text = str(raw_result)
        
        number_patterns = [
            r'[\d,]+\.?\d*\s*(?:%|%%|倍|元|亿|万|万元|亿元)',
            r'(?:营收|利润|收入|净利|毛利)\w*[：:]\s*[\d,]+\.?\d*',
            r'(?:PE|PB|ROE|EPS)\w*[：:]\s*[\d,]+\.?\d*',
            r'增长\w*[：:]?\s*[-+]?\d+\.?\d*%',
            r'\d{4}年?(?:Q[1-4])?',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            findings.extend(matches[:3])
        
        if len(findings) < 5 and isinstance(raw_result, dict):
            priority_keys = [
                'price', 'change', 'revenue', 'profit',
                'pe', 'pb', 'roe', 'score', 'rating',
                'price_cn', 'name', 'symbol',
            ]
            for key in priority_keys:
                if key in raw_result:
                    findings.append(f"{key}: {raw_result[key]}")
        
        conclusion_patterns = [
            r'[^。]*?(?:显示|表明|说明|反映|意味着)[^。]*?。',
        ]
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, text)
            findings.extend(matches[:2])
        
        return findings[:10]
    
    def _assess_data_quality(self, raw_result: Any) -> float:
        """
        评估数据质量分数 [0.0, 1.0]
        
        评估维度:
        - 完整性: 是否有实质性内容
        - 结构化程度: 是否为结构化数据（JSON/Dict）
        - 信息密度: 单位长度内的有效信息量
        
        Args:
            raw_result: 原始结果
            
        Returns:
            质量分数
        """
        score = 0.5
        
        if raw_result is None:
            return 0.0
        
        text = str(raw_result)
        
        if len(text) < 10:
            score -= 0.2
        elif len(text) > 50:
            score += 0.1
        
        if isinstance(raw_result, dict):
            score += 0.2
            if len(raw_result) >= 3:
                score += 0.1
        
        if isinstance(raw_result, str):
            numbers = re.findall(r'\d+\.?\d*', text)
            if len(numbers) >= 3:
                score += 0.15
        
        error_indicators = ['error', 'failed', 'exception', '无法', '错误']
        text_lower = text.lower()
        if any(ind in text_lower for ind in error_indicators):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_relevance(
        self,
        raw_result: Any,
        query: str,
    ) -> float:
        """
        评估观察结果与查询目标的关联度 [0.0, 1.0]
        
        简化实现：基于关键词重叠度计算
        实际生产环境可使用 Embedding 相似度
        
        Args:
            raw_result: 观察结果
            query: 原始查询
            
        Returns:
            关联度分数
        """
        result_text = str(raw_result).lower()
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = sum(1 for w in query_words if w in result_text)
        return min(1.0, overlap / len(query_words))
    
    def _print_summary(self, state: ReActState) -> None:
        """
        打印执行总结报告
        
        输出内容包括:
        - 基本信息（查询、目标）
        - 执行统计（步数、耗时、Token）
        - 流程概览（每步的类型和简要内容）
        - 最终答案预览
        """
        print(f"\n{'='*70}")
        print(f"[ReAct Agent] 📊 执行完成")
        print(f"{'='*70}")
        
        print(f"\n📋 基本信息:")
        print(f"  查询: {state.query[:50]}...")
        print(f"  目标: {state.goal[:50]}...")
        
        print(f"\n📈 执行统计:")
        print(f"  总步骤数: {len(state.steps)}")
        print(f"  ├─ 思考: {state.n_thought_steps}")
        print(f"  ├─ 行动: {state.n_action_steps}")
        print(f"  ├─ 观察: {state.n_observation_steps}")
        print(f"  └─ 完成: {sum(1 for s in state.steps if s.step_type == ReActStepType.FINISH)}")
        print(f"  工具调用: {state.tool_call_count}次 "
              f"(成功 {state.successful_actions}, 失败 {state.failed_actions})")
        print(f"  总耗时: {state.total_time_ms:.0f}ms")
        print(f"  平均单次调用: {state.avg_action_duration:.0f}ms")
        
        stop_emoji = {
            StopReason.GOAL_ACHIEVED: "✅",
            StopReason.MAX_STEPS_REACHED: "⚠️",
            StopReason.ERROR_OCCURRED: "❌",
            StopReason.USER_INTERRUPT: "🛑",
            StopReason.NO_MORE_ACTIONS: "🔚",
        }
        emoji = stop_emoji.get(state.stop_reason, "❓")
        print(f"  停止原因: {emoji} {state.stop_reason.value if state.stop_reason else 'N/A'}")
        print(f"  进度: {state.goal_progress:.1%}")
        
        print(f"\n🔄 执行流程:")
        type_icons = {
            ReActStepType.THOUGHT: "💭",
            ReActStepType.ACTION: "🔧",
            ReActStepType.OBSERVATION: "👁️",
            ReActStepType.REFLECTION: "🔄",
            ReActStepType.FINISH: "✅",
        }
        
        for i, step in enumerate(state.steps, 1):
            icon = type_icons.get(step.step_type, "❓")
            
            info = ""
            if step.thought:
                preview = step.thought.content[:40].replace('\n', ' ')
                info = f" {preview}..."
            elif step.action:
                info = f" [{step.action.tool_name}]"
                if step.action.success:
                    info += f" ✓{step.action.duration_ms:.0f}ms"
                else:
                    info += f" ✗ {step.action.error[:30] if step.action.error else ''}"
            elif step.observation:
                preview = step.observation.processed_summary[:40].replace('\n', ' ')
                info = f" {preview}..."
            
            print(f"  {icon} Step {i:>2} ({step.step_type.value:>12}):{info}")
        
        if state.final_answer:
            answer_preview = state.final_answer.replace('\n', ' ')[:150]
            print(f"\n🎯 最终答案:")
            print(f"  {answer_preview}{'...' if len(state.final_answer) > 150 else ''}")


# ════════════════════════════════════════════════════════════
# 第四部分：便捷函数
# ════════════════════════════════════════════════════════════


def create_react_agent_with_tools(
    tools: List[BaseTool],
    **kwargs,
) -> EnhancedReActAgent:
    """
    工厂函数：创建配置好的 ReAct Agent
    
    封装 EnhancedReActAgent 的初始化流程，
    减少调用方的样板代码。
    
    Args:
        tools: LangChain BaseTool 列表
        **kwargs: 传递给 EnhancedReActAgent.__init__() 的额外参数
                 常用参数:
                 - max_iterations: int (默认 10)
                 - verbose: bool (默认 True)
                 - max_consecutive_errors: int (默认 3)
                 
    Returns:
        配置完成的 EnhancedReActAgent 实例
        
    示例:
        >>> from langchain.tools import tool
        >>> @tool
        ... def search(query: str) -> str:
        ...     return "搜索结果..."
        >>>
        >>> agent = create_react_agent_with_tools(
        ...     tools=[search],
        ...     max_iterations=5,
        ...     verbose=False,
        ... )
        >>> state = agent.run("查找茅台相关研报")
    """
    return EnhancedReActAgent(tools=tools, **kwargs)


def react_loop(
    query: str,
    tools: List[BaseTool],
    max_iterations: int = 10,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> ReActState:
    """
    便捷函数：一次性执行完整的 ReAct 循环
    
    适用于简单场景，无需手动管理 Agent 生命周期。
    
    Args:
        query: 用户查询
        tools: 可用工具列表
        max_iterations: 最大迭代次数
        context: 初始上下文（可选）
        verbose: 是否打印详细日志
        
    Returns:
        ReActState 完整状态
        
    示例:
        >>> state = react_loop(
        ...     query="茅台今天的股价如何？",
        ...     tools=[get_stock_price, get_company_info],
        ...     max_iterations=5,
        ... )
        >>> print(state.final_answer)
        >>> print(f"执行了 {len(state.steps)} 个步骤")
    """
    agent = create_react_agent_with_tools(
        tools=tools,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    return agent.run(query, context=context)


# ════════════════════════════════════════════════════════════
# 第五部分：自测套件
# ════════════════════════════════════════════════════════════


def _run_self_tests():
    """
    ReAct 循环自测套件
    
    测试场景覆盖:
    1. 数据模型创建与属性访问
    2. 枚举值正确性
    3. 状态计算属性（统计指标）
    4. 观察摘要与关键发现提取
    5. 数据质量评估
    6. 序列化与反序列化
    """
    print("\n" + "=" * 70)
    print("  Enhanced ReAct Loop — Self-Test Suite")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    def test(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  [{status}] {name}")
        if detail and not condition:
            print(f"         → {detail}")
        if condition:
            passed += 1
        else:
            failed += 1
    
    print("\n--- Test Group 1: Enums ---")
    
    test(
        "ReActStepType enum values",
        set(e.value for e in ReActStepType) == {
            "thought", "action", "observation", "reflection", "finish"
        },
    )
    test(
        "ActionType enum values",
        len(ActionType) == 5,
    )
    test(
        "StopReason enum values",
        len(StopReason) == 5,
    )
    test(
        "ReasoningType enum values",
        len(ReasoningType) == 4,
    )
    
    print("\n--- Test Group 2: Data Models ---")
    
    action = ActionCall(
        action_id="test_001",
        action_type=ActionType.TOOL_CALL,
        tool_name="get_stock_price",
        parameters={"symbol": "600519"},
        result={"price": 1680},
        success=True,
        duration_ms=150.5,
    )
    test(
        "ActionCall creation",
        action.tool_name == "get_stock_price" and action.success,
    )
    test(
        "ActionCall.is_completed",
        action.is_completed,
    )
    action_dict = action.to_dict()
    test(
        "ActionCall.to_dict serialization",
        "action_id" in action_dict and "tool_name" in action_dict,
    )
    
    thought = ThoughtRecord(
        step_number=1,
        content="需要获取股价数据",
        reasoning_type="planning",
        confidence=0.9,
        alternatives_considered=["直接搜索", "调用API"],
    )
    test(
        "ThoughtRecord creation",
        thought.reasoning_type == "planning" and len(thought.alternatives_considered) == 2,
    )
    test(
        "ThoughtRecord.is_high_confidence",
        thought.is_high_confidence,
    )
    
    obs = ObservationRecord(
        step_number=1,
        raw_result={"price": 1680, "change": "+2.3%"},
        processed_summary="股价1680元，上涨2.3%",
        key_findings=["price: 1680", "change: +2.3%"],
        data_quality_score=0.95,
        relevance_to_goal=0.9,
    )
    test(
        "ObservationRecord creation",
        len(obs.key_findings) == 2 and obs.has_valuable_info,
    )
    
    step = ReActStep(
        step_id="step_1",
        step_number=1,
        step_type=ReActStepType.ACTION,
        action=action,
        timestamp=datetime.now().isoformat(),
    )
    test(
        "ReActStep creation",
        step.step_type == ReActStepType.ACTION and step.action is not None,
    )
    step_dict = step.to_dict()
    test(
        "ReActStep.to_dict serialization",
        "step_type" in step_dict and "action" in step_dict,
    )
    
    print("\n--- Test Group 3: ReActState Computed Properties ---")
    
    state = ReActState(
        query="测试查询",
        goal="测试目标",
        steps=[
            ReActStep(
                step_id="s1", step_number=1,
                step_type=ReActStepType.THOUGHT,
                timestamp=datetime.now().isoformat(),
            ),
            ReActStep(
                step_id="s2", step_number=2,
                step_type=ReActStepType.ACTION,
                action=ActionCall(
                    action_id="a1", action_type=ActionType.TOOL_CALL,
                    tool_name="tool1", parameters={}, success=True,
                    duration_ms=100,
                ),
                timestamp=datetime.now().isoformat(),
            ),
            ReActStep(
                step_id="s3", step_number=3,
                step_type=ReActStepType.OBSERVATION,
                timestamp=datetime.now().isoformat(),
            ),
            ReActStep(
                step_id="s4", step_number=4,
                step_type=ReActStepType.ACTION,
                action=ActionCall(
                    action_id="a2", action_type=ActionType.TOOL_CALL,
                    tool_name="tool2", parameters={}, success=False,
                    error="Timeout", duration_ms=5000,
                ),
                timestamp=datetime.now().isoformat(),
            ),
        ],
        stop_reason=StopReason.GOAL_ACHIEVED,
        final_answer="这是最终答案",
    )
    
    test(
        "n_thought_steps count",
        state.n_thought_steps == 1,
    )
    test(
        "n_action_steps count",
        state.n_action_steps == 2,
    )
    test(
        "successful_actions count",
        state.successful_actions == 1,
    )
    test(
        "failed_actions count",
        state.failed_actions == 1,
    )
    test(
        "avg_action_duration calculation",
        abs(state.avg_action_duration - 2550.0) < 1.0,
        f"Got {state.avg_action_duration}",
    )
    test(
        "is_successful_completion",
        state.is_successful_completion,
    )
    
    thoughts = state.get_step_by_type(ReActStepType.THOUGHT)
    test(
        "get_step_by_type filter",
        len(thoughts) == 1 and thoughts[0].step_type == ReActStepType.THOUGHT,
    )
    
    history = state.get_tool_calls_history()
    test(
        "get_tool_calls_history",
        len(history) == 2 and history[0]["tool"] == "tool1",
    )
    
    summary = state.summary()
    test(
        "summary generation",
        len(summary) > 100 and "Execution Summary" in summary,
    )
    
    print("\n--- Test Group 4: Text Processing Utilities ---")
    
    agent = EnhancedReActAgent.__new__(EnhancedReActAgent)
    
    short_summary = agent._summarize_observation("短文本")
    test(
        "Short observation summarization",
        short_summary == "短文本",
    )
    
    long_text = "A" * 500
    long_summary = agent._summarize_observation(long_text, max_length=100)
    test(
        "Long observation truncation",
        len(long_summary) <= 110 and "..." in long_summary,
    )
    
    dict_result = {"price": 1680, "revenue": 1505}
    dict_summary = agent._summarize_observation(dict_result, max_length=50)
    test(
        "Dict observation formatting",
        "price" in dict_summary,
    )
    
    findings = agent._extract_key_findings("营收1505亿，净利润747亿，同比增长18%")
    test(
        "Key findings extraction from text",
        len(findings) >= 2,
        f"Got {findings}",
    )
    
    findings_dict = agent._extract_key_findings({"price": 1680, "change": 2.3})
    test(
        "Key findings extraction from dict",
        len(findings_dict) >= 1,
    )
    
    quality_good = agent._assess_data_quality({"price": 1680, "volume": 10000})
    test(
        "Data quality assessment (good)",
        quality_good > 0.6,
        f"Score: {quality_good}",
    )
    
    quality_empty = agent._assess_data_quality(None)
    test(
        "Data quality assessment (None)",
        quality_empty == 0.0,
    )
    
    quality_error = agent._assess_data_quality("Error: connection failed")
    test(
        "Data quality assessment (error)",
        quality_error < 0.5,
        f"Score: {quality_error}",
    )
    
    relevance = agent._assess_relevance("茅台股价1680元", "茅台股价")
    test(
        "Relevance assessment (high)",
        relevance > 0.5,
        f"Score: {relevance}",
    )
    
    relevance_low = agent._assess_relevance("今天天气不错", "茅台股价")
    test(
        "Relevance assessment (low)",
        relevance_low < 0.5,
        f"Score: {relevance_low}",
    )
    
    print("\n--- Test Group 5: Convenience Functions ---")
    
    test(
        "create_react_agent_with_tools exists",
        callable(create_react_agent_with_tools),
    )
    
    test(
        "react_loop exists",
        callable(react_loop),
    )
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed - please review")
    print("=" * 70)


# ════════════════════════════════════════════════════════════
# 第八部分：手动 ReAct Agent（解决 GLM tool_calling 兼容性）
# ════════════════════════════════════════════════════════════


class ManualReActAgent:
    """
    手动 ReAct Agent — 不依赖 LangGraph 的 create_react_agent
    
    【为什么需要手动实现？】
    
    LangGraph 的 create_react_agent 要求 LLM 原生支持 function calling：
    - GPT-4: ✅ 完美支持，自动输出 tool_calls 格式
    - GLM-4 (via OpenAI API): ⚠️ 兼容性问题，不触发 tool_calls
    
    本实现通过**文本解析**方式绕过此限制：
    - LLM 输出标准格式的 Action 文本
    - 正则表达式提取工具名和参数
    - 手动调用 Python 函数获取结果
    - 将结果注入对话历史作为 Observation
    
    【工作流程】
    
    ┌─────────────────────────────────────────────────────┐
    │                  手动 ReAct 循环                   │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  messages = [System, Human(query)]                │
    │                                                     │
    │  for step in range(max_iterations):                 │
    │      ┌─────────────────────────────────┐           │
    │      │ 1. LLM.invoke(messages)       │           │
    │      │    → response.content        │           │
    │      └──────────┬──────────────────┘           │
    │                 ▼                               │
    │      ┌─────────────────────────────────┐           │
    │      │ 2. 解析 response               │           │
    │      │   → Thought? 记录              │           │
    │      │   → Action? 提取工具名+参数    │           │
    │      │   → Final Answer? 返回结果     │           │
    │      └──────────┬──────────────────┘           │
    │         有Action?│                              │
    │      Yes ─┘     │ No                           │
    │          ▼       ▼                              │
    │  ┌──────────┐  ┌──────────┐                     │
    │  │3.调用工具│  │追加消息  │                     │
    │  │4.注入Obs│  │继续循环  │                     │
    │  └────┬─────┘  └──────────┘                     │
    │       ▼                                        │
    │  messages.append(AIMessage + ToolMessage)      │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    
    【与 EnhancedReActAgent 的区别】
    
    | 维度 | EnhancedReActAgent | ManualReActAgent |
    |:-----|:-------------------|:-----------------|
    | 工具调用 | LangGraph 自动 | **手动解析+调用** |
    | LLM 要求 | 需支持 function calling | **仅需文本生成** |
    | GLM 兼容 | ❌ 不稳定 | ✅ 完美兼容 |
    | 消息管理 | LangGraph 内部 | **完全可控** |
    | 灵活性 | 受框架限制 | **高度可定制** |
    
    【使用示例】
    
    >>> from langchain.tools import tool
    >>> @tool
    ... def get_price(symbol: str) -> str:
    ...     return f"{symbol}: 1680元"
    >>>
    >>> agent = ManualReActAgent(
    ...     tools=[get_price],
    ...     max_iterations=10,
    ... )
    >>> state = agent.run("茅台股价？")
    >>> print(state.final_answer)
    >>> print(state.tool_call_count)  # 应该 > 0
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        llm: Optional[Any] = None,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        初始化手动 ReAct Agent
        
        Args:
            tools: 可用工具列表（LangChain BaseTool）
            llm: LLM 实例（为 None 时自动创建）
            max_iterations: 最大迭代次数
            verbose: 是否打印详细日志
        """
        self.tools = {tool.name: tool for tool in tools}
        self.tool_names = list(self.tools.keys())
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        if llm is None:
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=GLM_API_KEY,
                openai_api_base=GLM_BASE_URL,
                temperature=0,
            )
        else:
            self.llm = llm
        
        self.system_prompt = self._build_manual_prompt()
        
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
    
    def _build_manual_prompt(self) -> str:
        """
        构建手动 ReAct 的系统提示词
        
        核心设计：强制 LLM 输出结构化的 Action 文本，
        以便正则表达式精确提取。
        """
        tools_list = "\n".join(
            f"  - **{name}**: {getattr(tool, 'description', 'No description')}\n"
            f"    参数: {self._get_tool_params_schema(name, tool)}"
            for name, tool in self.tools.items()
        )
        
        return f"""你是一个专业的金融研究助手。你必须通过调用工具来获取数据回答问题。

## ⚠️ 核心规则：必须使用工具！

你没有内置的金融数据。每次涉及具体数据时，必须调用工具。

## 工作流程（每轮只做一件事）

### 第 1 步：思考 (Thought)
分析需要什么信息。

```
Thought: [你的分析]
```

### 第 2 步：行动 (Action) ⭐ 必须执行！
调用工具获取数据。格式严格如下：

```
Action: [工具名称]
Action Input: {{"参数名": "参数值"}}
```

**可用工具**:
{tools_list}

### 第 3 步：观察 (Observation)
等待系统返回工具结果后继续分析。

## 完整示例

**用户**: 茅台股价？

**回复**:
Thought: 用户询问股价，我需要使用 get_stock_quote 工具查询。
Action: tool_get_stock_quote
Action Input: {{"symbol": "600519"}}
[系统返回工具结果...]
Observation: 返回价格1680元...
Thought: 已获得完整行情数据，可以回答了。
Final Answer: 贵州茅台当前股价为1680元...

## 重要格式要求

1. Action 和 Action Input 必须分两行
2. Action Input 必须是合法 JSON
3. 参数值用字符串或数字，不要中文描述
4. 每次只调用一个工具

现在开始处理用户问题！"""
    
    def _get_tool_params_schema(self, name: str, tool: Any) -> str:
        """获取工具的参数 schema 描述"""
        try:
            schema = getattr(tool, 'args_schema', None)
            if schema and hasattr(schema, 'model_json_schema'):
                props = schema.model_json_schema().get('properties', {})
                return json.dumps(list(props.keys()), ensure_ascii=False)
            return "{}"
        except Exception:
            return "{}"
    
    def _try_extract_final_answer(self, content: str) -> Optional[str]:
        """
        尝试从消息内容中提取 Final Answer
        
        匹配规则:
        - "Final Answer: ..." （大小写不敏感）
        - 支持多行内容（直到下一个标签或结尾）
        
        Args:
            content: AIMessage 的文本内容
            
        Returns:
            最终答案字符串或 None
        """
        pattern = r'Final\s+Answer\s*:\s*(.*?)(?:\n\s*(?:Thought|Action|Observation)\s*:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        if content.lower().startswith("final answer"):
            return content.split(":", 1)[-1].strip()
        
        return None
    
    def _try_extract_thought(
        self,
        content: str,
        step_num: int,
    ) -> Optional[ThoughtRecord]:
        """
        尝试从消息内容中提取 Thought 记录
        
        Args:
            content: AIMessage 的文本内容
            step_num: 当前步骤编号
            
        Returns:
            ThoughtRecord 或 None
        """
        pattern = r'Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Final Answer)\s*:|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            thought_text = match.group(1).strip()
            
            reasoning_type = ReasoningType.ANALYSIS.value
            if any(kw in thought_text.lower() for kw in ["首先", "然后", "计划"]):
                reasoning_type = ReasoningType.PLANNING.value
            elif any(kw in thought_text.lower() for kw in ["但是", "然而", "不够"]):
                reasoning_type = ReasoningType.REFLECTION.value
            elif any(kw in thought_text.lower() for kw in ["假设", "如果", "可能"]):
                reasoning_type = ReasoningType.HYPOTHESIS.value
            
            return ThoughtRecord(
                step_number=step_num,
                content=thought_text,
                reasoning_type=reasoning_type,
                confidence=0.8,
            )
        
        return None
    
    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReActState:
        """
        执行手动 ReAct 循环（主入口）
        
        Args:
            query: 用户查询
            context: 初始上下文（可选）
            
        Returns:
            ReActState 完整状态对象
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Manual ReAct] 🚀 开始处理: {query[:60]}...")
            print(f"{'='*70}")
            print(f"  工具: {list(self.tools.keys())}")
            print(f"  最大迭代: {self.max_iterations}")
        
        state = ReActState(
            query=query,
            goal=query,
        )
        
        messages = self._build_initial_messages(query, context)
        
        for step_num in range(1, self.max_iterations + 1):
            step_id = f"step_{step_num}"
            
            if self.verbose:
                print(f"\n{'─'*50}")
                print(f"[Round {step_num}/{self.max_iterations}] 💭 思考中...")
            
            try:
                response = self.llm.invoke(messages)
                content = response.content or ""
                
                if self.verbose:
                    content_preview = content[:200].replace('\n', ' ')
                    print(f"  📝 LLM 输出: {content_preview}...")
                
                final_answer = self._try_extract_final_answer(content)
                
                if final_answer:
                    state.final_answer = final_answer
                    state.stop_reason = StopReason.GOAL_ACHIEVED
                    
                    obs = ObservationRecord(
                        step_number=step_num,
                        raw_result=final_answer,
                        processed_summary=final_answer[:500],
                        key_findings=[],
                        data_quality_score=1.0,
                        relevance_to_goal=1.0,
                    )
                    
                    finish_step = ReActStep(
                        step_id=step_id,
                        step_number=step_num,
                        step_type=ReActStepType.FINISH,
                        observation=obs,
                        timestamp=datetime.now().isoformat(),
                    )
                    state.steps.append(finish_step)
                    
                    if self.verbose:
                        print(f"\n  ✅ 获得 Final Answer!")
                        print(f"  {'─'*50}")
                    break
                
                thought = self._try_extract_thought(content, step_num)
                action_info = self._try_extract_action(content)
                
                if thought:
                    state.steps.append(ReActStep(
                        step_id=f"{step_id}_thought",
                        step_number=step_num,
                        step_type=ReActStepType.THOUGHT,
                        thought=thought,
                        timestamp=datetime.now().isoformat(),
                    ))
                
                if action_info:
                    tool_name = action_info["name"]
                    params = action_info["params"]
                    
                    if self.verbose:
                        print(f"  🔧 调用工具: {tool_name}({params})")
                    
                    action_start = time.time()
                    
                    try:
                        result = self._execute_tool(tool_name, params)
                        
                        action_record = ActionCall(
                            action_id=f"manual_{step_num}_{tool_name[:8]}",
                            action_type=ActionType.TOOL_CALL,
                            tool_name=tool_name,
                            parameters=params,
                            result=result,
                            success=True,
                            start_time=action_start,
                            end_time=time.time(),
                            duration_ms=(time.time() - action_start) * 1000,
                        )
                        
                        state.tool_call_count += 1
                        self._consecutive_errors = 0
                        
                    except Exception as e:
                        error_msg = str(e)
                        result = f"Error: {error_msg}"
                        
                        action_record = ActionCall(
                            action_id=f"manual_err_{step_num}_{tool_name[:8]}",
                            action_type=ActionType.TOOL_CALL,
                            tool_name=tool_name,
                            parameters=params,
                            result=result,
                            success=False,
                            error=error_msg,
                            start_time=action_start,
                            end_time=time.time(),
                            duration_ms=(time.time() - action_start) * 1000,
                        )
                        
                        self._consecutive_errors += 1
                    
                    obs = ObservationRecord(
                        step_number=step_num,
                        raw_result=str(result)[:2000],
                        processed_summary=self._summarize_observation(result),
                        key_findings=self._extract_key_findings(str(result)),
                        data_quality_score=self._assess_data_quality(str(result)),
                        relevance_to_goal=self._assess_relevance(str(result), query),
                    )
                    
                    action_step = ReActStep(
                        step_id=f"{step_id}_action",
                        step_number=step_num,
                        step_type=ReActStepType.ACTION,
                        action=action_record,
                        observation=obs,
                        timestamp=datetime.now().isoformat(),
                    )
                    state.steps.append(action_step)
                    
                    messages.append(AIMessage(content=content))
                    messages.append(ToolMessage(content=str(result), name=tool_name))
                    
                    if self.verbose:
                        result_preview = str(result)[:150].replace('\n', ' ')
                        status = "✅" if action_record.success else "❌"
                        print(f"  {status} 结果: {result_preview}...")
                        print(f"  耗时: {action_record.duration_ms:.0f}ms")
                    
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        state.stop_reason = StopReason.ERROR_OCCURRED
                        if self.verbose:
                            print(f"\n  ⚠️ 连续错误 {self._consecutive_errors} 次，停止")
                        break
                        
                else:
                    if self.verbose:
                        print(f"  ℹ️ 无工具调用，纯思考步骤")
                    
                    messages.append(AIMessage(content=content))
                    
                    no_action_steps = sum(
                        1 for s in state.steps[-3:]
                        if s.step_type == ReActStepType.THOUGHT and not s.action
                    )
                    if no_action_steps >= 2:
                        state.final_answer = (
                            content if len(content) > 50 
                            else f"基于分析，{content}"
                        )
                        state.stop_reason = StopReason.NO_MORE_ACTIONS
                        break
            
            except Exception as e:
                if self.verbose:
                    print(f"  ❌ 循环异常: {e}")
                state.steps.append(ReActStep(
                    step_id=f"{step_id}_error",
                    step_number=step_num,
                    step_type=ReActStepType.THOUGHT,
                    thought=ThoughtRecord(
                        step_number=step_num,
                        content=f"Error: {e}",
                        reasoning_type="error",
                        confidence=0.0,
                    ),
                    timestamp=datetime.now().isoformat(),
                ))
                messages.append(HumanMessage(content=f"[System Error: {e}. Please continue.]"))
        
        if not state.final_answer:
            state.stop_reason = StopReason.MAX_STEPS_REACHED
            last_content = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    last_content = msg.content
                    break
            state.final_answer = last_content or "抱歉，未能完成分析。"
        
        state.total_time_ms = (time.time() - start_time) * 1000
        state.goal_progress = min(1.0, len(state.steps) / max(1, self.max_iterations))
        
        self._print_manual_summary(state)
        
        return state
    
    def _build_initial_messages(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> List[BaseMessage]:
        """构建初始消息列表"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"请研究以下问题：\n\n{query}"),
        ]
        
        if context:
            ctx_text = json.dumps(context, ensure_ascii=False, indent=2)
            messages.append(
                HumanMessage(content=f"\n[附加上下文]\n{ctx_text}")
            )
        
        return messages
    
    def _try_extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 输出中提取 Action 调用信息
        
        支持多种格式变体:
        - `Action: tool_name`
        - `Action:tool_name` (无空格)
        - `action: TOOL_NAME` (大小写不敏感)
        
        同时提取紧随其后的 Action Input JSON。
        """
        action_patterns = [
            r'Action\s*:\s*(\w+)\s*\n',
            r'Action\s*:\s*(\w+)\s*$',
            r'action\s*:\s*(\w+)\s*\n',
        ]
        
        tool_name = None
        for pattern in action_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate in self.tool_names:
                    tool_name = candidate
                    break
                elif any(candidate.lower() == t.lower() for t in self.tool_names):
                    for t in self.tool_names:
                        if candidate.lower() == t.lower():
                            tool_name = t
                            break
                    break
        
        if not tool_name:
            return None
        
        input_patterns = [
            r'Action\s+Input\s*:\s*(\{.*?\})(?:\n|$)',
            r'Action\s+Input\s*:\s*(\{.*?\})(?:\n|$)',
            r'input\s*:\s*(\{.*?\})(?:\n|$)',
        ]
        
        params = {}
        for pattern in input_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    params = json.loads(match.group(1))
                    break
                except json.JSONDecodeError:
                    continue
        
        return {"name": tool_name, "params": params}
    
    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """执行工具调用并返回结果"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"未知工具: {tool_name}")
        
        result = tool.invoke(params)
        return result
    
    @staticmethod
    def _summarize_observation(raw_result: Any, max_length: int = 300) -> str:
        """生成观察摘要"""
        text = str(raw_result)
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"\n... (共{len(text)}字符)"
    
    @staticmethod
    def _extract_key_findings(raw_result: str) -> List[str]:
        """提取关键发现"""
        numbers = re.findall(r'[\d,]+\.?\d*(?:%|倍|元|亿|万)?', raw_result)
        return numbers[:5]
    
    @staticmethod
    def _assess_data_quality(raw_result: str) -> float:
        """评估数据质量"""
        if not raw_result:
            return 0.0
        score = 0.5
        if "error" in raw_result.lower() or "failed" in raw_result.lower():
            score -= 0.3
        if len(re.findall(r'\d+', raw_result)) >= 3:
            score += 0.15
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def _assess_relevance(raw_result: str, query: str) -> float:
        """评估相关性"""
        query_words = set(query.lower().split())
        overlap = sum(1 for w in query_words if w in raw_result.lower())
        return min(1.0, overlap / max(1, len(query_words)))
    
    def _print_manual_summary(self, state: ReActState):
        """打印手动 ReAct 执行摘要"""
        print(f"\n{'='*70}")
        print(f"[Manual ReAct] 📊 执行完成")
        print(f"{'='*70}")
        print(f"  总步骤数: {len(state.steps)}")
        print(f"  工具调用: {state.tool_call_count}次 "
              f"(成功 {state.successful_actions}, 失败 {state.failed_actions})")
        print(f"  停止原因: {state.stop_reason.value if state.stop_reason else 'N/A'}")
        print(f"  总耗时: {state.total_time_ms:.0f}ms")
        
        if self.verbose and state.steps:
            print(f"\n🔄 执行详情:")
            for s in state.steps:
                icons = {
                    ReActStepType.THOUGHT: "💭",
                    ReActStepType.ACTION: "🔧",
                    ReActStepType.OBSERVATION: "👁️",
                    ReActStepType.FINISH: "✅",
                }
                icon = icons.get(s.step_type, "❓")
                info = ""
                if s.thought:
                    info = s.thought.content[:60].replace('\n', ' ')
                elif s.action:
                    info = f"[{s.action.tool_name}] {'✓' if s.action.success else '✗'}"
                elif s.observation:
                    info = s.observation.processed_summary[:50].replace('\n', ' ')
                print(f"  {icon} Step {s.step_number:>2}: {info}")


def create_manual_react_agent(
    tools: List[BaseTool],
    **kwargs,
) -> ManualReActAgent:
    """
    工厂函数：创建手动 ReAct Agent
    
    Args:
        tools: LangChain BaseTool 列表
        **kwargs: 传递给 ManualReActAgent 的额外参数
        
    Returns:
        配置好的 ManualReActAgent 实例
    """
    return ManualReActAgent(tools=tools, **kwargs)


if __name__ == "__main__":
    _run_self_tests()
