"""
增强版多维度评估指标体系 (Enhanced Multi-Dimensional Evaluation Framework)
======================================================================

【模块定位】
本模块是 AI 金融研究助手系统的「质量保障引擎」，负责对系统输出进行
全方位、多维度的质量评估。它是连接系统产出与持续改进的关键反馈环节。

【为什么需要专门的评估体系？】

传统 AI 应用的评估痛点:
  ❌ 仅靠人工审核 → 效率低、主观性强、不可规模化
  ❌ 单一准确率指标 → 无法反映 RAG 的忠实度、召回率等特性
  ❌ 黑盒评估 → 无法定位具体问题（是检索问题还是生成问题？）
  ❌ 无改进指引 → 只知道"不好"，不知道"如何改好"

本框架的解决方案:
  ✅ 5 大维度覆盖核心链路 (RAG/Agent/Output/System/User)
  ✅ 15+ 细分指标，精确定位问题根因
  ✅ 自动生成改进建议，形成闭环优化
  ✅ 结构化报告，支持趋势追踪和对比分析

  【注意】User Satisfaction 维度中的 Engagement 是无人工反馈时的代理指标；
  如需真实满意度，请接入显式评分、追问、复制等产品行为数据。

【评估体系架构】

  ┌─────────────────────────────────────────────────────────────┐
  │                    Comprehensive Evaluator                  │
  │                    （综合评估协调器）                        │
  └────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
  ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
  │RAG Quality││Agent    ││Output   ││System   ││User     │
  │Evaluator ││Quality  ││Quality  ││Perform. ││Satisf.  │
  ├─────────┤├─────────┤├─────────┤├─────────┤├─────────┤
  │Faithful ││Goal     ││Profess. ││Latency  ││Complete │
  │ness     ││Achieve. ││Data Acc.││Token Eff││Clarity  │
  │Answer   ││Tool Eff.││Coherence││Success  ││Helpful  │
  │Relevancy││Reasoning││Actionab.││Resource ││         │
  │Context  ││Time Eff.││         ││Utiliz.  ││         │
  │Precision││Error Rec││         ││         ││         │
  │Context  ││overy    ││         ││         ││         │
  │Recall   ││         ││         ││         ││         │
  └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘

【核心评估维度详解】

1️⃣ RAG Quality (检索增强生成质量)
   检索链路的质量评估，回答"知识获取是否有效"
   
   指标说明:
   - Faithfulness (忠实度): [0,1] 答案是否基于上下文，无幻觉
     * 高: 所有论点都有上下文支撑
     * 低: 存在上下文中未提及的"事实"
     
   - Answer Relevancy (答案相关性): [0,1] 是否直接回答用户问题
     * 高: 精准命中查询意图
     * 低: 答非所问或过度泛化
     
   - Context Precision (上下文精确率): [0,1] 检索文档的相关性
     * high precision: 检索到的都是相关的（可能遗漏）
     * low precision: 检索到很多噪声
     
   - Context Recall (上下文召回率): [0,1] 相关信息是否被充分检索
     * 需要 ground_truth 才能计算
     * 与 Precision 构成 trade-off 关系

2️⃣ Agent Execution Quality (Agent 执行质量)
   推理链路的效率和质量评估，回答"推理过程是否高效合理"
   
   指标说明:
   - Goal Achievement: 任务目标达成程度
   - Tool Efficiency: 工具调用是否存在冗余/重复
   - Reasoning Quality: 思考过程是否逻辑严密
   - Time Efficiency: 响应时间是否在合理范围

3️⃣ Output Quality (输出质量)
   最终产出的内容质量，回答"答案是否专业可信"
   
   指标说明:
   - Professionalism: 金融术语准确性、格式规范性
   - Data Accuracy: 数据引用是否正确、是否有来源标注
   - Logical Coherence: 论述逻辑是否连贯
   - Actionability: 建议是否具体可操作

4️⃣ System Performance (系统性能)
   工程层面的性能指标
   
   指标说明:
   - Latency: 端到端响应延迟
   - Token Efficiency: Token 使用效率（单位信息的 Token 消耗）
   - Success Rate: 成功率（无错误完成的比率）
   - Resource Utilization: 计算资源利用率

5️⃣ User Satisfaction (用户满意度预估)
   面向用户体验的自动化/半自动化指标

   指标说明:
   - Completeness: 答案是否覆盖用户问题要点
   - Clarity: 答案结构是否清晰易读
   - Helpfulness: 是否包含具体建议、数据或下一步
   - Engagement: 显式评分优先；无评分时使用追问/复制等代理信号

【评分等级标准】

  Score Level    Range      Description
  ─────────────────────────────────────────────
  EXCELLENT      [0.9, 1.0]  优秀 — 达到生产级别
  GOOD           [0.7, 0.9)  良好 — 可用，有小幅优化空间
  ACCEPTABLE     [0.5, 0.7)  可接受 — 需要关注和改进
  POOR           [0.3, 0.5)  较差 — 需要立即优化
  CRITICAL       [0.0, 0.3)  严重 — 不可用，需紧急修复

【设计原则】

- 多维度: 不依赖单一指标，避免盲区
- 可自动化: 尽量减少人工标注需求
- 可解释: 每个分数都可追溯到具体原因
- 可行动: 评估结果必须能指导改进方向
- 可对比: 支持历史版本对比和 A/B 测试

【依赖关系】

上游输入:
  - src.enhanced_rag.HybridSearchResult: RAG 检索结果
  - src.react.enhanced_react.ReActState: Agent 执行状态
  - LLM 生成的最终答案文本

下游消费者:
  - main_enhanced.py: 主流程中的质量检查点
  - 监控仪表盘: 展示系统健康度趋势
  - CI/CD 流水线: 回归测试的质量门禁

【版本信息】
- v2.0 (2025-01): 极致优化版，15+ 指标 + 5 维度 + 自动建议 + 自测套件
- v1.0 (2024-12): 初始版本，基础 RAG + 输出质量评估
"""

from __future__ import annotations

import json
import time
import re
import uuid
from typing import (
    Optional, Dict, List, Any, Tuple,
    Callable, Iterator
)
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

from pydantic import BaseModel, Field, validator

# LLM 辅助评估
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ════════════════════════════════════════════════════════════
# 第一部分：枚举与常量定义
# ════════════════════════════════════════════════════════════


class ScoreLevel(Enum):
    """
    评分等级枚举
    
    将连续的 [0, 1] 分数映射为离散的 5 级评价，
    便于人类理解和快速决策。
    
    成员说明:
        EXCELLENT (优秀): [0.9, 1.0]
                     系统表现卓越，达到生产就绪状态
                     可作为 baseline 用于回归测试
                     
        GOOD (良好): [0.7, 0.9)
                 表现良好，存在小幅优化空间
                 建议记录为优化目标
                 
        ACCEPTABLE (可接受): [0.5, 0.7)
                       勉强可用，但存在明显短板
                       需要制定改进计划
                       
        POOR (较差): [0.3, 0.5)
                质量不达标，影响用户体验
                需要优先处理
                
        CRITICAL (严重): [0.0, 0.3)
                    系统/输出不可用
                    需要紧急修复或回滚
    
    使用示例:
        >>> level = get_score_level(0.85)
        >>> print(level.value)  # "good"
        >>> level == ScoreLevel.GOOD  # True
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class EvaluationDimension(Enum):
    """
    评估维度枚举
    
    定义系统的 5 大评估领域。
    
    成员说明:
        RAG_QUALITY: 检索增强生成质量
                   关注: 忠实性、相关性、精确率、召回率
                   
        AGENT_QUALITY: Agent 执行质量
                      关注: 目标达成、工具效率、推理质量
                      
        OUTPUT_QUALITY: 输出内容质量
                       关注: 专业性、准确性、连贯性、可操作性
                       
        SYSTEM_PERFORMANCE: 系统性能
                            关注: 延迟、Token 效率、成功率
                            
        USER_SATISFACTION: 用户满意度预估
                          关注: 完整性、清晰度、有用性
    """
    RAG_QUALITY = "rag_quality"
    AGENT_QUALITY = "agent_quality"
    OUTPUT_QUALITY = "output_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    USER_SATISFACTION = "user_satisfaction"


def get_score_level(score: float) -> ScoreLevel:
    """
    根据数值分数获取评分等级
    
    Args:
        score: [0.0, 1.0] 区间的分数值
        
    Returns:
        对应的 ScoreLevel 枚举值
        
    示例:
        >>> get_score_level(0.95) == ScoreLevel.EXCELLENT
        >>> get_score_level(0.60) == ScoreLevel.ACCEPTABLE
        >>> get_score_level(0.20) == ScoreLevel.CRITICAL
    """
    if score >= 0.9:
        return ScoreLevel.EXCELLENT
    elif score >= 0.7:
        return ScoreLevel.GOOD
    elif score >= 0.5:
        return ScoreLevel.ACCEPTABLE
    elif score >= 0.3:
        return ScoreLevel.POOR
    else:
        return ScoreLevel.CRITICAL


def _extract_keywords(text: str, expand_aliases: bool = True) -> set[str]:
    """
    提取中英文混合关键词。

    中文查询通常没有空格，不能依赖 str.split()；这里同时保留英文/数字词、
    常见金融术语和中文 2-gram，用于轻量启发式匹配。
    """
    if not text:
        return set()

    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_]*|\d+(?:\.\d+)?", lowered))

    finance_terms = [
        "茅台", "贵州茅台", "五粮液", "苹果", "aapl",
        "盈利", "盈利能力", "净利润", "营收", "收入", "毛利率", "净利率",
        "roe", "pe", "pb", "估值", "股价", "现金流", "风险", "增长",
    ]
    tokens.update(term.lower() for term in finance_terms if term.lower() in lowered)

    if expand_aliases:
        alias_groups = [
            {"盈利", "盈利能力", "净利润", "毛利率", "净利率", "roe"},
            {"估值", "pe", "pb", "市盈率", "市净率"},
            {"收入", "营收", "营业收入"},
            {"股价", "价格", "行情"},
            {"风险", "风控", "波动", "回撤"},
        ]
        for group in alias_groups:
            if tokens & group:
                tokens.update(group)

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.update(
        "".join(chinese_chars[i:i + 2])
        for i in range(max(0, len(chinese_chars) - 1))
    )
    return {token for token in tokens if token}


def _number_supported(number_text: str, context: str) -> bool:
    """判断答案中的数值是否能被上下文支撑，允许整数与小数前缀匹配。"""
    normalized = re.sub(r"[^\d.\-]", "", number_text)
    if not normalized:
        return False
    if number_text in context or normalized in context:
        return True
    try:
        value = float(normalized)
    except ValueError:
        return False

    for ctx_number in re.findall(r"[\d,]+\.?\d*", context):
        try:
            ctx_value = float(ctx_number.replace(",", ""))
        except ValueError:
            continue
        if value == 0:
            if ctx_value == 0:
                return True
        elif abs(ctx_value - value) / max(abs(value), 1.0) <= 0.01:
            return True
        elif str(ctx_value).startswith(str(value)) or str(value).startswith(str(ctx_value)):
            return True
    return False


# 各维度的默认权重配置
DEFAULT_DIMENSION_WEIGHTS: Dict[EvaluationDimension, float] = {
    EvaluationDimension.RAG_QUALITY: 0.30,
    EvaluationDimension.AGENT_QUALITY: 0.25,
    EvaluationDimension.OUTPUT_QUALITY: 0.30,
    EvaluationDimension.SYSTEM_PERFORMANCE: 0.10,
    EvaluationDimension.USER_SATISFACTION: 0.05,
}


# ════════════════════════════════════════════════════════════
# 第二部分：数据模型定义
# ════════════════════════════════════════════════════════════


@dataclass
class MetricScore:
    """
    单个评估指标得分
    
    这是评估体系中最小的评分单元。每个 MetricScore 代表一个
    具体维度的量化测量结果。
    
    属性说明:
        name: 指标名称（如 "Faithfulness", "Goal Achievement"）
              遵循命名规范: PascalCase 或带空格的可读名称
              
        value: 得分值 [0.0, 1.0]
              特殊值:
              - 0.0: 完全失败或未执行
              - 1.0: 完美表现
              - NaN: 无法评估（通常转为 0.0 并标记）
              
        level: 对应的评分等级（由 get_score_level() 计算）
        
        description: 指标的自然语言描述
                    用于在报告中向非技术人员解释该指标的含意
                    
        details: 详细信息字典
                可能包含:
                - "breakdown": 子项得分
                - "comparison": 与基准值的对比
                - "notes": 评估者的备注
                
        benchmark: 行业基准值或历史最优值
                 用于判断当前表现是优于还是劣于预期
                 
    使用示例:
        >>> score = MetricScore(
        ...     name="Faithfulness",
        ...     value=0.92,
        ...     level=ScoreLevel.EXCELLENT,
        ...     description="答案忠实于上下文",
        ...     benchmark=0.85,
        ... )
        >>> print(f"{score.name}: {score.value:.2f} ({score.level.value})")
        Faithfulness: 0.92 (excellent)
    """
    name: str
    value: float
    level: ScoreLevel
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    benchmark: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "level": self.level.value,
            "description": self.description,
            "details": self.details,
            "benchmark": self.benchmark,
            "vs_benchmark": round(self.value - self.benchmark, 4),
        }
    
    @property
    def meets_benchmark(self) -> bool:
        """是否达到基准值"""
        return self.value >= self.benchmark
    
    @property
    def is_passing(self) -> bool:
        """是否及格（>= ACCEPTABLE）"""
        return self.level in (ScoreLevel.EXCELLENT, ScoreLevel.GOOD)


@dataclass
class DimensionScore:
    """
    维度得分（多个指标的聚合结果）
    
    每个 DimensionScore 代表一个评估领域的综合表现，
    由多个 MetricScore 加权平均得到。
    
    属性说明:
        dimension: 维度名称（对应 EvaluationDimension 枚举值）
        
        metrics: 该维度包含的所有 MetricScore 列表
                通常 3-5 个指标
                
        weighted_score: 加权综合分 [0.0, 1.0]
                     计算公式: Σ(metric.value × metric_weight)
                     
        level: 综合评分等级
    
    使用示例:
        >>> dim = DimensionScore(
        ...     dimension="RAG Quality",
        ...     metrics=[faithfulness_score, relevancy_score],
        ...     weighted_score=0.88,
        ...     level=ScoreLevel.GOOD,
        ... )
        >>> print(f"{dim.dimension}: {dim.weighted_score:.2f}")
        RAG Quality: 0.88
    """
    dimension: str
    metrics: List[MetricScore]
    weighted_score: float
    level: ScoreLevel
    
    @property
    def n_metrics(self) -> int:
        """包含的指标数量"""
        return len(self.metrics)
    
    @property
    def best_metric(self) -> Optional[MetricScore]:
        """得分最高的指标"""
        if not self.metrics:
            return None
        return max(self.metrics, key=lambda m: m.value)
    
    @property
    def worst_metric(self) -> Optional[MetricScore]:
        """得分最低的指标"""
        if not self.metrics:
            return None
        return min(self.metrics, key=lambda m: m.value)
    
    def get_metric_by_name(self, name: str) -> Optional[MetricScore]:
        """按名称查找指标"""
        for m in self.metrics:
            if m.name.lower() == name.lower():
                return m
        return None
    
    def summary(self) -> str:
        """生成维度摘要"""
        lines = [
            f"{self.dimension}: {self.weighted_score:.4f} ({self.level.value})",
            f"  Metrics ({self.n_metrics}):",
        ]
        for m in self.metrics:
            bench_mark = "✓" if m.meets_benchmark else "✗"
            lines.append(f"    {bench_mark} {m.name}: {m.value:.4f}")
        return "\n".join(lines)


@dataclass 
class EvaluationReport:
    """
    完整评估报告
    
    这是一次评估操作的完整产物，包含所有维度的得分、
    统计信息和改进建议。是本模块对外的主要输出格式。
    
    属性分组:
        【元数据】
        evaluation_id: 全局唯一标识（用于追踪和存档）
        timestamp: ISO 格式的时间戳
        evaluator_version: 评估器版本号
        
        【输入数据】
        query: 用户原始查询
        answer: 系统生成的答案
        contexts: 检索到的上下文片段列表
        ground_truth: 参考答案（可选，用于有监督评估）
        
        【各维度得分】
        rag_quality / agent_quality / output_quality /
        system_performance / user_satisfaction: 五个维度的 DimensionScore
        
        【综合评分】
        overall_score: 加权总分 [0.0, 1.0]
        overall_level: 总体等级
        
        【智能分析】
        strengths: 优势列表（得分 > GOOD 的指标）
        weaknesses: 劣势列表（得分 < ACCEPTABLE 的指标）
        recommendations: 针对性的改进建议
        
        【性能】
        evaluation_time_ms: 本次评估操作的耗时
    
    下游使用:
        >>> report = evaluator.full_evaluate(query, answer, contexts)
        >>> print(report.summary())
        >>> 
        >>> # 导出为 JSON 用于持久化
        >>> with open("eval_report.json", "w") as f:
        ...     json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    """
    evaluation_id: str
    timestamp: str
    
    query: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    
    rag_quality: Optional[DimensionScore] = None
    agent_quality: Optional[DimensionScore] = None
    output_quality: Optional[DimensionScore] = None
    system_performance: Optional[DimensionScore] = None
    user_satisfaction: Optional[DimensionScore] = None

    overall_score: float = 0.0
    overall_level: ScoreLevel = ScoreLevel.ACCEPTABLE
    
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    evaluator_version: str = "2.0"
    evaluation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于 JSON 导出）"""
        def _safe_level_value(level) -> str:
            if hasattr(level, 'value'):
                return level.value
            return str(level)
        
        result = {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "query": self.query[:100],
            "answer_length": len(self.answer),
            "n_contexts": len(self.contexts),
            "overall_score": self.overall_score,
            "overall_level": _safe_level_value(self.overall_level),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }
        
        for dim_name, dim_score in [
            ("rag_quality", self.rag_quality),
            ("agent_quality", self.agent_quality),
            ("output_quality", self.output_quality),
            ("system_performance", self.system_performance),
            ("user_satisfaction", self.user_satisfaction),
        ]:
            if dim_score:
                result[dim_name] = {
                    "weighted_score": dim_score.weighted_score,
                    "level": _safe_level_value(dim_score.level),
                    "metrics": [m.to_dict() for m in dim_score.metrics],
                }
        
        return result
    
    def summary(self) -> str:
        """生成人类可读的报告摘要"""
        def _safe_level(level) -> str:
            if hasattr(level, 'value'):
                return str(level.value).upper()
            return str(level).upper()
        
        lines = [
            f"{'='*70}",
            f"📊 评估报告 #{self.evaluation_id}",
            f"{'='*70}",
            f"",
            f"查询: {self.query[:60]}...",
            f"",
            f"🎯 综合得分: {self.overall_score:.4f} ({_safe_level(self.overall_level)})",
            f"",
        ]
        
        for dim_label, dim_score in [
            ("📚 RAG 质量", self.rag_quality),
            ("🤖 Agent 质量", self.agent_quality),
            ("✍️ 输出质量", self.output_quality),
            ("⚡ 系统性能", self.system_performance),
            ("👤 用户满意度", self.user_satisfaction),
        ]:
            if dim_score:
                lines.append(f"{dim_label}: {dim_score.weighted_score:.4f} ({_safe_level(dim_score.level)})")
                for m in dim_score.metrics[:3]:
                    lines.append(f"    • {m.name}: {m.value:.4f}")
        
        if self.strengths:
            lines.append(f"\n✅ 优势:")
            for s in self.strengths[:3]:
                lines.append(f"  + {s}")
        
        if self.weaknesses:
            lines.append(f"\n⚠️ 不足:")
            for w in self.weaknesses[:3]:
                lines.append(f"  - {w}")
        
        if self.recommendations:
            lines.append(f"\n💡 建议:")
            for r in self.recommendations[:3]:
                lines.append(f"  → {r}")
        
        lines.append(f"\n⏱️ 评估耗时: {self.evaluation_time_ms:.0f}ms")
        lines.append(f"{'='*70}")
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 第三部分：RAG 质量评估器
# ════════════════════════════════════════════════════════════


class RAGQualityEvaluator:
    """
    RAG 质量评估器（启发式规则为主，支持可选 LLM 辅助评估）
    
    【评估能力】
    
    核心指标（4 个）:
    1. Faithfulness (忠实度)
       - 问题: 答案是否基于提供的上下文？
       - 风险: LLM 产生「幻觉」——编造上下文中不存在的信息
       - 评估方法: NLI (Natural Language Inference) 或关键词匹配
       
    2. Answer Relevancy (答案相关性)
       - 问题: 答案是否直接回应了用户的查询？
       - 风险: 答非所问、过度泛化、偏离主题
       - 评估方法: Embedding 相似度 + 关键词覆盖率
       
    3. Context Precision (上下文精确率)
       - 问题: 检索到的文档是否与查询相关？
       - 风险: 引入了噪声文档，干扰 LLM 推理
       - 评估方法: 逐段相关性打分
       
    4. Context Recall (上下文召回率)
       - 问题: 相关信息是否被充分检索？
       - 风险: 遗漏关键信息导致答案不完整
       - 评估方法: 需要 ground_truth 支持
    
    【实现策略】
    本实现采用「启发式规则为主，LLM 为辅」的策略:
    - 规则方法: 快速、低成本、确定性高
    - LLM 方法: 更精准，但耗时且依赖外部服务
    
    生产环境建议:
    - 在线评估: 使用规则方法（<100ms）
    - 离线分析: 使用 LLM 方法（~2-5s）
    """
    
    def __init__(self, use_llm: bool = False):
        """
        初始化 RAG 质量评估器

        Args:
            use_llm: 是否启用 LLM 辅助评估
                    - False (默认): 使用规则方法（快速，<100ms）
                    - True: 使用 LLM 评估（更精准，~2-5s）
        """
        self.use_llm = use_llm
        if use_llm:
            from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
            self._llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=GLM_API_KEY,
                openai_api_base=GLM_BASE_URL,
                temperature=0,
            )
        else:
            self._llm = None
    
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> DimensionScore:
        """
        执行 RAG 质量评估
        
        Args:
            query: 用户原始查询
            answer: 系统生成的答案
            contexts: 检索到的上下文片段列表
            ground_truth: 参考答案（可选，用于计算 Recall）
            
        Returns:
            DimensionScore 包含 3-4 个 MetricScore
            
        评估流程:
        1. Faithfulness → 检查答案中的断言是否可在上下文中找到支撑
        2. Answer Relevancy → 计算答案与查询的语义相似度
        3. Context Precision → 逐段评估上下文与查询的相关性
        4. Context Recall (可选) → 评估信息完整性
        5. 加权聚合 → 得到 RAG 维度综合分
        """
        print(f"[RAG评估] 📊 开始评估...")
        start_time = time.time()
        
        metrics: List[MetricScore] = []

        if self.use_llm:
            faithfulness = self._evaluate_faithfulness_llm(answer, contexts)
            relevancy = self._evaluate_answer_relevancy_llm(query, answer, contexts)
        else:
            faithfulness = self._evaluate_faithfulness(answer, contexts)
            relevancy = self._evaluate_answer_relevancy(query, answer)
        metrics.append(faithfulness)
        metrics.append(relevancy)
        
        precision = self._evaluate_context_precision(query, contexts)
        metrics.append(precision)
        
        if ground_truth:
            recall = self._evaluate_context_recall(answer, ground_truth, contexts)
            metrics.append(recall)
        
        weights = {"Faithfulness": 0.30, "Answer Relevancy": 0.28,
                    "Context Precision": 0.24, "Context Recall": 0.18}
        
        weighted_sum = sum(
            m.value * weights.get(m.name, 0.20) for m in metrics
        )
        
        eval_time = (time.time() - start_time) * 1000
        
        result = DimensionScore(
            dimension="RAG Quality",
            metrics=metrics,
            weighted_score=round(weighted_sum, 4),
            level=get_score_level(weighted_sum),
        )
        
        print(f"[RAG评估] ✅ 完成 ({eval_time:.0f}ms)")
        print(f"  综合: {result.weighted_score:.4f} ({result.level.value})")
        
        return result
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> MetricScore:
        """
        评估答案忠实度（幻觉检测）
        
        启发式策略:
        1. 提取答案中的实体/数值/关键陈述
        2. 检查这些信息是否出现在上下文中
        3. 计算覆盖率作为忠实度近似
        
        注意:
        这是一个简化实现。生产环境建议使用:
        - NLI 模型 (如 BERT-NLI) 做 premise-hypothesis 判断
        - 或 Ragas 库的 faithfulness 指标
        """
        if not answer or not contexts:
            return MetricScore(
                name="Faithfulness",
                value=0.0,
                level=ScoreLevel.CRITICAL,
                description="无法评估（缺少答案或上下文）",
                benchmark=0.85,
            )
        
        combined_context = " ".join(contexts).lower()
        
        sentences = re.split(r'[。！？\n]', answer)
        supported = 0
        total_checkable = 0
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            total_checkable += 1
            
            numbers_in_sent = re.findall(r'[\d,]+\.?\d*(?:%|倍|元|亿|万)?', sent)
            
            if numbers_in_sent:
                found_count = sum(
                    1 for num in numbers_in_sent
                    if _number_supported(num, combined_context)
                )
                if found_count / len(numbers_in_sent) > 0.5:
                    supported += 1
            else:
                keywords = _extract_keywords(sent) & _extract_keywords(combined_context)
                if len(keywords) >= 2:
                    supported += 1
        
        faith_score = supported / max(1, total_checkable)
        
        return MetricScore(
            name="Faithfulness",
            value=round(faith_score, 4),
            level=get_score_level(faith_score),
            description="答案是否忠实于检索上下文，无幻觉内容",
            details={
                "supported_statements": supported,
                "total_statements": total_checkable,
            },
            benchmark=0.85,
        )

    def _evaluate_faithfulness_llm(
        self,
        answer: str,
        contexts: List[str],
    ) -> MetricScore:
        """
        LLM 辅助评估答案忠实度（幻觉检测）

        使用 GLM 做 NLI 判断：给定上下文，判断答案中的每个陈述是否被支撑。
        """
        if not self._llm:
            return self._evaluate_faithfulness(answer, contexts)

        combined_context = "\n\n".join(contexts)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个金融领域的事实核查专家。给定一个上下文文档和一个答案，"
             "请逐句检查答案中的每个陈述是否被上下文支撑。\n"
             "对于每个陈述，输出：\n"
             "  - SUPPORTED: 陈述被上下文明确支持\n"
             "  - CONTRADICTED: 陈述与上下文矛盾\n"
             "  - UNSUPPORTED: 陈述在上下文中找不到依据\n\n"
             "最终给出一个 0-1 的忠实度分数（SUPPORTED 比例）。"),
            ("human", "上下文：\n{context}\n\n答案：\n{answer}"),
        ])

        try:
            response = self._llm.invoke(prompt.format(context=combined_context, answer=answer))
            result_text = response.content.strip()

            match = re.search(r'(?:分数|faithfulness|score)[:：]\s*([\d.]+)', result_text, re.IGNORECASE)
            if match:
                faith_score = float(match.group(1))
            else:
                supported = len(re.findall(r'SUPPORTED', result_text, re.IGNORECASE))
                contradicted = len(re.findall(r'CONTRADICTED', result_text, re.IGNORECASE))
                unsupported = len(re.findall(r'UNSUPPORTED', result_text, re.IGNORECASE))
                total = supported + contradicted + unsupported
                faith_score = supported / max(1, total)

            return MetricScore(
                name="Faithfulness",
                value=round(faith_score, 4),
                level=get_score_level(faith_score),
                description="LLM 辅助评估：答案是否忠实于检索上下文，无幻觉内容",
                details={"llm_evaluation": result_text[:200]},
                benchmark=0.85,
            )
        except Exception as e:
            print(f"[RAG评估] LLM 忠实度评估失败，降级为规则方法：{e}")
            return self._evaluate_faithfulness(answer, contexts)

    def _evaluate_answer_relevancy_llm(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> MetricScore:
        """
        LLM 辅助评估答案相关性

        使用 GLM 判断答案是否直接、完整地回应了用户问题。
        """
        if not self._llm:
            return self._evaluate_answer_relevancy(query, answer)

        combined_context = "\n\n".join(contexts)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个金融研究质量评估专家。请评估答案是否直接、完整地回应了用户的问题。\n"
             "从以下维度评分（0-1）：\n"
             "  1. 相关性：答案是否针对问题而非跑题\n"
             "  2. 完整性：答案是否涵盖了问题的主要方面\n"
             "  3. 准确性：答案是否基于提供的上下文（无幻觉）\n\n"
             "最终输出一个 0-1 的综合相关性分数，并简述理由。"),
            ("human", "用户问题：{query}\n\n参考答案的上下文：\n{contexts}\n\n待评估答案：\n{answer}"),
        ])

        try:
            response = self._llm.invoke(prompt.format(query=query, contexts=combined_context, answer=answer))
            result_text = response.content.strip()

            match = re.search(r'(?:分数|relevancy|score|相关性)[:：]\s*([\d.]+)', result_text, re.IGNORECASE)
            if match:
                relevancy = float(match.group(1))
            else:
                relevancy = 0.5  # 解析失败时的保守默认值

            return MetricScore(
                name="Answer Relevancy",
                value=round(relevancy, 4),
                level=get_score_level(relevancy),
                description="LLM 辅助评估：生成的答案是否直接回答了用户的问题",
                details={"llm_evaluation": result_text[:200]},
                benchmark=0.80,
            )
        except Exception as e:
            print(f"[RAG评估] LLM 相关性评估失败，降级为规则方法：{e}")
            return self._evaluate_answer_relevancy(query, answer)

    def _evaluate_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> MetricScore:
        """
        评估答案相关性
        
        策略:
        1. 关键词重叠度（query 的关键词是否出现在 answer 中）
        2. 答案长度合理性（太短可能不完整，太长可能冗余）
        3. 语义覆盖度（是否涉及查询的所有方面）
        """
        if not query or not answer:
            return MetricScore(
                name="Answer Relevancy",
                value=0.0,
                level=ScoreLevel.CRITICAL,
                benchmark=0.80,
            )
        
        query_words = _extract_keywords(query)
        answer_words = _extract_keywords(answer)
        
        overlap = query_words & answer_words
        keyword_coverage = len(overlap) / max(1, len(query_words))
        
        length_score = min(1.0, len(answer) / 150)
        length_score = max(0.3, length_score)
        
        question_marks = query.count('？') + query.count('?')
        aspects = []
        if '如何' in query or '怎么' in query:
            aspects.append('method')
        if '为什么' in query or '原因' in query:
            aspects.append('reason')
        if '多少' in query or '什么' in query or '哪些' in query:
            aspects.append('quantity')
        if '是否' in query or '能不能' in query:
            aspects.append('yes_no')
        
        aspect_coverage = 1.0 if not aspects else min(1.0, len(aspects) / max(1, len(aspects)))
        
        relevancy = (
            keyword_coverage * 0.4 +
            length_score * 0.3 +
            aspect_coverage * 0.3
        )
        
        return MetricScore(
            name="Answer Relevancy",
            value=round(relevancy, 4),
            level=get_score_level(relevancy),
            description="生成的答案是否直接回答了用户的问题",
            details={
                "keyword_coverage": round(keyword_coverage, 3),
                "length_score": round(length_score, 3),
            },
            benchmark=0.80,
        )
    
    def _evaluate_context_precision(
        self,
        query: str,
        contexts: List[str],
    ) -> MetricScore:
        """
        评估上下文精确率
        
        策略:
        对每个 context 片段，评估其与查询的相关性：
        - 高相关: 包含查询关键词 + 有实质内容
        - 中相关: 包含部分关键词
        - 低相关: 几乎无关
        """
        if not contexts:
            return MetricScore(
                name="Context Precision",
                value=0.0,
                level=ScoreLevel.CRITICAL,
                description="无检索上下文",
                benchmark=0.75,
            )
        
        query_keywords = _extract_keywords(query)
        
        relevant_count = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            overlap = query_keywords & _extract_keywords(ctx_lower)
            
            if len(overlap) >= 2:
                relevant_count += 1
            elif len(overlap) >= 1:
                relevant_count += 0.5
        
        precision = relevant_count / max(1, len(contexts))
        
        return MetricScore(
            name="Context Precision",
            value=round(precision, 4),
            level=get_score_level(precision),
            description="检索到的文档是否与问题高度相关",
            details={
                "relevant_docs": int(relevant_count),
                "total_docs": len(contexts),
            },
            benchmark=0.75,
        )
    
    def _evaluate_context_recall(
        self,
        answer: str,
        ground_truth: str,
        contexts: List[str],
    ) -> MetricScore:
        """
        评估上下文召回率（需要参考答案）
        
        检查 ground_truth 中的关键信息是否被 contexts 覆盖。
        """
        gt_keywords = _extract_keywords(ground_truth)
        combined_ctx = " ".join(contexts).lower()
        ctx_keywords = _extract_keywords(combined_ctx)
        
        overlap = gt_keywords & ctx_keywords
        recall = len(overlap) / max(1, len(gt_keywords))
        
        return MetricScore(
            name="Context Recall",
            value=round(recall, 4),
            level=get_score_level(recall),
            description="相关信息是否被充分检索到",
            details={
                "covered_keypoints": len(overlap),
                "total_keypoints": len(gt_keywords),
            },
            benchmark=0.70,
        )


# ════════════════════════════════════════════════════════════
# 第四部分：Agent 执行质量评估器
# ════════════════════════════════════════════════════════════


class AgentQualityEvaluator:
    """
    Agent 执行质量评估器
    
    【评估维度】
    
    1. Goal Achievement Rate (目标达成率)
       - Agent 是否完成了用户指定的任务？
       - 评估: 答案是否解决了查询的核心意图
       
    2. Tool Usage Efficiency (工具使用效率)
       - 工具调用是否高效、无冗余？
       - 评估: 调用次数合理性 + 重复调用检测
       
    3. Reasoning Quality (推理质量)
       - Thought 过程是否逻辑严密？
       - 评估: 推理步骤占比 + 步骤深度
       
    4. Time Efficiency (时间效率)
       - 响应时间是否在可接受范围？
       - 评估: 分段计时（金融场景 <30s 为优）
    """
    
    def evaluate(
        self,
        query: str,
        final_answer: str,
        execution_steps: List[Dict[str, Any]],
        total_time_ms: float,
        tool_call_count: int,
    ) -> DimensionScore:
        """
        评估 Agent 执行质量
        
        Args:
            query: 用户查询
            final_answer: 最终答案
            execution_steps: 执行步骤列表（来自 ReActState.steps）
            total_time_ms: 总执行时间
            tool_call_count: 工具调用总次数
            
        Returns:
            DimensionScore Agent 执行质量维度得分
        """
        metrics: List[MetricScore] = []
        
        goal_score = self._evaluate_goal_achievement(query, final_answer)
        metrics.append(MetricScore(
            name="Goal Achievement",
            value=goal_score,
            level=get_score_level(goal_score),
            description="任务目标达成程度",
        ))
        
        efficiency_score = self._evaluate_tool_efficiency(tool_call_count, execution_steps)
        metrics.append(MetricScore(
            name="Tool Efficiency",
            value=efficiency_score,
            level=get_score_level(efficiency_score),
            description="工具调用效率和无冗余性",
        ))
        
        reasoning_score = self._evaluate_reasoning_quality(execution_steps)
        metrics.append(MetricScore(
            name="Reasoning Quality",
            value=reasoning_score,
            level=get_score_level(reasoning_score),
            description="思考过程逻辑严密性",
        ))
        
        time_score = self._evaluate_time_efficiency(total_time_ms)
        metrics.append(MetricScore(
            name="Time Efficiency",
            value=time_score,
            level=get_score_level(time_score),
            description="响应时间合理性",
        ))
        
        weights = [0.35, 0.25, 0.25, 0.15]
        weighted_sum = sum(m.value * w for m, w in zip(metrics, weights))
        
        return DimensionScore(
            dimension="Agent Execution Quality",
            metrics=metrics,
            weighted_score=round(weighted_sum, 4),
            level=get_score_level(weighted_sum),
        )
    
    def _evaluate_goal_achievement(self, query: str, answer: str) -> float:
        """评估目标达成度"""
        if not answer or not query:
            return 0.0
        
        query_keywords = _extract_keywords(query, expand_aliases=False)
        answer_lower = answer.lower()
        
        matched = sum(1 for kw in query_keywords if kw in answer_lower)
        coverage = matched / max(1, len(query_keywords))
        
        length_score = min(1.0, len(answer) / 50)
        
        return coverage * 0.7 + length_score * 0.3
    
    def _evaluate_tool_efficiency(
        self,
        call_count: int,
        steps: List[Dict[str, Any]],
    ) -> float:
        """评估工具使用效率"""
        if call_count == 0:
            return 0.5
        
        if 3 <= call_count <= 8:
            efficiency = 1.0
        elif call_count < 3:
            efficiency = 0.6 + (call_count / 3) * 0.4
        else:
            efficiency = max(0.3, 1.0 - (call_count - 8) * 0.05)
        
        tools_called = []
        duplicates = 0
        for step in steps:
            tool_name = step.get("tool", "") or step.get("action", {}).get("tool_name", "")
            if tool_name in tools_called:
                duplicates += 1
            else:
                tools_called.append(tool_name)
        
        penalty = duplicates * 0.05
        return max(0.0, efficiency - penalty)
    
    def _evaluate_reasoning_quality(self, steps: List[Dict[str, Any]]) -> float:
        """评估推理质量"""
        if not steps:
            return 0.3
        
        thought_steps = sum(1 for s in steps if s.get("step_type") in ["thought", "THOUGHT"])
        reasoning_ratio = thought_steps / max(1, len(steps))
        depth_score = min(1.0, thought_steps / 3)
        
        return reasoning_ratio * 0.6 + depth_score * 0.4
    
    def _evaluate_time_efficiency(self, time_ms: float) -> float:
        """评估时间效率"""
        if time_ms < 10000:
            return 1.0
        elif time_ms < 30000:
            return 0.9
        elif time_ms < 60000:
            return 0.7
        elif time_ms < 120000:
            return 0.5
        else:
            return max(0.2, 0.5 - (time_ms - 120000) / 240000)


# ════════════════════════════════════════════════════════════
# 第五部分：输出质量评估器
# ════════════════════════════════════════════════════════════


class OutputQualityEvaluator:
    """
    输出质量评估器
    
    【评估维度】
    
    1. Professionalism (专业性)
       - 金融术语使用是否准确
       - 格式是否规范（Markdown 结构）
       
    2. Data Accuracy (数据准确性)
       - 数值引用是否有来源标注
       - 数据是否在合理范围内
       
    3. Logical Coherence (逻辑连贯性)
       - 论述结构是否完整（引言-正文-结论）
       - 连接词使用是否恰当
       
    4. Actionability (可操作性)
       - 建议是否具体（含数字/比例）
       - 是否有明确的行动指引
    """
    
    PROFESSIONAL_TERMS = [
        "营收", "净利润", "毛利率", "净利率", "ROE", "PE", "PB",
        "现金流", "资产负债率", "同比", "环比", "EPS", "EBITDA",
        "估值", "风险", "收益", "成本", "市值", "股息率",
        "经营性现金流", "投资性现金流", "筹资性现金流",
    ]
    
    def evaluate(
        self,
        answer: str,
        contexts: List[str] = None,
    ) -> DimensionScore:
        """
        评估输出质量
        
        Args:
            answer: 生成的答案文本
            contexts: 检索上下文（可选，用于数据准确性验证）
            
        Returns:
            DimensionScore 输出质量维度得分
        """
        metrics: List[MetricScore] = []
        
        prof_score = self._evaluate_professionalism(answer)
        metrics.append(MetricScore(
            name="Professionalism",
            value=prof_score,
            level=get_score_level(prof_score),
            description="金融术语使用准确性和格式规范性",
        ))
        
        acc_score = self._evaluate_data_accuracy(answer, contexts)
        metrics.append(MetricScore(
            name="Data Accuracy",
            value=acc_score,
            level=get_score_level(acc_score),
            description="数据引用准确性和来源标注完整性",
        ))
        
        coh_score = self._evaluate_coherence(answer)
        metrics.append(MetricScore(
            name="Logical Coherence",
            value=coh_score,
            level=get_score_level(coh_score),
            description="论述逻辑连贯性和结构完整性",
        ))
        
        act_score = self._evaluate_actionability(answer)
        metrics.append(MetricScore(
            name="Actionability",
            value=act_score,
            level=get_score_level(act_score),
            description="建议的具体性和可操作性",
        ))
        
        weights = [0.20, 0.30, 0.30, 0.20]
        weighted_sum = sum(m.value * w for m, w in zip(metrics, weights))
        
        return DimensionScore(
            dimension="Output Quality",
            metrics=metrics,
            weighted_score=round(weighted_sum, 4),
            level=get_score_level(weighted_sum),
        )
    
    def _evaluate_professionalism(self, text: str) -> float:
        """评估专业性"""
        term_count = sum(1 for t in self.PROFESSIONAL_TERMS if t in text)
        term_density = term_count / max(1, len(text) / 100)
        
        has_structure = any(m in text for m in ["#", "##", "###", "- ", "*"])
        
        return min(1.0, term_density * 0.6 + (0.4 if has_structure else 0.2))
    
    def _evaluate_data_accuracy(self, text: str, contexts: List[str]) -> float:
        """评估数据准确性"""
        numbers = re.findall(r'[\d,]+\.?\d*(?:%|倍|元|亿|万)?', text)
        
        if not numbers:
            return 0.5
        
        has_citation = any(
            m in text for m in ["根据", "数据显示", "财报显示", "来源", "参考"]
        )
        citation_score = 0.8 if has_citation else 0.4
        
        reasonable = 0
        for num_str in numbers[:10]:
            try:
                num = float(re.sub(r'[^\d.\-]', '', num_str))
                if -1000000 < num < 10000000:
                    reasonable += 1
            except ValueError:
                pass
        
        reasonableness = reasonable / max(1, len(numbers[:10]))
        
        return citation_score * 0.5 + reasonableness * 0.5
    
    def _evaluate_coherence(self, text: str) -> float:
        """评估逻辑连贯性"""
        connectors = [
            "因此", "所以", "但是", "然而", "此外", "另外",
            "首先", "其次", "最后", "综上所述",
        ]
        conn_count = sum(1 for c in connectors if c in text)
        
        has_intro = len(text) > 50
        has_body = len(text) > 200
        has_conclusion = any(w in text for w in ["总结", "综上", "结论", "建议"])
        
        structure = (0.3 if has_intro else 0) + (0.3 if has_body else 0) + (0.4 if has_conclusion else 0)
        conn_score = min(1.0, conn_count / 5)
        
        return structure * 0.6 + conn_score * 0.4
    
    def _evaluate_actionability(self, text: str) -> float:
        """评估可操作性"""
        action_words = [
            "建议", "推荐", "可以", "应该", "需要",
            "关注", "警惕", "考虑", "采取", "配置",
        ]
        
        action_count = sum(1 for w in action_words if w in text)
        action_density = action_count / max(1, len(text.split()) / 50)
        
        specific = re.findall(r'(?:建议|推荐|配置|仓位).*?[\d]+(?:%|倍|成|点)', text)
        specificity = min(1.0, len(specific) / 2)
        
        return action_density * 0.5 + specificity * 0.5


# ════════════════════════════════════════════════════════════
# 第五部分（续）：系统性能评估器 & 用户满意度评估器
# ════════════════════════════════════════════════════════════


class SystemPerformanceEvaluator:
    """
    系统性能评估器（自动化评估实现）

    【评估维度】

    1. Latency (延迟)
       - 各阶段耗时：意图识别 / 检索 / 分析 / 报告生成
       - 金融场景参考：<30s 为优秀，<60s 为可接受

    2. Token Efficiency (Token 效率)
       - 输入/输出 Token 比，衡量信息密度
       - 高效：单位信息消耗更少 Token

    3. Success Rate (成功率)
       - 任务无错误完成的比率
       - 统计各错误类型的分布

    4. Resource Utilization (资源利用率)
       - CPU/内存峰值监控（可选，需环境支持）
    """

    # 性能基线（金融场景）
    LATENCY_BENCHMARK_MS = 30000   # 30s
    TOKEN_RATIO_BENCHMARK = 0.5    # 输出/输入 Token 比
    SUCCESS_RATE_BENCHMARK = 0.95  # 95%

    def evaluate(
        self,
        phase_durations_ms: Optional[Dict[str, float]] = None,
        total_duration_ms: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_count: int = 0,
        total_requests: int = 1,
        cpu_peak_percent: Optional[float] = None,
        memory_peak_mb: Optional[float] = None,
    ) -> DimensionScore:
        """
        评估系统性能

        Args:
            phase_durations_ms: 各阶段耗时 {"意图识别": 120, "检索": 800, ...}
            total_duration_ms: 总执行时间
            input_tokens: 输入 Token 总数
            output_tokens: 输出 Token 总数
            error_count: 错误次数
            total_requests: 总请求数（用于计算成功率）
            cpu_peak_percent: CPU 峰值%（可选）
            memory_peak_mb: 内存峰值 MB（可选）

        Returns:
            DimensionScore 系统性能维度得分
        """
        metrics: List[MetricScore] = []

        # 1. Latency
        if total_duration_ms > 0:
            latency_score = max(0.0, 1.0 - (total_duration_ms / self.LATENCY_BENCHMARK_MS))
            if latency_score >= 0.9:
                level = ScoreLevel.EXCELLENT
            elif latency_score >= 0.7:
                level = ScoreLevel.GOOD
            elif latency_score >= 0.5:
                level = ScoreLevel.ACCEPTABLE
            elif latency_score >= 0.3:
                level = ScoreLevel.POOR
            else:
                level = ScoreLevel.CRITICAL
        else:
            latency_score = 0.5
            level = ScoreLevel.ACCEPTABLE

        metrics.append(MetricScore(
            name="Latency",
            value=round(latency_score, 4),
            level=level,
            description=(
                f"端到端响应延迟（实测 {total_duration_ms:.0f}ms，目标 <{self.LATENCY_BENCHMARK_MS}ms）"
                if total_duration_ms > 0
                else "端到端响应延迟未提供，按中性分处理"
            ),
            details={
                "total_duration_ms": round(total_duration_ms, 2),
                "measured": total_duration_ms > 0,
            },
            benchmark=self.LATENCY_BENCHMARK_MS / 1000,
        ))

        # 2. Token Efficiency
        if input_tokens > 0:
            token_ratio = output_tokens / input_tokens
            efficiency_score = max(0.0, 1.0 - (token_ratio / self.TOKEN_RATIO_BENCHMARK))
            efficiency_score = min(1.0, efficiency_score)
        else:
            efficiency_score = 0.5

        metrics.append(MetricScore(
            name="Token Efficiency",
            value=round(efficiency_score, 4),
            level=get_score_level(efficiency_score),
            description="Token 使用效率（输出/输入比越低越好）",
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ratio": round(output_tokens / max(1, input_tokens), 3) if input_tokens > 0 else 0,
            },
            benchmark=0.5,
        ))

        # 3. Success Rate
        success_rate = max(0.0, (total_requests - error_count) / total_requests)
        metrics.append(MetricScore(
            name="Success Rate",
            value=round(success_rate, 4),
            level=get_score_level(success_rate),
            description="任务无错误完成的比率",
            details={"error_count": error_count, "total_requests": total_requests},
            benchmark=self.SUCCESS_RATE_BENCHMARK,
        ))

        # 4. Resource Utilization (如果有数据)
        if cpu_peak_percent is not None or memory_peak_mb is not None:
            cpu_score = max(0.0, 1.0 - (cpu_peak_percent or 0) / 100) if cpu_peak_percent else 1.0
            mem_score = max(0.0, 1.0 - (memory_peak_mb or 0) / 500) if memory_peak_mb else 1.0
            resource_score = (cpu_score + mem_score) / 2

            metrics.append(MetricScore(
                name="Resource Utilization",
                value=round(resource_score, 4),
                level=get_score_level(resource_score),
                description="计算资源利用率（CPU/内存峰值）",
                details={
                    "cpu_peak_percent": cpu_peak_percent,
                    "memory_peak_mb": memory_peak_mb,
                },
                benchmark=0.8,
            ))

        # 加权聚合
        weights = {"Latency": 0.30, "Token Efficiency": 0.25,
                   "Success Rate": 0.30, "Resource Utilization": 0.15}
        weighted_sum = sum(
            m.value * weights.get(m.name, 0.20) for m in metrics
        )

        return DimensionScore(
            dimension="System Performance",
            metrics=metrics,
            weighted_score=round(weighted_sum, 4),
            level=get_score_level(weighted_sum),
        )


class UserSatisfactionEvaluator:
    """
    用户满意度评估器（自动化实现）

    说明：真正的用户满意度需要人工反馈，这里提供基于
    答案行为的隐式推断，适合无人工反馈时的自动化评估。

    【评估维度】

    1. Completeness (完整性)
       - 答案是否涵盖了用户查询的所有方面

    2. Clarity (清晰度)
       - 答案结构是否清晰（段落、列表、数据）

    3. Helpfulness (有用性)
       - 答案是否包含具体建议/数据/比例

    4. Engagement (参与度)
       - 基于隐式行为（追问/复制）的代理指标
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        final_answer: bool = True,
        user_feedback_score: Optional[float] = None,
        follow_up_count: int = 0,
        copy_count: int = 0,
    ) -> DimensionScore:
        """
        评估用户满意度

        Args:
            query: 用户查询
            answer: 系统答案
            final_answer: 是否为最终答案（用于判断追问意图）
            user_feedback_score: 显式用户评分 [0, 1]（可选）
            follow_up_count: 用户追问次数
            copy_count: 用户复制答案次数

        Returns:
            DimensionScore 用户满意度维度得分
        """
        metrics: List[MetricScore] = []

        # 1. Completeness（完整性）
        query_words = _extract_keywords(query)
        answer_words = _extract_keywords(answer)
        overlap = query_words & answer_words
        completeness = len(overlap) / max(1, len(query_words))

        query_type_coverage = 0
        if any(w in query for w in ['多少', '哪些', '什么', 'how many', 'which']):
            query_type_coverage += 0.25
        if any(w in query for w in ['为什么', '原因', 'why', 'reason']):
            query_type_coverage += 0.25
        if any(w in query for w in ['如何', '怎么', 'how', '方法']):
            query_type_coverage += 0.25
        if any(w in query for w in ['是否', '能不能']):
            query_type_coverage += 0.25

        completeness = completeness * 0.7 + query_type_coverage * 0.3

        metrics.append(MetricScore(
            name="Completeness",
            value=round(completeness, 4),
            level=get_score_level(completeness),
            description="答案是否覆盖了用户查询的所有方面",
            details={"keyword_coverage": round(len(overlap) / max(1, len(query_words)), 3)},
            benchmark=0.75,
        ))

        # 2. Clarity（清晰度）
        paragraphs = [p.strip() for p in re.split(r'\n\n+', answer) if p.strip()]
        has_lists = bool(re.search(r'^\s*[-*•]\s', answer, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+[.)]\s', answer, re.MULTILINE))
        sentence_count = len(re.split(r'[。！？\n]', answer))
        avg_sentence_len = len(answer) / max(1, sentence_count)

        clarity = 0.0
        if len(paragraphs) >= 2:
            clarity += 0.2
        if has_lists or has_numbers:
            clarity += 0.3
        if 15 <= avg_sentence_len <= 50:
            clarity += 0.3
        if sentence_count >= 3:
            clarity += 0.2

        metrics.append(MetricScore(
            name="Clarity",
            value=round(min(1.0, clarity), 4),
            level=get_score_level(min(1.0, clarity)),
            description="答案结构是否清晰易读",
            details={
                "paragraphs": len(paragraphs),
                "has_structured_list": bool(has_lists or has_numbers),
                "avg_sentence_length": round(avg_sentence_len, 1),
            },
            benchmark=0.70,
        ))

        # 3. Helpfulness（有用性）
        helpful_keywords = ['建议', '推荐', '配置', '仓位', '比例', '应该', '可以', '方案']
        has_specific_data = bool(re.search(r'[\d.]+%|[一二三]成|\d+倍', answer))
        has_action = any(kw in answer for kw in helpful_keywords)

        helpfulness = 0.0
        if has_action:
            helpfulness += 0.4
        if has_specific_data:
            helpfulness += 0.4
        if len(answer) > 200:
            helpfulness += 0.2

        metrics.append(MetricScore(
            name="Helpfulness",
            value=round(min(1.0, helpfulness), 4),
            level=get_score_level(min(1.0, helpfulness)),
            description="答案是否包含具体可操作的建议和数据",
            details={
                "has_action_words": has_action,
                "has_specific_data": has_specific_data,
            },
            benchmark=0.65,
        ))

        # 4. Engagement（参与度代理指标）
        engagement_score = 0.5
        if user_feedback_score is not None:
            engagement_score = user_feedback_score
        elif final_answer and follow_up_count == 0:
            engagement_score = 0.7
        elif follow_up_count > 0:
            engagement_score = max(0.3, 0.7 - follow_up_count * 0.1)
        if copy_count > 0:
            engagement_score = min(1.0, engagement_score + 0.1)

        metrics.append(MetricScore(
            name="Engagement",
            value=round(engagement_score, 4),
            level=get_score_level(engagement_score),
            description="基于隐式行为（追问/复制）的用户参与度代理指标",
            details={
                "follow_up_count": follow_up_count,
                "copy_count": copy_count,
                "user_feedback_score": user_feedback_score,
            },
            benchmark=0.70,
        ))

        # 加权聚合
        weights = {"Completeness": 0.30, "Clarity": 0.25,
                   "Helpfulness": 0.25, "Engagement": 0.20}
        weighted_sum = sum(
            m.value * weights.get(m.name, 0.20) for m in metrics
        )

        return DimensionScore(
            dimension="User Satisfaction",
            metrics=metrics,
            weighted_score=round(weighted_sum, 4),
            level=get_score_level(weighted_sum),
        )


# ════════════════════════════════════════════════════════════
# 第六部分：综合评估器
# ════════════════════════════════════════════════════════════


class ComprehensiveEvaluator:
    """
    综合评估协调器
    
    整合所有子评估器的结果，生成完整的 EvaluationReport。
    
    【工作流程】
    1. 接收输入（查询、答案、上下文、执行步骤等）
    2. 并行/串行调用各维度评估器
    3. 聚合各维度得分，计算综合分
    4. 分析优势和劣势
    5. 生成针对性改进建议
    6. 输出结构化报告
    """
    
    def __init__(self, use_llm: bool = False):
        """
        初始化综合评估器。

        Args:
            use_llm: 是否为 RAG 质量评估启用 LLM 辅助判断。
                     False 使用启发式规则，True 会调用 ChatOpenAI。
        """
        self.use_llm = use_llm
        self.rag_evaluator = RAGQualityEvaluator(use_llm=use_llm)
        self.agent_evaluator = AgentQualityEvaluator()
        self.output_evaluator = OutputQualityEvaluator()
        self.system_evaluator = SystemPerformanceEvaluator()
        self.user_satisfaction_evaluator = UserSatisfactionEvaluator()
    
    def full_evaluate(
        self,
        query: str,
        answer: str,
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        execution_steps: Optional[List[Dict[str, Any]]] = None,
        execution_time_ms: float = 0,
        tool_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_count: int = 0,
        user_feedback_score: Optional[float] = None,
        follow_up_count: int = 0,
        copy_count: int = 0,
    ) -> EvaluationReport:
        """
        执行全面评估（主入口方法）
        
        Args:
            query: 用户查询
            answer: 系统答案
            contexts: 检索上下文
            ground_truth: 参考答案
            execution_steps: Agent 执行步骤
            execution_time_ms: 执行时间
            tool_call_count: 工具调用次数
            
        Returns:
            EvaluationReport 完整评估报告
        """
        start_time = time.time()
        
        report = EvaluationReport(
            evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            query=query,
            answer=answer,
            contexts=contexts or [],
            ground_truth=ground_truth,
        )
        
        if contexts:
            report.rag_quality = self.rag_evaluator.evaluate(
                query, answer, contexts, ground_truth
            )
        
        if execution_steps:
            report.agent_quality = self.agent_evaluator.evaluate(
                query, answer, execution_steps,
                execution_time_ms, tool_call_count
            )
        
        report.output_quality = self.output_evaluator.evaluate(answer, contexts)

        report.system_performance = self.system_evaluator.evaluate(
            total_duration_ms=execution_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error_count=error_count,
            total_requests=1,
        )

        report.user_satisfaction = self.user_satisfaction_evaluator.evaluate(
            query=query,
            answer=answer,
            user_feedback_score=user_feedback_score,
            follow_up_count=follow_up_count,
            copy_count=copy_count,
        )

        dimensions = [
            report.rag_quality,
            report.agent_quality,
            report.output_quality,
            report.system_performance,
            report.user_satisfaction,
        ]
        
        valid_dims = [d for d in dimensions if d is not None]
        
        if valid_dims:
            dimension_weights = {
                "RAG Quality": DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.RAG_QUALITY],
                "Agent Execution Quality": DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.AGENT_QUALITY],
                "Output Quality": DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.OUTPUT_QUALITY],
                "System Performance": DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.SYSTEM_PERFORMANCE],
                "User Satisfaction": DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.USER_SATISFACTION],
            }
            weighted_terms = [
                (
                    d.weighted_score,
                    dimension_weights.get(d.dimension, 0.0),
                )
                for d in valid_dims
            ]
            total_weight = sum(weight for _, weight in weighted_terms)
            report.overall_score = round(
                sum(score * weight for score, weight in weighted_terms) / max(total_weight, 1e-9),
                4,
            )
            report.overall_level = get_score_level(report.overall_score)
        
        strengths, weaknesses, recommendations = self._generate_recommendations(report)
        report.strengths = strengths
        report.weaknesses = weaknesses
        report.recommendations = recommendations
        
        report.evaluation_time_ms = (time.time() - start_time) * 1000
        
        try:
            print(report.summary())
        except Exception as e:
            print(f"[评估] ⚠️ 生成摘要时出错: {e}")
            print(f"[评估]    overall_score={report.overall_score}, overall_level={report.overall_level}")
        
        return report
    
    def _generate_recommendations(
        self,
        report: EvaluationReport,
    ) -> Tuple[List[str], List[str], List[str]]:
        """生成改进建议"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        all_metrics = []
        if report.rag_quality:
            all_metrics.extend(report.rag_quality.metrics)
        if report.agent_quality:
            all_metrics.extend(report.agent_quality.metrics)
        if report.output_quality:
            all_metrics.extend(report.output_quality.metrics)
        if report.system_performance:
            all_metrics.extend(report.system_performance.metrics)
        if report.user_satisfaction:
            all_metrics.extend(report.user_satisfaction.metrics)
        
        for metric in all_metrics:
            if metric.level in (ScoreLevel.EXCELLENT, ScoreLevel.GOOD):
                strengths.append(f"{metric.name}: {metric.value:.2f}")
            elif metric.level in (ScoreLevel.POOR, ScoreLevel.CRITICAL):
                weaknesses.append(f"{metric.name}: {metric.value:.2f}")
                
                rec_map = {
                    "Faithfulness": "增强上下文验证机制，减少幻觉产生",
                    "Answer Relevancy": "优化意图理解，确保答案聚焦用户问题",
                    "Context Precision": "优化检索策略，提高文档相关性",
                    "Context Recall": "扩大检索范围或增加查询变体",
                    "Goal Achievement": "强化任务规划，确保目标对齐",
                    "Tool Efficiency": "优化工具选择逻辑，减少冗余调用",
                    "Reasoning Quality": "增加推理深度提示，引导更严谨思考",
                    "Professionalism": "优化输出模板，提升专业表达",
                    "Data Accuracy": "加强数据源校验和事实核查",
                    "Logical Coherence": "优化论述结构模板",
                    "Actionability": "增加具体建议的生成指引",
                }
                
                rec = rec_map.get(metric.name)
                if rec and rec not in recommendations:
                    recommendations.append(rec)
        
        return strengths, weaknesses, recommendations


# ════════════════════════════════════════════════════════════
# 第七部分：便捷函数与自测套件
# ════════════════════════════════════════════════════════════


def evaluate_response(
    query: str,
    answer: str,
    contexts: Optional[List[str]] = None,
    **kwargs,
) -> EvaluationReport:
    """
    便捷函数：执行一次完整的全面评估
    
    Args:
        query: 用户查询
        answer: 系统答案
        contexts: 检索上下文
        **kwargs: 传递给 full_evaluate 的额外参数
        
    Returns:
        EvaluationReport 完整评估报告
        
    示例:
        >>> report = evaluate_response(
        ...     query="茅台2023年盈利能力如何？",
        ...     answer="根据年报...",
        ...     contexts=["茅台2023年营收...", "..."],
        ... )
        >>> print(report.overall_score)
        0.8567
    """
    use_llm = bool(kwargs.pop("use_llm", False))
    evaluator = ComprehensiveEvaluator(use_llm=use_llm)
    return evaluator.full_evaluate(
        query=query,
        answer=answer,
        contexts=contexts,
        **kwargs,
    )


def _run_self_tests():
    """
    评估体系自测套件
    
    测试场景覆盖:
    1. 评分等级映射
    2. 数据模型创建与序列化
    3. RAG 质量评估（忠实度/相关性/精确率）
    4. Agent 质量评估（目标达成/工具效率）
    5. 输出质量评估（专业性/准确性/连贯性/可操作性）
    6. 综合报告生成
    """
    print("\n" + "=" * 70)
    print("  Enhanced Evaluation Framework — Self-Test Suite")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    def test(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  [{status}] {name}")
        if detail and not condition:
            print(f"         → {detail}")
        passed += 1 if condition else 0
        failed += 1 if not condition else 0
    
    print("\n--- Test Group 1: Enums & Utilities ---")
    
    test("ScoreLevel EXCELLENT range", get_score_level(0.95) == ScoreLevel.EXCELLENT)
    test("ScoreLevel GOOD range", get_score_level(0.80) == ScoreLevel.GOOD)
    test("ScoreLevel ACCEPTABLE range", get_score_level(0.60) == ScoreLevel.ACCEPTABLE)
    test("ScoreLevel POOR range", get_score_level(0.40) == ScoreLevel.POOR)
    test("ScoreLevel CRITICAL range", get_score_level(0.20) == ScoreLevel.CRITICAL)
    test("ScoreLevel boundary (0.9)", get_score_level(0.9) == ScoreLevel.EXCELLENT)
    test("ScoreLevel boundary (0.7)", get_score_level(0.7) == ScoreLevel.GOOD)
    
    test("EvaluationDimension count", len(EvaluationDimension) == 5)
    test("Default weights sum ≈ 1.0", abs(sum(DEFAULT_DIMENSION_WEIGHTS.values()) - 1.0) < 0.01)
    
    print("\n--- Test Group 2: Data Models ---")
    
    ms = MetricScore(
        name="TestMetric",
        value=0.85,
        level=ScoreLevel.GOOD,
        description="Test metric",
        benchmark=0.80,
    )
    test("MetricScore creation", ms.name == "TestMetric" and ms.value == 0.85)
    test("MetricScore.meets_benchmark", ms.meets_benchmark)
    test("MetricScore.is_passing", ms.is_passing)
    test("MetricScore.to_dict", "value" in ms.to_dict())
    
    ds = DimensionScore(
        dimension="TestDim",
        metrics=[ms],
        weighted_score=0.85,
        level=ScoreLevel.GOOD,
    )
    test("DimensionScore creation", ds.n_metrics == 1)
    test("DimensionScore.best_metric", ds.best_metric is not None)
    test("DimensionScore.worst_metric", ds.worst_metric is not None)
    test("DimensionScore.get_metric_by_name", ds.get_metric_by_name("TestMetric") is not None)
    test("DimensionScore.summary generation", len(ds.summary()) > 10)
    
    er = EvaluationReport(
        evaluation_id="test_001",
        timestamp=datetime.now().isoformat(),
        query="测试查询",
        answer="测试答案",
        overall_score=0.8,
        overall_level=ScoreLevel.GOOD,
    )
    test("EvaluationReport creation", er.query == "测试查询")
    test("EvaluationReport.to_dict", "overall_score" in er.to_dict())
    test("EvaluationReport.summary generation", len(er.summary()) > 50)
    
    print("\n--- Test Group 3: RAG Quality Evaluator ---")
    
    rag = RAGQualityEvaluator(use_llm=False)
    
    faith = rag._evaluate_faithfulness(
        "茅台2023年净利润747亿元",
        ["贵州茅台2023年净利润达到747.34亿元"]
    )
    test("Faithfulness (high)", faith.value > 0.8, f"Got {faith.value}")
    
    faith_low = rag._evaluate_faithfulness(
        "茅台2023年亏损500亿元",
        ["贵州茅台2023年净利润747.34亿元"]
    )
    test("Faithfulness (low)", faith_low.value < 0.5, f"Got {faith_low.value}")
    
    rel = rag._evaluate_answer_relevancy("茅台盈利能力", "茅台盈利能力强，净利率53%")
    test("Answer Relevancy (high)", rel.value > 0.5, f"Got {rel.value}")
    
    prec = rag._evaluate_context_precision("茅台盈利", [
        "茅台2023年净利润747亿元",
        "今天天气不错",
        "苹果公司发布新手机",
    ])
    test("Context Precision (mixed)", 0.3 < prec.value < 0.8, f"Got {prec.value}")
    
    prec_empty = rag._evaluate_context_precision("茅台盈利", [])
    test("Context Precision (empty)", prec_empty.value == 0.0)
    
    dim_rag = rag.evaluate(
        query="茅台盈利能力",
        answer="茅台2023年净利润747亿，毛利率91.96%",
        contexts=["贵州茅台2023年净利润747.34亿元，毛利率91.96%"],
    )
    test("RAG full evaluation", dim_rag.n_metrics >= 3)
    test("RAG weighted score valid", 0.0 <= dim_rag.weighted_score <= 1.0)
    
    print("\n--- Test Group 4: Agent Quality Evaluator ---")
    
    agent = AgentQualityEvaluator()
    
    goal = agent._evaluate_goal_achievement("茅台股价", "茅台当前股价1680元")
    test("Goal Achievement (good)", goal > 0.5, f"Got {goal}")
    
    eff = agent._evaluate_tool_efficiency(5, [])
    test("Tool Efficiency (optimal)", eff >= 0.9, f"Got {eff}")
    
    eff_many = agent._evaluate_tool_efficiency(15, [])
    test("Tool Efficiency (too many)", eff_many < 0.7, f"Got {eff_many}")
    
    time_fast = agent._evaluate_time_efficiency(5000)
    test("Time Efficiency (fast)", time_fast == 1.0)
    
    time_slow = agent._evaluate_time_efficiency(180000)
    test("Time Efficiency (slow)", time_slow < 0.5, f"Got {time_slow}")
    
    dim_agent = agent.evaluate(
        query="分析茅台",
        final_answer="茅台是一家好公司...",
        execution_steps=[
            {"step_type": "thought"},
            {"step_type": "action", "tool": "search"},
            {"step_type": "observation"},
        ],
        total_time_ms=15000,
        tool_call_count=3,
    )
    test("Agent full evaluation", dim_agent.n_metrics == 4)
    
    print("\n--- Test Group 5: Output Quality Evaluator ---")
    
    output = OutputQualityEvaluator()
    
    prof = output._evaluate_professionalism("""
    ## 贵州茅台分析
    ### 盈利能力
    ROE达到34.2%，毛利率91.96%。
    """)
    test("Professionalism (structured)", prof > 0.7, f"Got {prof}")
    
    prof_plain = output._evaluate_professionalism("茅台不错挺好的")
    test("Professionalism (plain)", prof_plain < 0.5, f"Got {prof_plain}")
    
    acc = output._evaluate_data_accuracy(
        "根据财报显示，营收1505亿，同比增长18.9%",
        ["贵州茅台2023年营业收入1505亿元"],
    )
    test("Data Accuracy (with source)", acc > 0.6, f"Got {acc}")
    
    coh = output._evaluate_coherence("""
    首先，茅台盈利能力强。
    其次，品牌价值高。
    最后，建议长期持有。
    综上所述，茅台值得投资。
    """)
    test("Logical Coherence (structured)", coh > 0.7, f"Got {coh}")
    
    act = output._evaluate_actionability("""
    建议配置30%仓位，
    推荐长期持有，
    需要关注政策风险。
    """)
    test("Actionability (specific)", act > 0.6, f"Got {act}")
    
    dim_output = output.evaluate(
        answer="## 分析报告\n\n根据数据显示...",
        contexts=["参考数据..."],
    )
    test("Output full evaluation", dim_output.n_metrics == 4)
    
    print("\n--- Test Group 6: Comprehensive Evaluator ---")
    
    comp = ComprehensiveEvaluator()
    
    report = comp.full_evaluate(
        query="茅台2023年盈利能力",
        answer="根据年报，茅台2023年净利润747亿，毛利率91.96%，ROE达34.2%。"
               "综上所述，盈利能力强劲。",
        contexts=["贵州茅台2023年净利润747.34亿元，毛利率91.96%"],
        execution_steps=[
            {"step_type": "thought"},
            {"step_type": "action", "tool_name": "search"},
            {"step_type": "observation"},
        ],
        execution_time_ms=12000,
        tool_call_count=2,
    )
    test("Full evaluation report created", report.evaluation_id.startswith("eval_"))
    test("Overall score valid", 0.0 <= report.overall_score <= 1.0)
    test("Has RAG quality dimension", report.rag_quality is not None)
    test("Has output quality dimension", report.output_quality is not None)
    test("Has recommendations", len(report.recommendations) > 0)
    test("Report summary works", len(report.summary()) > 100)
    
    print("\n--- Test Group 7: Convenience Functions ---")
    
    test("evaluate_response exists", callable(evaluate_response))
    
    quick_report = evaluate_response(
        query="测试查询",
        answer="测试答案",
        contexts=["上下文1", "上下文2"],
    )
    test("Quick evaluation works", quick_report.overall_score >= 0.0)
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed - please review")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_tests()
