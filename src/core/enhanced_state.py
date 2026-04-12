"""
增强版 Agent State 定义模块 (Enhanced Agent State)

本模块定义了企业级Multi-Agent系统的完整状态结构，
支持从用户输入到最终研报生成的全链路数据流转。

设计原则：
1. 类型安全：使用TypedDict + Optional确保类型完整性
2. 可扩展性：预留扩展字段，支持未来功能迭代
3. 可追溯性：完整的执行链路追踪，便于调试和审计
4. 性能友好：使用Optional避免不必要的数据存储

核心能力：
- Supervisor+SubAgent层级架构状态管理
- 槽位提取结果的结构化存储
- 多层次意图分类的层级表示
- Query改写历史的版本管理
- 动态任务规划的状态追踪
- ReAct思考过程的完整记录
- 多维度评估结果的聚合

作者：AI Agent Team
创建时间：2026-04-12
最后修改：2026-04-12
版本：2.0.0-enterprise
"""

from __future__ import annotations

# 标准库类型注解
from typing import (
    Any,                    # 任意类型（用于动态字段）
    Dict,                   # 字典类型
    List,                   # 列表类型
    Optional,               # 可选类型（允许None）
    TypedDict,              # 类型化字典（类似dataclass但更轻量）
    Annotated,              # 带元数据的类型注解（用于LangChain消息列表）
)

# LangChain核心组件
from langchain_core.messages import BaseMessage

# LangGraph reducer 函数（用于消息列表的追加语义）
from langgraph.graph.message import add_messages


class EnhancedAgentState(TypedDict):
    """
    增强版Agent状态定义（TypedDict）
    
    这是整个Multi-Agent系统的"中央神经系统"，承载了从用户输入
    到最终输出的所有中间状态和数据。
    
    设计架构（分层设计）：
    ┌─────────────────────────────────────────────┐
    │ Layer 1: 对话层 (Conversation)           │ ← 用户交互
    ├─────────────────────────────────────────────┤
    │ Layer 2: 输入层 (Input)                 │ ← 原始输入保留
    ├─────────────────────────────────────────────┤
    │ Layer 3: 理解层 (Understanding)          │ ← 槽位+意图
    ├─────────────────────────────────────────────┤
    │ Layer 4: 规划层 (Planning)              │ ← 任务分解
    ├─────────────────────────────────────────────┤
    │ Layer 5: 执行层 (Execution)             │ ← ReAct循环
    ├─────────────────────────────────────────────┤
    │ Layer 6: 输出层 (Output)                │ ← 各Agent结果
    ├─────────────────────────────────────────────┤
    │ Layer 7: 控制层 (Control)               │ ← 流程控制
    ├─────────────────────────────────────────────┤
    │ Layer 8: 观测层 (Observability)          │ ← 追踪&调试
    └─────────────────────────────────────────────┘
    
    使用示例：
        state = {
            "user_query": "分析茅台2023年盈利能力",
            "slots": {"company": "贵州茅台", "year": 2023},
            "intent": "financial_analysis",
            "task_plan": [...],
            "final_report": "# 投资研究报告..."
        }
    
    注意事项：
        - 所有Optional字段在首次访问前需检查是否为None
        - messages字段使用Annotated实现追加语义（非覆盖）
        - iteration字段由Supervisor自动递增，手动修改可能导致死循环
    
    性能优化建议：
        - 对于大型上下文，考虑使用__init__.py中的lazy loading
        - execution_trace在生产环境可设为None以节省内存
        - thoughts列表可通过配置限制最大长度防止OOM
    """
    
    # ══════════════════════════════════════════════
    # Layer 1: 对话层 (Conversation Layer)
    # ══════════════════════════════════════════════
    
    messages: Annotated[list[BaseMessage], add_messages]
    """
    对话历史消息列表（LangChain标准格式）
    
    类型：list[BaseMessage] - LangChain的消息对象列表
    语义：add_messages - 追加模式（新消息append，非覆盖）
    
    包含的消息类型：
    - HumanMessage: 用户输入
    - AIMessage: Agent回复（含工具调用记录）
    - ToolMessage: 工具执行结果
    - SystemMessage: 系统提示词（通常在首条）
    
    使用场景：
    - 多轮对话上下文保持
    - ReAct循环的工具调用记录
    - 错误恢复时的历史回溯
    
    内存占用估算：
    - 平均每条消息约 1-5KB（取决于content长度）
    - 建议10轮对话后进行摘要压缩
    """
    
    # ══════════════════════════════════════════════
    # Layer 2: 输入层 (Input Layer)
    # ══════════════════════════════════════════════
    
    user_query: str
    """
    用户原始问题（未经任何处理的原始输入）
    
    数据来源：main.py 或 API接口直接传入
    用途：作为最终报告的问题引用，保证可追溯性
    示例："分析贵州茅台2023年的盈利能力和现金流状况"
    
    不变原则：此字段一旦设置不应被修改，所有改写操作
              应作用于 current_query 字段
    """
    
    original_query: str
    """
    原始查询的副本（与user_query相同，用于内部一致性检查）
    
    设计原因：
    - 防止意外覆盖user_query
    - 支持断言检查 assert(state['original_query'] == state['user_query'])
    - 在Query改写模块中可作为基准对比
    
    最佳实践：
    初始化时：original_query = user_query = input_string
    后续只读：不再修改这两个字段
    """
    
    # ══════════════════════════════════════════════
    # Layer 3: 理解层 (Understanding Layer)
    # ══════════════════════════════════════════════
    
    slots: Optional[Dict[str, Any]]
    """
    槽位提取结果（结构化的实体信息）
    
    由 SlotExtractor 模块生成，包含以下类型的槽位：
    
    结构示例（完整版）：
    {
        "companies": [                          # 公司实体列表
            {
                "name": "贵州茅台",            # 公司全称
                "stock_code": "600519.SH",     # 股票代码（A股格式）
                "aliases": ["茅台", "贵州茅台"]  # 别名列表
            }
        ],
        "time_info": {                         # 时间相关信息
            "year": 2023,                      # 具体年份
            "quarter": None,                   # 季度（1-4）
            "period": None,                    # 时间范围描述（如"近三年"）
            "time_range": None                 # 起止时间范围tuple
        },
        "metrics": {                           # 财务指标信息
            "primary_metrics": ["毛利率", "净利率", "ROE"],  # 主要指标
            "secondary_metrics": [],                       # 次要指标
            "metric_categories": ["盈利能力"]                # 指标类别
        },
        "comparison_targets": [               # 对比分析目标
            {
                "name": "五粮液",
                "stock_code": "000858.SZ"
            }
        ],
        "industry": "白酒行业",               # 所属行业
        "analysis_type": "深度解读",           # 分析类型
        "confidence": 0.95,                   # 提取置信度(0-1)
        "missing_slots": [],                  # 缺失的关键槽位
        "clarification_needed": False,         # 是否需要向用户澄清
        "clarification_questions": []          # 澄清问题列表
    }
    
    为空时的含义：
    - None: 尚未执行槽位提取或提取失败
    - {}: 执行了提取但未提取到任何有效槽位（应触发降级处理）
    
    下游消费者：
    - IntentClassifier: 利用company和metrics辅助分类
    - TaskPlanner: 用time_info和metrics参数化任务模板
    - QueryRewriter: 用structured info优化检索query
    """
    
    slot_confidence: Optional[float]
    """
    槽位提取的整体置信度评分（0.0 - 1.0）
    
    计算方式（SlotExtractor内部）：
    - 公司实体存在: +0.25分
    - 时间信息完整: +0.15分
    - 财务指标明确: +0.20分
    - 有对比目标: +0.10分
    - 有行业信息: +0.10分
    - 有分析类型: +0.10分
    - 混合方法加成: ×1.1（规则+LLM融合时）
    
    阈值参考：
    - ≥ 0.8: 高置信度，可直接使用
    - 0.5 - 0.8: 中等置信度，建议人工确认
    - < 0.5: 低置信度，必须澄清后继续
    
    典型应用：
    if state.get('slot_confidence', 0) < 0.7:
        # 触发澄清流程
        return ask_clarification(state['slots']['clarification_questions'])
    """
    
    intent: Optional[str]
    """
    主意图标签（Primary Intent Classification Result）
    
    可能的值（8大类）：
    - "financial_analysis": 财务分析（财报解读、指标计算）
    - "stock_comparison": 个股对比（多公司横向比较）
    - "industry_overview": 行业综述（产业链、竞争格局）
    - "realtime_query": 实时行情（股价、新闻查询）
    - "portfolio_analysis": 组合分析（持仓优化、风险评估）
    - "risk_alert": 风险预警（异常检测、黑天鹅事件）
    - "strategy_backtest": 策略回测（历史验证、绩效评估）
    - "market_sentiment": 市场情绪（舆情监控、投资者心理）
    
    分类器：EnhancedIntentClassifier（Few-shot + LLM混合）
    输入依赖：user_query + slots（可选，提升准确率）
    
    下游影响：
    决定TaskPlanner选择哪个任务模板（见TASK_TEMPLATES字典）
    """
    
    intent_hierarchy: Optional[Dict[str, Any]]
    """
    意图的层级结构（支持主意图+子意图+候选排序）
    
    完整结构示例：
    {
        "primary": "financial_analysis",       # 主意图（最高置信）
        "secondary": "profitability_analysis", # 子意图（细粒度分类）
        "tertiary": None,                     # 三级意图（未来扩展用）
        
        "confidence": 0.95,                    # 主意图置信度
        
        "alternatives": [                     # Top-N候选意图（用于消歧）
            {
                "intent": "stock_comparison", 
                "confidence": 0.03             # 候选置信度
            },
            {
                "intent": "industry_overview",
                "confidence": 0.02
            }
        ],
        
        "definition": "解读公司财报、财务指标...",  # 意图的自然语言描述
        "ambiguity_detected": False,            # 是否检测到歧义
        "disambiguation_questions": [...]      # 消歧问题（如歧义则非空）
    }
    
    歧义判定条件：
    - alternatives[0].confidence > 0.15 且
    - |primary.confidence - alternatives[0].confidence| < 0.15
    
    使用场景：
    - TaskPlanner根据primary选择任务模板
    - 当ambiguity_detected=True时，Supervisor应暂停并询问用户
    """
    
    # ══════════════════════════════════════════════
    # Layer 4: 规划层 (Planning Layer)
    # ══════════════════════════════════════════════
    
    rewritten_queries: Optional[List[str]]
    """
    Query改写后的候选查询列表（按优化程度降序排列）
    
    由 QueryRewriter 模块生成，包含多轮改写的所有版本：
    [
        "原始查询: 分析茅台去年赚了多少钱",
        "第1轮(标准化): 解读贵州茅台2023年度盈利能力财务报告...",
        "第2轮(特化): 获取贵州茅台2023年净利润、营收增速、ROE等关键盈利指标...",
        "第3轮(扩展): 贵州茅台2023年盈利能力 （相关概念：净利润 营收 ROE 毛利率）"
    ]
    
    选择策略：
    - current_query: 当前正在使用的最优查询（通常取第一项或评分最高的）
    - 备选方案：当当前查询检索效果不佳时可回退到其他候选
    
    最大长度限制：MAX_QUERY_REWRITES（默认3轮）
    超出限制时的行为：保留已有的，不再新增
    """
    
    current_query: Optional[str]
    """
    当前实际使用的查询（经过改写优化的最终版本）
    
    初始值：等于 original_query（未改写时）
    改写后：取 rewritten_queries 中 quality_score 最高的一个
    
    使用位置：
    - RAG检索器的输入
    - IntentClassifier的补充输入
    - 最终报告中引用的"研究问题"文本
    
    一致性保证：
    应始终满足：current_query in rewritten_queries or current_query == original_query
    """
    
    query_rewrite_count: int
    """
    Query改写的实际执行次数（计数器）
    
    初始值：0（未执行改写）
    每次调用QueryRewriter.rewrite()后 +1
    上限：MAX_QUERY_REWRITES（默认3）
    
    监控意义：
    - count=0: 直接使用原始查询（可能检索效果差）
    - count=1-2: 正常优化范围
    - count=3: 达到上限（可能问题本身复杂或模糊）
    
    性能关联：
    每次改写消耗约 500-1000 tokens（LLM调用）
    总改写成本 ≈ query_rewrite_count * 750 tokens
    """
    
    task_plan: Optional[List[Dict[str, Any]]]
    """
    任务分解计划（DAG结构的任务列表）
    
    由 TaskPlanner.generate() 方法生成，基于意图选择对应模板：
    
    单个任务项结构：
    {
        "task_id": "T1",                    # 任务唯一标识（T前缀+序号）
        "task_type": "data_retrieval",       # 任务类型枚举值
        "description": "获取茅台2023年财报数据",  # 自然语言描述
        "parameters": {...},                 # 执行参数（已填充槽位值）
        "dependencies": [],                  # 依赖的任务ID列表（DAG边）
        "assigned_agent": "retrieval_agent", # 分配执行的Agent名称
        "assigned_skill": "rag_retrieval",   # 使用的Skill名称
        "status": "pending",                 # 当前状态（pending/running/completed/failed）
        "priority": "high",                 # 优先级（critical/high/medium/low）
        
        # 运行时字段（初始为空，执行过程中填充）
        "result": None,                      # 任务输出结果
        "error": None,                       # 错误信息（失败时）
        "start_time": "",                    # 开始时间戳
        "end_time": "",                      # 结束时间戳
        "retry_count": 0,                    # 重试次数
        "estimated_tokens": 500,             # 预估Token消耗
        "estimated_time_seconds": 2.0        # 预估耗时（秒）
    }
    
    执行顺序：
    按 execution_order 分组（支持并行），例如：
    [[T1, T2], [T3], [T4, T5], [T6]]
    表示：T1和T2并行 → T3 → T4和T5并行 → T6
    
    典型计划长度：
    - financial_analysis: 6个任务（3阶段并行）
    - stock_comparison: 5个任务
    - realtime_query: 3个任务（快速路径）
    """
    
    current_task_index: int
    """
    当前正在执行的任务在 task_plan 列表中的索引（0-based）
    
    初始值：0（尚未开始）
    递增时机：Supervisor每次dispatch后 +1
    上限：len(task_plan) - 1
    
    使用场景：
    - 进度显示：f"正在执行 {current_task_index+1}/{total_tasks}"
    - 断点续传：从上次中断的index恢复
    - 循环检测：防止重复执行同一任务
    
    边界检查：
    if current_task_index >= len(task_plan or []):
        # 所有任务已完成，进入报告生成阶段
        return NODE_REPORT_GENERATOR
    """
    
    # ══════════════════════════════════════════════
    # Layer 5: 执行层 (Execution Layer)
    # ══════════════════════════════════════════════
    
    selected_skills: Optional[List[str]]
    """
    当前任务选择的技能名称列表（来自SkillRegistry）
    
    示例：["rag_retrieval", "financial_analysis", "risk_assessment"]
    
    选择依据：
    - task_plan[current_task].assigned_skill
    - SkillSelector根据任务类型和可用技能匹配
    
    技能类型（SkillCategory）：
    - data_retrieval: 数据检索类（RAG、API调用）
    - analysis: 分析类（财务分析、技术分析）
    - calculation: 计算类（指标计算、估值模型）
    - generation: 生成类（报告生成、摘要）
    - validation: 验证类（数据校验、合规检查）
    """
    
    skill_parameters: Optional[Dict[str, Dict[str, Any]]]
    """
    各选中技能的运行参数（key=skill_name, value=params_dict）
    
    结构示例：
    {
        "rag_retrieval": {
            "query": "贵州茅台2023年盈利能力",
            "top_k": 5,
            "use_hyde": True
        },
        "financial_analysis": {
            "analysis_type": "profitability",
            "focus_metrics": ["ROE", "净利率"]
        }
    }
    
    参数来源：
    - TaskPlan中的预填参数（基于slots）
    - SkillContext中的运行时参数
    - 用户显式指定的覆盖参数
    """
    
    skill_execution_order: List[str]
    """
    技能的实际执行顺序（可能不同于selected_skills的声明顺序）
    
    决定因素：
    - 依赖关系（某些skill依赖其他skill的输出）
    - 管道模式（SkillPipeline定义的组合顺序）
    - 条件分支（根据中间结果动态调整）
    
    示例：["rag_retrieval", "metric_calculator", "financial_analysis"]
    表示：先检索 → 再计算精确指标 → 最后做综合分析
    """
    
    # ══════════════════════════════════════════════
    # ReAct 执行层详情（Thought-Action-Observation循环）
    # ══════════════════════════════════════════════
    
    thoughts: Optional[List[Dict[str, Any]]]
    """
    ReAct思考过程的完整记录（结构化日志）
    
    单步记录结构：
    {
        "step": 1,                           # 步骤序号（从1开始）
        "thought_type": "reasoning",         # 思考类型
                                         #   reasoning: 推理决策
                                         #   planning: 规划下一步
                                         #   reflection: 反思评估
                                         #   analysis: 数据分析
        
        "thought": "用户想了解茅台2023年盈利能力...",
                                           # 思考内容文本
        "action": "retrieve_financial_data",  # 决定采取的行动（工具/Skill名）
        "action_input": {                    # 行动的输入参数
            "company": "600519.SH",
            "year": 2023,
            "metrics": ["净利润", "ROE"]
        },
        "observation": "成功获取2023年年报数据：\n净利润747亿...",  # 行动结果
        "observation_summary": "数据完整，包含利润表和现金流表",  # 结果摘要
        "reflection": "数据质量良好，可以进行深度分析",  # 对结果的反思
        "confidence": 0.9,                  # 本步决策的置信度
        "tokens_used": 250,                  # 本步消耗的token数
        "duration_ms": 1200                  # 本步耗时（毫秒）
    }
    
    存储策略：
    - 默认保存所有步骤（用于调试和分析）
    - 生产环境可通过配置限制最大长度（如最近50步）
    - 可通过execution_trace字段导出为JSON供外部分析
    
    分析价值：
    - 推理链可解释性（Why this answer?）
    - 错误定位（哪一步出了问题？）
    - Token消耗审计（哪一步最费资源？）
    """
    
    react_step_count: int
    """
    ReAct循环已执行的步数（计数器）
    
    初始值：0
    每完成一轮 Thought→Action→Observation 后 +1
    上限：max_react_steps（默认10）
    
    终止条件（任一满足即停止）：
    1. react_step_count >= max_react_steps（达到上限）
    2. Final Answer已生成（目标达成）
    3. 连续2步Observation无新信息（早停）
    4. 发生不可恢复错误
    
    性能统计：
    平均每步消耗：~800 tokens（LLM推理）+ ~200ms（工具调用）
    总成本 ≈ react_step_count * 1000 tokens
    """
    
    max_react_steps: int
    """
    ReAct循环的最大允许步数（安全阀值）
    
    默认值：10
    配置位置：config/settings.py 或环境变量 MAX_REACT_STEPS
    
    设定依据：
    - 金融研报场景平均需要5-8步（检索→计算→分析→风控→实时→生成）
    - 过小可能导致任务无法完成（截断风险）
    - 过大导致Token浪费和延迟增加
    
    调优建议：
    - 简单查询（实时行情）：3-5步足够
    - 复杂分析（深度研报）：10-15步
    - 多跳推理（因果链）：可能需要20+步（需特殊配置）
    """
    
    # ══════════════════════════════════════════════
    # Layer 6: 输出层 (Output Layer)
    # ══════════════════════════════════════════════
    
    rag_context: Optional[str]
    """
    RAG检索Agent输出的Markdown格式上下文
    
    内容组成：
    ### 参考片段 1（相关度：0.923）
    > 来源：贵州茅台2023年年报.pdf
    
    根据合并报表，公司全年实现营业总收入1,505.60亿元...
    （包含具体的财务数据和表格内容）
    
    ---
    ### 参考片段 2（相关度：0.854）
    > 来源：贵州茅台2023年三季报.pdf
    ...
    
    特点：
    - 包含来源标注（支持溯源验证）
    - 包含相关性得分（用于质量评估）
    - Markdown格式（可直接嵌入LLM prompt）
    
    截断策略：
    - 默认保留top-5文档的内容
    - 总长度超过2000字符时会截断（避免超出context window）
    - 截断标记："...（已截断，显示前2000字符）"
    
    下游消费者：
    - FinancialAgent: 作为分析依据
    - RiskAgent: 用于独立审视（防止回声室效应）
    - ReportGenerator: 作为参考资料引用
    """
    
    financial_analysis: Optional[str]
    """
    财务分析Agent的专业分析结论
    
    输出者：FinancialAgent（CFA专家人设）
    分析框架（固定4维度）：
    1. 盈利能力：毛利率、净利率、ROE趋势及驱动因素
    2. 成长性：营收/净利润同比增速、未来增长引擎
    3. 现金流：经营现金流与净利润匹配度（质量判断）
    4. 估值水平：PE/PB历史分位、同业对标
    
    输出要求：
    - 语言专业简洁（金融术语准确）
    - 数据有据可查（引用参考片段编号如[Ref1]）
    - 明确亮点和风险点
    - 结尾给出评级：强烈推荐/推荐/中性/回避
    
    典型长度：800-1500字（取决于复杂度）
    LLM温度：0.3（适度创造性但不偏离事实）
    """
    
    risk_assessment: Optional[str]
    """
    风控Agent的独立风险评估报告
    
    输出者：RiskAgent（独立风控专家人设）
    设计原则：故意挑战FinancialAgent结论（防回声室效应）
    
    评估维度（4大类）：
    1. 财务风险：高负债率、商誉减值、现金流背离
    2. 行业竞争：价格战、替代品威胁、政策冲击
    3. 政策监管：反垄断调查、环保合规压力
    4. 估值风险：当前价格是否充分反映基本面
    
    每个风险点包含：
    - 触发条件：什么情况下会触发？（如"PE>50倍时"）
    - 影响程度：高/中/低（量化损失预估）
    - 缓解措施：如何规避或降低？（ actionable advice）
    
    最终输出：
    - 综合风险等级：高风险/中等风险/低风险
    - 2-3条具体的风险规避建议
    
    LLM温度：0.2（更保守、确定性输出）
    """
    
    realtime_data: Optional[str]
    """
    实时数据Agent的市场行情摘要
    
    数据来源：Alpha Vantage API（通过MCP工具调用）
    包含内容：
    ## 实时行情
    - 股票代码：600519.SH
    - 最新价：1756.00元
    - 涨跌额：+23.50元（+1.37%）
    - 成交量：3.2万手
    
    ## 最新资讯
    1. [标题] 茅台发布新品...（Bullish ⬆️）
    2. [标题] 机构调研报告...（Neutral ➡️）
    
    特点：
    - 数据时效性：API调用时刻的数据（非缓存）
    - 新闻情感标签：Bullish/Bearish/Neutral
    - Markdown格式（适合直接展示）
    
    更新频率：每次查询实时获取（无缓存）
    调用成本：每次约300ms + Alpha Vantage配额
    """
    
    quantitative_result: Optional[Dict[str, Any]]
    """
    量化分析Agent的计算结果（结构化数据，非文本）
    
    与其他Agent的区别：
    - FinancialAgent/RiskAgent: 输出自然语言分析
    - QuantitativeAgent: 输出精确数值计算结果
    
    数据结构示例：
    {
        "pe_ratio": {
            "value": 25.3,              # 计算结果
            "unit": "倍",               # 单位
            "formula": "PE = 股价/EPS",  # 公式
            "inputs": {                 # 输入参数（可追溯）
                "price": 1756.00,
                "eps": 69.45
            },
            "timestamp": "2026-04-12T23:55:00"  # 计算时间
        },
        "roe": {
            "value": 34.2,
            "unit": "%",
            ...
        },
        "summary": {                    # 汇总统计
            "metrics_calculated": 5,    # 计算了几个指标
            "all_successful": True,     # 是否全部成功
            "avg_calc_time_ms": 45.2    # 平均计算耗时
        }
    }
    
    支持的指标（8种）：
    pe_ratio, pb_ratio, roe, gross_margin, net_margin,
    debt_ratio, cagr, ev_ebitda
    
    计算特点：
    - Python原生运算（非LLM，避免数学幻觉）
    - 异常值处理（除零保护、负数警告）
    - 精度控制（四舍五入到2位小数）
    """
    
    # ══════════════════════════════════════════════
    # Layer 6.5: 结果聚合层 (Aggregation Layer)
    # ══════════════════════════════════════════════
    
    aggregated_results: Optional[Dict[str, Any]]
    """
    各Sub-Agent输出的中间聚合结果（ReportGenerator使用前的预处理）
    
    聚合时机：所有主要Agent完成后、报告生成前
    聚合内容：
    {
        "retrieval_stats": {              # 检索统计
            "docs_retrieved": 5,
            "avg_score": 0.87,
            "sources": ["2023年报.pdf", "2023Q3季报.pdf"]
        },
        "analysis_summary": {            # 分析要点提炼
            "financial_highlights": ["ROE创新高", "现金流充沛"],
            "risk_warnings": ["估值偏高", "行业竞争加剧"],
            "realtime_signals": ["股价突破均线", "资金净流入"]
        },
        "cross_validation": {            # 交叉验证
            "data_consistency": True,      # 各Agent数据是否一致
            "conclusion_alignment": "大部分一致",  # 结论是否矛盾
            "conflicts": []               # 具体冲突点（如有）
        }
    }
    
    目的：
    - 去重：去除各Agent重复的信息
    - 矛盾检测：发现FinancialAgent和RiskAgent结论冲突
    - 重要性排序：决定哪些内容放入最终报告
    - 截断决策：超长内容时的优先级裁剪
    
    处理方式：
    通常由ReportGenerator._aggregate()方法内部处理，
    此字段主要用于调试时查看聚合前的原始数据。
    """
    
    final_report: Optional[str]
    """
    最终生成的完整投资研究报告（Markdown格式）
    
    生成者：ReportGenerator（券商研报专家人设）
    标准章节结构（5部分）：
    
    # 执行摘要（50字以内）
    核心结论 + 投资建议 + 关键数据点
    
    # 公司/行业概况
    基本信息 + 近期动态 + 实时行情整合
    
    # 财务分析（来自FinancialAgent）
    四维度分析 + 数据支撑 + 专业评级
    
    # 风险评估（来自RiskAgent）
    主要风险 + 触发条件 + 规避建议
    
    # 投资结论
    综合评级 + 目标价区间（如有数据）+ 建议持仓比例
    
    元数据（YAML Front Matter）：
    ---
    generated_at: 2026-04-12T23:58:00
    query: 分析茅台2023年盈利能力
    evaluation_scores: {...}
    processing_time_seconds: 12.5
    token_consumption: 8500
    ---
    
    保存位置：reports/{timestamp}_{query前20字}.md
    文件编码：UTF-8
    LLM温度：0.4（平衡专业性和可读性）
    """
    
    # ══════════════════════════════════════════════
    # Layer 7: 控制层 (Control Flow Layer)
    # ══════════════════════════════════════════════
    
    next_agent: Optional[str]
    """
    Supervisor决定的下一个要执行的节点名称（路由目标）
    
    可能的值（NODE_*常量）：
    - "supervisor": 回到Supervisor审核
    - "slot_extractor": 执行槽位提取
    - "intent_classifier": 执行意图识别
    - "task_planner": 执行任务规划
    - "retrieval_agent": 执行RAG检索
    - "financial_agent": 执行财务分析
    - "risk_agent": 执行风险评估
    - "realtime_agent": 执行实时数据获取
    - "quantitative_agent": 执行量化计算
    - "report_generator": 执行报告生成
    - "evaluator": 执行质量评估
    - "__end__": 结束流程
    
    设置时机：
    - 每个Agent节点的return语句中设置
    - 格式：{**state, "next_agent": NODE_XXX}
    
    读取时机：
    - LangGraph的条件边函数 route_supervisor() 中读取
    - 根据此值决定下一个调用的节点函数
    
    特殊路由规则：
    1. intent="realtime_query" 时跳过RAG/分析，直达realtime_agent
    2. iteration >= max_iterations 时强制进入report_generator
    3. final_report有值且quality_pass时进入__end__
    """
    
    iteration: int
    """
    当前Supervisor的迭代次数（循环计数器）
    
    初始值：0（首次进入Supervisor时）
    递增时机：每次Supervisor.__call__()返回时 +1
    含义：已经完成了多少轮"分配→执行→审核"循环
    
    与react_step_count的区别：
    - iteration: Supervisor层面的宏观循环次数
    - react_step_count: 单个Agent内部的微观ReAct步数
    
    典型流程的iteration变化：
    Start → supervisor(iter=1) → retrieval(iter=2) → supervisor(iter=3) 
    → financial(iter=4) → supervisor(iter=5) → risk(iter=6) 
    → supervisor(iter=7) → realtime(iter=8) → supervisor(iter=9) 
    → report(iter=10) → END
    
    安全机制：
    每次进入Supervisor首先检查：if iteration >= max_iterations: 强制结束
    防止因路由错误导致的无限循环。
    """
    
    max_iterations: int
    """
    整个流程的最大允许迭代次数（全局安全阀）
    
    默认值：15
    设定依据：
    - 最复杂的流程（financial_analysis）需要约10次iteration
    - 加上容错重试（每步可能retry 2-3次）
    - 预留5次余量应对异常情况
    
    调优指南：
    - 快速查询（realtime）：设为5-8即可
    - 标准分析（financial）：设为10-12
    - 深度研究（comparison）：设为15-20
    - 调试阶段：设为100（确保不会过早终止）
    
    超限行为：
    强制跳转到 report_generator（即使数据不完整）
    并在final_report中添加警告："⚠️ 因达到最大迭代次数提前终止"
    """
    
    # ══════════════════════════════════════════════
    # Layer 8: 观测层 (Observability & Debugging)
    # ══════════════════════════════════════════════
    
    execution_trace: Optional[List[Dict[str, Any]]]
    """
    完整执行链路的追踪日志（用于调试、审计、性能分析）
    
    记录粒度：每个节点（Agent）的进出时间和状态快照
    
    单条trace记录结构：
    {
        "node_name": "supervisor",        # 节点名称
        "entry_time": "2026-04-12T23:55:01.123",  # 进入时间
        "exit_time": "2026-04-12T23:55:01.456",   # 退出时间
        "duration_ms": 333,               # 本节点耗时
        "input_state_snapshot": {...},      # 进入时的state副本
        "output_state_diff": {...},        # 退出时相对于进入的变化
        "status": "success",               # 执行状态
        "error": None,                    # 错误信息（如有）
        "tokens_consumed": 150,            # 本节点Token消耗
        "memory_usage_mb": 45.2           # 内存占用（可选，性能 profiling时启用）
    }
    
    使用场景：
    1. 调试：复现问题现场（"在第3步时state变成了什么？"）
    2. 审计：谁在什么时候修改了什么数据？
    3. 性能瓶颈定位：哪个节点最耗时？哪个节点Token消耗最大？
    4. 回滚：能否从某个trace point重新执行？
    
    存储开销：
    - 每条约 1-2KB（轻量级快照）
    - 10次迭代的完整trace约 10-20KB（可接受）
    - 生产环境可通过配置关闭（set to None省内存）
    """
    
    error_log: Optional[List[Dict[str, Any]]]
    """
    全局错误日志（跨所有Agent和模块的错误汇总）
    
    单条错误记录：
    {
        "timestamp": "2026-04-12T23:56:00.789",  # 错误发生时间
        "node": "retrieval_agent",            # 出错的节点/模块
        "error_type": "APIError",            # 错误类型（Exception类名）
        "error_message": "Connection timeout to ChromaDB",  # 错误消息
        "severity": "high",                   # 严重程度（critical/high/medium/low）
        "recoverable": True,                 # 是否可恢复
        "recovery_action": "Retried with exponential backoff",  # 恢复动作
        "stack_trace": "..."                  # 完整堆栈（debug模式下）
        "user_visible": False                # 是否需要向用户展示
    }
    
    错误分级处理：
    - critical: 立即终止流程，返回错误给用户
    - high: 记录日志，尝试降级方案（如HyDE失败→用原query）
    - medium: 重试一次后继续
    - low: 仅记录，不影响主流程
    
    统计价值：
    - 错误率 = len(error_log) / total_operations
    - 可用于SLA监控和告警
    """
    
    performance_metrics: Optional[Dict[str, Any]]
    """
    性能指标汇总（整个流程的资源消耗统计）
    
    结构示例：
    {
        "total_duration_ms": 12500.5,        # 总耗时（毫秒）
        "llm_total_tokens": 8500,            # LLM总Token消耗
        "llm_api_calls": 12,                # LLM API调用次数
        "external_api_calls": 5,            # 外部API调用（Alpha Vantage等）
        "vector_searches": 2,               # 向量检索次数
        "rerank_operations": 1,             # 重排序操作次数
        
        # 分阶段耗时（可选，详细profiling时启用）
        "phase_breakdown": {
            "slot_extraction": 230.5,        # 槽位提取耗时
            "intent_classification": 450.2,  # 意图识别耗时
            "query_rewriting": 680.3,        # Query改写耗时
            "task_planning": 120.0,          # 任务规划耗时
            "rag_retrieval": 2100.5,         # RAG检索耗时
            "react_execution": 6500.0,       # ReAct执行耗时
            "report_generation": 1800.0,     # 报告生成耗时
            "evaluation": 618.0              # 评估耗时
        },
        
        # 资源利用率（高级）
        "resource_utilization": {
            "cpu_peak_percent": 85.2,        # CPU峰值使用率
            "memory_peak_mb": 512.3,          # 内存峰值占用
            "network_bytes_sent": 102400,    # 网络发送量
            "network_bytes_received": 204800  # 网络接收量
        }
    }
    
    应用场景：
    1. 成本核算：本次查询花费了多少Token费用？
    2. SLA监控：响应时间是否符合<15秒的要求？
    3. 容量规划：当前资源配置能否支撑QPS目标？
    4. 优化指导：哪个phase是瓶颈？（通常React占50%+）
    """
    
    # ══════════════════════════════════════════════
    # Layer 9: 评估层 (Evaluation Layer)
    # ══════════════════════════════════════════════
    
    evaluation_scores: Optional[Dict[str, Any]]
    """
    多维度自动化评估得分（由ComprehensiveEvaluator生成）
    
    评估维度（4大维度，12+子指标）：
    
    1. RAG质量评估（基于Ragas框架）：
    {
        "faithfulness": 0.92,          # 忠实度：答案是否忠实于上下文
        "answer_relevancy": 0.88,      # 相关性：是否回答了问题
        "context_precision": 0.85,     # 精确率：检索文档是否有用
        "context_recall": 0.78,        # 召回率：信息是否充分
        "overall_rag_quality": 0.86    # RAG综合分（加权平均）
    }
    
    2. Agent执行质量：
    {
        "goal_achievement": 0.90,       # 目标达成率
        "tool_efficiency": 0.82,        # 工具使用效率
        "reasoning_quality": 0.75,      # 推理质量
        "time_efficiency": 0.88,        # 时效性
        "overall_agent_quality": 0.84  # Agent综合分
    }
    
    3. 输出质量：
    {
        "professionalism": 0.91,        # 专业性
        "data_accuracy": 0.87,          # 数据准确性
        "logical_coherence": 0.83,      # 逻辑连贯性
        "actionability": 0.79,          # 可操作性
        "overall_output_quality": 0.85 # 输出综合分
    }
    
    4. 系统性能（可选，生产环境开启）：
    {
        "latency_percentile_p50": 8500,  # 中位数延迟(ms)
        "latency_percentile_p99": 15000, # P99延迟
        "error_rate": 0.02,              # 错误率
        "throughput_qps": 10,            # 吞吐量
        "cost_per_query": 0.012          # 单次查询成本($)
    }
    
    综合计算：
    overall_score = weighted_avg(
        rag_quality * 0.35 +
        agent_quality * 0.30 +
        output_quality * 0.35
    )
    
    评分等级映射：
    - ≥ 0.9: EXCELLENT（优秀）✨
    - 0.7-0.9: GOOD（良好）👍
    - 0.5-0.7: ACCEPTABLE（可接受）⚠️
    - 0.3-0.5: POOR（较差）❌
    - < 0.3: CRITICAL（严重）🚨
    
    使用场景：
    - 自动质量门控：score < 0.6 时拒绝输出，提示用户重新表述
    - A/B测试对比：不同prompt版本的score对比
    - 持续优化方向：weaknesses字段指出最低分的维度
    """
    
    quality_flags: Optional[List[str]]
    """
    质量标记/警告标签（用于快速判断输出是否可信）
    
    可能的标记：
    - "数据缺失"：RAG检索返回空或过短
    - "置信度低"：slot_confidence < 0.7 或 intent ambiguity
    - "工具失败"：某个关键API调用失败导致数据不全
    - "超时终止"：达到max_iterations被迫截断
    - "内容冲突"：FinancialAgent和RiskAgent结论矛盾
    - "幻觉风险"：faithfulness score < 0.7
    - "数据过期"：realtime_data超过30分钟
    - "格式异常"：最终报告不符合Markdown规范
    
    使用方式：
    if quality_flags:
        warning_msg = "⚠️ 本次回答存在以下问题：\n"
        for flag in quality_flags:
            warning_msg += f"• {flag}\n"
        final_answer = warning_msg + final_report
    
    门控策略：
    flags数量 >= 3时，应在答案开头显著标注"⚠️ 低置信度回答"
    并建议用户通过官方渠道二次核实关键数据。
    """


# ══════════════════════════════════════════════════════════════
# 常量定义区域 (Constants)
# ══════════════════════════════════════════════════════════════

# ── 意图类型常量（8大主意图）─────────────────────────────

INTENT_FINANCIAL_ANALYSIS: str = "financial_analysis"""
"""财务分析意图标识符

适用场景：
- 财报解读（年报、季报、月报）
- 财务指标计算与分析（PE、ROE、现金流等）
- 经营数据分析（收入结构、成本构成）
- 盈利能力/成长性/现金流专项研究

典型用户问题示例：
- "分析贵州茅台2023年年报"
- "比亚迪的毛利率为什么这么高？"
- "计算招商银行的ROE趋势"

下游路由：retrieval_agent → financial_agent → risk_agent → realtime_agent → report
"""

INTENT_STOCK_COMPARISON: str = "stock_comparison"""
"""个股对比分析意图标识符

适用场景：
- 两家/多家公司的横向比较
- 竞争力分析（市场份额、技术壁垒）
- 估值对标（相对估值法）

典型用户问题示例：
- "对比茅台和五粮液的估值水平"
- "比亚迪vs宁德时代，谁更值得投资？"
- "新能源车企前三名竞争力排行"

下游路由：retrieval_agent(multi-company) → financial_agent(comparative) → report
"""

INTENT_INDUSTRY_OVERVIEW: str = "industry_overview"""
"""行业综述意图标识符

适用场景：
- 行业整体情况与发展趋势
- 产业链上下游关系
- 竞争格局与市场集中度
- 政策环境影响分析

典型用户问题示例：
- "新能源汽车行业发展趋势"
- "半导体产业链分析"
- "白酒行业的竞争格局如何？"

下游路由：retrieval_agent(broad) → financial_agent → report
"""

INTENT_REALTIME_QUERY: str = "realtime_query"""
"""实时行情查询意图标识符

适用场景：
- 实时股价查询（当前价、涨跌幅）
- 今日市场动态（成交量、资金流向）
- 最新新闻资讯（突发消息、公告）

典型用户问题示例：
- "茅台今天股价多少"
- "比亚迪现在的价格"
- "今天大盘怎么样？"

特殊路由：直接 → realtime_agent → report（跳过RAG和分析环节，快速响应）
"""

INTENT_PORTFOLIO_ANALYSIS: str = "portfolio_analysis"""
"""投资组合分析意图标识符【增强版新增】

适用场景：
- 持仓组合风险评估
- 资产配置优化建议
- 分散化程度分析
- 再平衡策略制定

典型用户问题示例：
- "我的投资组合风险大不大？"
- "如何优化股票持仓结构？"
- "评估这个组合的收益风险比"

前置需求：需要用户已提供持仓数据（或从账户系统导入）
"""

INTENT_RISK_ALERT: str = "risk_alert"""
"""风险预警意图标识符【增强版新增】

适用场景：
- 个股风险监测（暴雷预警、财务异常）
- 行业系统性风险（政策突变、黑天鹅事件）
- 组合层面风险（集中度过高、流动性不足）

典型用户问题示例：
- "茅台有哪些潜在风险？"
- "这只股票会暴雷吗？"
- "监测比亚迪的负面新闻"

特殊能力：可结合实时舆情数据进行情感分析
"""

INTENT_STRATEGY_BACKTEST: str = "strategy_backtest"""
"""策略回测意图标识符【增强版新增】

适用场景：
- 历史策略表现验证
- 因子有效性测试
- 交易规则绩效评估
- 参数敏感性分析

典型用户问题示例：
- "回测均线交叉策略的表现"
- "价值投资策略的历史收益率"
- "模拟动量策略的收益曲线"

前置需求：需要历史行情数据库（AKShare/Tushare/Wind）
"""

INTENT_MARKET_SENTIMENT: str = "market_sentiment"""
"""市场情绪分析意图标识符【增强版新增】

适用场景：
- 投资者情绪指标（贪婪/恐惧指数）
- 资金流向分析（主力/散户动向）
- 舆情监控（社交媒体、论坛热度）
- 宏观经济预期（PMI、CPI等前瞻指标）

典型用户问题示例：
- "现在市场情绪怎么样？"
- "投资者对新能源怎么看？"
- "分析今日市场舆情"

数据源：雪球/微博/东方财富/巨潮资讯
"""


# ── 子意图分类常量（细粒度意图）─────────────────────────────

SUB_INTENT_PROFITABILITY: str = "profitability_analysis"""
"""盈利能力分析子意图

关注指标：毛利率、净利率、ROE、ROA、EBITDA利润率
分析维度：趋势变化、同业对比、驱动因素拆解
"""

SUB_INTENT_GROWTH: str = "growth_analysis"""
"""成长性分析子意图

关注指标：营收增速、净利润增速、EPS增速、CAGR
分析维度：同比增长、环比增长、季度波动
"""

SUB_INTENT_CASH_FLOW: str = "cashflow_analysis"""
"""现金流分析子意图

关注指标：经营现金流、自由现金流、资本支出
分析维度：现金质量（与利润匹配度）、再投资需求
"""

SUB_INTENT_VALUATION: str = "valuation_analysis"""
"""估值水平分析子意图

关注指标：PE、PB、PS、EV/EBITDA、PEG、股息率
分析维度：历史分位、绝对估值、相对估值（vs同业）
"""

SUB_INTENT_TECHNICAL: str = "technical_analysis"""
"""技术面分析子意图【新增】

关注指标：MA/MACD/KDJ/RSI/BOLL/成交量
分析方法：趋势跟踪、均值回归、动量策略
适用场景：短期交易信号、入场/出场点判断
"""

SUB_INTENT_FUNDAMENTAL: str = "fundamental_analysis"""
"""基本面分析子意图【新增】

关注指标：护城河、管理层质量、商业模式可持续性
分析方法：SWOT、波特五力、杜邦分析
适用场景：长期价值投资、深度研究
"""


# ── 节点名称常量（LangGraph StateGraph节点标识）──────────────

NODE_SUPERVISOR: str = "supervisor"""
"""Supervisor节点：总调度中枢

职责：
1. 意图识别（首次进入时）
2. 任务分配（根据状态决定下一个Agent）
3. 结果审核（Agent完成后回到此处）
4. 流程控制（循环/终止/异常处理）

特殊属性：
- 唯一不执行业务逻辑的节点（纯调度角色）
- 每个流程必然多次经过的枢纽节点
"""

NODE_SLOT_EXTRACTOR: str = "slot_extractor"
"""槽位提取节点：实体识别与结构化

输入：user_query
输出：slots, slot_confidence
依赖：无（流程的第一个业务节点）
"""

NODE_INTENT_CLASSIFIER: str = "intent_classifier"
"""意图分类节点：用户需求理解

输入：user_query, slots（可选，提升准确率）
输出：intent, intent_hierarchy
依赖：可与slot_extractor并行
"""

NODE_TASK_PLANNER: str = "task_planner"
"""任务规划节点：目标分解与编排

输入：intent, slots, query
输出：task_plan, current_task_index
依赖：intent_classifier（必须先有意图才能规划）
"""

NODE_SKILL_SELECTOR: str = "skill_selector"
"""技能选择节点：动态能力匹配

输入：task_plan[current_task], available_skills
输出：selected_skills, skill_parameters
依赖：task_planner（必须有任务才能选技能）
"""

NODE_RETRIEVAL_AGENT: str = "retrieval_agent"
"""RAG检索Agent：知识库检索

输入：current_query（改写后的）
输出：rag_context（Markdown格式的检索上下文）
依赖：task_planner（通常是第一个执行的业务Agent）
"""

NODE_FINANCIAL_AGENT: str = "financial_agent"
"""财务分析Agent：专业财务解读

输入：rag_context, user_query
输出：financial_analysis（结构化分析报告）
依赖：retrieval_agent（需要检索到的财报数据）
"""

NODE_RISK_AGENT: str = "risk_agent"
"""风控Agent：独立风险评估

输入：rag_context, financial_analysis（挑战对象）
输出：risk_assessment（独立的风险报告）
依赖：financial_agent（需要财务分析结论来挑战）
设计原则：故意唱反调，防止回声室效应
"""

NODE_REALTIME_AGENT: str = "realtime_agent"
"""实时数据Agent：市场行情获取

输入：user_query（或slots中的公司信息）
输出：realtime_data（股价+新闻摘要）
特点：唯一真正使用ReAct循环的Agent（create_react_agent）
依赖：可与其他Agent并行（无需等待RAG结果）
"""

NODE_QUANTITATIVE_AGENT: str = "quantitative_agent"
"""量化分析Agent：精确计算引擎【新增】

输入：rag_context中的数值数据
输出：quantitative_result（结构化计算结果，Dict格式）
特点：纯Python运算，不调用LLM（避免数学幻觉）
依赖：retrieval_agent（需要财报中的原始数字）
"""

NODE_REPORT_GENERATOR: str = "report_generator"
"""报告生成器：最终产出节点

输入：aggregated_results（所有Agent输出的汇总）
输出：final_report（Markdown格式研报）
依赖：所有主要Agent完成后（financial+risk+realtime+quantitative）
"""

NODE_EVALUATOR: str = "evaluator"
"""质量评估节点：自动化评测【新增】

输入：final_report, query, contexts
输出：evaluation_scores（多维度的0-1分数）
依赖：report_generator（必须有报告才能评估）
特点：可选节点，不影响主流程（仅用于质量监控和持续优化）
"""

NODE_END: str = "__end__"
"""终止节点：流程结束标志

触发条件：
1. final_report已生成且质量达标
2. 达到max_iterations上限
3. 用户主动终止（输入quit/exit）
4. 发生不可恢复的致命错误
"""


# ── 控制参数常量（系统配置）────────────────────────────────────

MAX_ITERATIONS: int = 15
"""
全局最大迭代次数（Supervisor循环上限）

安全考虑：
- 防止无限循环（路由错误或状态异常时）
- 控制最大资源消耗（Token和时间）
- 保证系统可响应性（不会永久阻塞）

调优建议：
- 开发调试：设为100（确保不会中途停止）
- 生产环境：设为15-20（平衡完整性和速度）
- 快速通道（realtime）：设为5-8
"""

MAX_REACT_STEPS: int = 10
"""
单个Agent内ReAct循环的最大步数

组成部分：
- Thought（思考）：LLM推理
- Action（行动）：工具/Skill调用
- Observation（观察）：结果处理

资源消耗估算：
- 每步约消耗 800-1200 tokens（LLM推理）
- 每步约耗时 500-2000ms（含网络IO）
- 10步总计约 10k tokens + 10-20秒

终止策略（任一触发即停）：
1. 达到max_react_steps
2. LLM生成Final Answer
3. 连续2步Observation无新增信息
4. 工具调用连续失败3次
"""

MAX_QUERY_REWRITES: int = 3
"""
Query改写的最大轮数上限

改写策略序列（固定3轮）：
Round 1: standardize（标准化术语+槽位填充）
Round 2: specialize（针对目标数据源优化）
Round 3: expand/simplify（同义词扩展或去噪简化）

为何限制为3轮：
- 第1轮后通常已有显著改善（召回率提升20-40%）
- 第2轮针对特定数据源优化（精准度提升10-20%）
- 第3轮边际收益递减（通常<5%改善）
- 更多轮次会导致延迟增加但收益甚微

Token成本：
- 每轮改写约消耗 500-800 tokens（LLM调用）
- 3轮总计约 1500-2400 tokens（可接受的额外开销）
"""

SLOT_CONFIDENCE_THRESHOLD: float = 0.7
"""
槽位提取的置信度阈值（触发澄清流程的门限值）

阈值含义：
- ≥ 0.7: 高置信度，直接使用提取结果继续流程
- < 0.7: 低置信度，必须向用户确认后再继续

低置信度的常见原因：
1. 问题过于模糊（"分析一下那个公司"）
2. 缺少关键实体（未提及公司名或年份）
3. 多意性歧义（"苹果"可能是公司也可能是水果）
4. 领域外问题（超出金融词典覆盖范围）

澄清流程（当confidence < threshold时）：
1. 从clarification_questions中选择最关键的1-2个问题
2. 向用户展示这些问题
3. 等待用户回答后更新slots
4. 重新评估confidence，若仍< threshold则使用默认值兜底

示例交互：
User: "分析一下它的业绩"
System: ⚠️ 请确认：您想了解哪家公司的业绩？
  [ ] 贵州茅台 (600519.SH)
  [ ] 比亚迪 (002594.SZ)
  [ ] 其他公司（请输入名称）
User: [x] 贵州茅台
System: ✅ 已确认为"贵州茅台"，继续分析...
"""
