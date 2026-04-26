# Task Plan: AI-Agent-Finance 代码库修复计划

## Goal
对 `/Users/liuyang/Desktop/AIAgent/AI-agent-finance` 代码库进行全面修复，解决 6 大模块中的 22 个严重/架构性问题。

## Current Phase
Phase 1: 修复 MCP 层与 Server（最严重断裂点）

## Phases

### Phase 1: 修复 MCP 层 (`src/mcp/server.py`)
- [ ] 修复 `get_mcp_tools()` 返回 `None` 的问题
- [ ] 实现正确的 tools 返回逻辑
- [ ] 修复 `call_tool` 硬编码路由
- **Status:** pending

### Phase 2: 修复 ReportGenerator 集成 (`report/generator.py` + `main_enhanced.py`)
- [ ] 在 `main_enhanced.py` 中添加 `_phase_report_generation()` 调用
- [ ] 修复 key 不匹配：`rag_contexts` vs `rag_context`
- [ ] 确保 `final_report` 被正确读取并返回
- **Status:** pending

### Phase 3: 修复融合逻辑 (`enhanced_rag/enhanced_rag_pipeline.py`)
- [ ] 实现真正的 RRF 融合算法
- [ ] 实现加权融合（使用配置中的 `fusion_weights`）
- [ ] 根据 `RAGConfig.fusion_strategy` 选择融合方式
- **Status:** pending

### Phase 4: 修复 TaskPlanner LLM fallback (`core/task_planner.py`)
- [ ] 当模板不匹配时，调用 LLM 生成任务计划
- [ ] 确保 `planning_strategy` 正确标记为 `llm_generated`
- **Status:** pending

### Phase 5: 修复 ReAct 虚假声明 (`agents/financial_agent.py`, `agents/retrieval_agent.py`)
- [ ] `financial_agent.py`: 实现真正的 ReAct 循环（自我审查+补充检索）
- [ ] `retrieval_agent.py`: 实现真正的 action→observation→reasoning 循环
- [ ] 移除文档中的 "ReAct" 声明如果无法实现
- **Status:** pending

### Phase 6: 修复 Supervisor 路由 (`agents/supervisor.py`, `agents/graph.py`)
- [ ] 确认 `report_generator` 节点是否存在，如不存在则修复路由目标
- [ ] 实现 `_review_and_dispatch` 的真实质量审查逻辑
- **Status:** pending

### Phase 7: 清理并行评估器 (`evaluation/evaluator.py` + `enhanced_evaluation/enhanced_evaluator.py`)
- [ ] 统一评估接口或明确两者关系
- [ ] 移除 `enhanced_evaluator.py` 中 `use_llm=True` 但从未使用 LLM 的假象
- [ ] 实现 `SystemPerformanceEvaluator` 和 `UserSatisfactionPredictor` 或从架构图中移除
- **Status:** pending

### Phase 8: 修复 convenience 函数内存泄漏
- [ ] `intent_classifier.py`: `classify_intent()` 复用单例而非每次 new
- [ ] 检查其他模块的类似问题
- **Status:** pending

### Phase 9: 修复 `query_rewriter.py` 危险静默降级
- [ ] import 失败时抛出异常而非静默设置空 dict
- [ ] 修复 `_evaluate_query_quality` 使用 slot 信息而非重新匹配
- **Status:** pending

### Phase 10: 最终验证与清理
- [x] 验证所有模块间调用链完整
- [x] 移除所有 dead code
- [x] 更新 README 中的性能数据（添加"需实际 benchmark 验证"标注）
- **Status:** complete

## Key Questions
1. `report_generator` 节点应该存在于 graph.py 中还是应该从 supervisor 路由中移除？
2. `FusionStrategy.RRF` 实现时，k 参数建议值是多少（通常 60）？
3. TaskPlanner 的 LLM fallback prompt 是否需要专门设计？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| MCP tools 直接 import 而非通过 `get_mcp_tools()` 获取 | 当前 `get_mcp_tools()` 返回 None，直接 import 是实际工作方式 |
| RRF 融合实现时使用 k=60（标准值） | 参考文献推荐范围 40-100，60 是安全默认值 |
| ReportGenerator 作为独立调用而非 LangGraph 节点 | 避免大幅重构，保持现有架构 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `get_mcp_tools()` 返回 None | 1 | 需要重写函数逻辑，正确返回 tools 列表 |
| RRF 枚举值存在但从未被调用 | 1 | 需要在 `query()` 方法中根据 `fusion_strategy` 分派 |
| `rag_contexts` vs `rag_context` key 不匹配 | 1 | 统一为 `rag_contexts`，修复合并逻辑 |