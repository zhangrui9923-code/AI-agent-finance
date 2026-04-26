# Findings: AI-Agent-Finance 代码库分析

## 用户提出的三个问题（已确认）

### 1. 打分机制冗余
**确认：部分正确**。RRF/加权融合全部未实现，`fusion_weights` 配置了但实际融合逻辑未使用。

### 2. TaskPlanner LLM fallback 没实现
**完全确认**。`_llm` 在 `__init__` 时初始化了，但 `plan()` 方法从未调用它。不匹配时直接回退到 `financial_analysis` 硬编码模板。

### 3. README 性能提升无 baseline
**完全确认**。代码库中无任何 benchmark 脚本、对比实验或基线测量代码。

### 4. RRF 从未被调用
**完全确认**。三通道用不同 key（`vec_`, `hyde_`, `bm25_`），永远不会碰撞。实际融合只有 key 碰撞时一次 `score * 0.6 + result.score * 0.4`。

---

## 严重问题清单（按模块）

### MCP 层断裂
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `server.py:116-136` | `get_mcp_tools()` 返回 `None` | CRITICAL |
| `server.py:116-136` | MultiServerMCPClient 创建后未返回 tools | CRITICAL |
| `tools.py` | Alpha Vantage 每日 25 次限流无保护 | CRITICAL |
| `main_enhanced.py:405-409` | 直接 import tools 绕过 MCP 层 | HIGH |

### ReportGenerator 从未调用
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `main_enhanced.py` | `_phase_report_generation()` 未定义也未调用 | CRITICAL |
| `main_enhanced.py:849` | `rag_contexts` 存入但 `generator.py:83` 用 `rag_context` 读取 | CRITICAL |
| `main_enhanced.py:1069-1093` | `_build_result()` 未读取 `final_report` | CRITICAL |
| `generator.py:83` | 使用 `state.get("rag_context")` 而非 `rag_contexts` | CRITICAL |

### 融合逻辑问题
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `enhanced_rag_pipeline.py:1238-1276` | 三通道不同 key，score 直接 passthrough | HIGH |
| `RAGConfig.fusion_strategy` | 配置了但 `query()` 方法未根据此值选择算法 | HIGH |
| `DEFAULT_FUSION_WEIGHTS` | 定义了权重但实际融合未使用 | MEDIUM |

### ReAct 虚假声明
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `financial_agent.py` | 声称 ReAct，实际只有单次 LLM 调用 | HIGH |
| `retrieval_agent.py` | 声称 ReAct，实际只有 2 次顺序调用 | HIGH |
| `supervisor.py:82` | 路由到 `"report_generator"` 但该节点不存在 | CRITICAL |

### 评估器问题
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `enhanced_evaluator.py` | 架构图 5 维度，只实现 3 个 | HIGH |
| `enhanced_evaluator.py:602-632` | `use_llm=True` 但评分全是启发式规则，LLM 未被调用 | HIGH |
| `evaluator.py` vs `enhanced_evaluator.py` | 两套并行评估器，从不互通 | MEDIUM |

### 其他架构问题
| 文件 | 问题 | 严重度 |
|------|------|--------|
| `intent_classifier.py:1236-1260` | `classify_intent()` 每次 new ChatOpenAI client | MEDIUM |
| `query_rewriter.py:1227-1232` | import 失败静默降级为空 dict | HIGH |
| `slot_extractor.py:1299` | 代码限制 3 个 metric，docstring 说 5 个 | MEDIUM |
| `enhanced_state.py:1048-1180` | Intent 字符串常量与 `intent_classifier.py` 的 Enum 重复 | LOW |

---

## 关键代码位置

### 融合逻辑（最需修复）
```
enhanced_rag_pipeline.py:1219-1280
- all_candidates[key] 使用不同前缀，永不碰撞
- key 碰撞时只有 0.6/0.4 混合，无 RRF/加权
```

### TaskPlanner LLM fallback
```
task_planner.py:901-904
- template 为 None 时直接 fallback 到 financial_analysis
- 从未调用 self._llm
```

### ReportGenerator 集成
```
main_enhanced.py:缺失 _phase_report_generation()
generator.py:83 state.get("rag_context") vs main:849 state["rag_contexts"]
```

---

## 修复优先级排序

1. **CRITICAL**（会导致功能完全不可用）:
   - `get_mcp_tools()` 返回 None
   - ReportGenerator 从未调用 + key 不匹配
   - Supervisor 路由到不存在的节点

2. **HIGH**（功能受损）:
   - 融合逻辑未实现 RRF/加权
   - ReAct 虚假声明
   - `query_rewriter.py` 静默降级

3. **MEDIUM**（性能/最佳实践）:
   - convenience 函数内存泄漏
   - 两套评估器并存
   - `slot_extractor.py` metric 数量矛盾