"""
AI 金融研究助手 — Web 服务入口 (v2.0)
====================================

提供 RESTful API 和交互式 Web 界面。
启动: python web_app.py
访问: http://localhost:8000
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import uuid

app = FastAPI(title="AI 金融研究助手 v2.0", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    use_rag: bool = False
    max_iterations: int = 10


class QueryResponse(BaseModel):
    query_id: str = ""
    success: bool = False
    query: str = ""
    final_answer: Optional[str] = None
    slots: Optional[Dict] = None
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None
    task_plan: Optional[List] = None
    execution_steps: List = []
    step_count: int = 0
    tool_call_count: int = 0
    evaluation: Optional[Dict] = None
    response_time_ms: float = 0.0
    stop_reason: Optional[str] = None
    timestamp: str = ""
    errors: List[str] = []
    warnings: List[str] = []


# ════════════════════════════════════════════════════════════
# 全局助手实例
# ════════════════════════════════════════════════════════════

assistant_instance = None


def get_assistant():
    global assistant_instance
    if assistant_instance is None:
        from main_enhanced import EnhancedFinancialAssistant, SystemConfig
        config = SystemConfig(verbose=True)
        assistant_instance = EnhancedFinancialAssistant(config=config)
        print("[Web] ✅ 助手初始化完成")
    return assistant_instance


# ════════════════════════════════════════════════════════════
# API 端点
# ════════════════════════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def index():
    """Web 首页"""
    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 金融研究助手</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.container{background:white;border-radius:20px;box-shadow:0 25px 50px rgba(0,0,0,.15);width:100%;max-width:900px;padding:40px}
.header{text-align:center;margin-bottom:30px}
.header h1{font-size:28px;color:#333;margin-bottom:8px}
.header p{color:#666;font-size:14px}
.capabilities{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:20px 0}
.capability{background:linear-gradient(135deg,#f5f7fa,#c3cfe2);padding:6px 14px;border-radius:20px;font-size:12px;color:#555}
.input-area{margin-bottom:30px}
.query-input{width:100%;padding:16px 20px;border:2px solid #e0e0e0;border-radius:12px;font-size:16px;transition:all .3s;resize:vertical;min-height:80px}
.query-input:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 4px rgba(102,126,234,.1)}
.options{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap}
.option-group{flex:1;min-width:150px}
.option-group label{display:block;font-size:13px;color:#666;margin-bottom:6px;font-weight:600}
.option-group select,.option-group input{width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:8px;font-size:14px}
.submit-btn{width:100%;padding:16px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none;border-radius:12px;font-size:16px;font-weight:600;cursor:pointer;transition:all .3s}
.submit-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 10px 30px rgba(102,126,234,.3)}
.submit-btn:disabled{opacity:.7;cursor:not-allowed}
.result-section{display:none;margin-top:30px;padding-top:30px;border-top:2px solid #f0f0f0}
.result-section.visible{display:block}
.result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.result-header h2{font-size:18px;color:#333}
.status-badge{padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600}
.status-success{background:#d4edda;color:#155724}
.status-error{background:#f8d7da;color:#721c24}
.status-processing{background:#fff3cd;color:#856404}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.metric-card{background:#f8f9fa;padding:14px;border-radius:10px;text-align:center}
.metric-label{font-size:12px;color:#888;margin-bottom:4px}
.metric-value{font-size:18px;font-weight:700;color:#333}
.answer-box{background:#f8f9fa;border-radius:12px;padding:24px;line-height:1.8;white-space:pre-wrap;word-break:break-word;max-height:500px;overflow-y:auto}
.steps-timeline{margin-top:20px}
.step-item{display:flex;gap:12px;padding:12px 0;border-left:3px solid #e0e0e0;padding-left:20px;position:relative}
.step-item:before{content:'';position:absolute;left:-8px;top:14px;width:13px;height:13px;border-radius:50%;background:white;border:3px solid #ccc}
.step-thought:before{border-color:#667eea}
.step-action:before{border-color:#28a745}
.step-finish:before{border-color:#ffc107}
.step-icon{font-size:18px;min-width:28px}
.step-content{flex:1}
.step-type{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px}
.step-text{font-size:14px;color:#444;margin-top:4px}
.loading-spinner{display:inline-block;width:20px;height:20px;border:3px solid #f3f3f3;border-top:3px solid #667eea;border-radius:50%;animation:spin 1s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.examples{margin-top:30px;padding-top:20px;border-top:1px dashed #ddd}
.examples h3{font-size:14px;color:#888;margin-bottom:12px}
.example-btn{display:inline-block;padding:8px 16px;margin:4px;background:#f5f5f5;border:1px solid #e0e0e0;border-radius:8px;cursor:pointer;font-size:13px;color:#555;transition:all .2s}
.example-btn:hover{background:#e8e8ff;border-color:#667eea;color:#667eea}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>🚀 AI 金融研究助手</h1>
<p>企业级智能投研与研报自动化系统 (Enhanced v2.0)</p>
<div class="capabilities">
<span class="capability">✅ Supervisor+SubAgent</span>
<span class="capability">✅ 智能槽位提取</span>
<span class="capability">✅ 多轮 Query 改写</span>
<span class="capability">✅ 8 类意图识别</span>
<span class="capability">✅ 动态任务规划</span>
<span class="capability">✅ 手动 ReAct 循环</span>
<span class="capability">✅ 增强版 RAG</span>
<span class="capability">✅ 多维质量评估</span>
</div>
</div>

<div class="input-area">
<textarea id="queryInput" class="query-input" placeholder="输入您的研究需求...

示例：
- 分析贵州茅台2023年的盈利能力
- 茅台当前的股价是多少？
- 比较茅台和五粮液的估值水平"></textarea>

<div class="options">
<div class="option-group">
<label>RAG 增强</label>
<select id="useRag"><option value="false">关闭</option><option value="true">开启</option></select>
</div>
<div class="option-group">
<label>最大迭代次数</label>
<select id="maxIterations"><option value="5">5次(快速)</option><option value="10" selected>10次(标准)</option><option value="15">15次(深度)</option></select>
</div>
</div>

<button id="submitBtn" class="submit-btn" onclick="submitQuery()">🚀 开始分析</button>
</div>

<div id="resultSection" class="result-section">
<div class="result-header"><h2>📊 分析结果</h2><span id="statusBadge" class="status-badge status-processing"><span class="loading-spinner"></span>处理中...</span></div>
<div id="metricsGrid" class="metrics-grid"></div>
<h3 style="margin:20px 0 12px;font-size:16px;color:#555">💡 研究报告</h3>
<div id="answerBox" class="answer-box"><em style="color:#999">等待分析...</em></div>
<h3 style="margin:20px 0 12px;font-size:16px;color:#555">🔄 执行流程</h3>
<div id="stepsTimeline" class="steps-timeline"></div>
</div>

<div class="examples">
<h3>💡 快速体验</h3>
<button class="example-btn" onclick="setQuery('分析贵州茅台2023年的盈利能力')">📊 盈利能力分析</button>
<button class="example-btn" onclick="setQuery('茅台当前的股价是多少？')">💰 实时行情查询</button>
<button class="example-btn" onclick="setQuery('比较茅台和五粮液的估值水平')">⚖️ 公司对比分析</button>
<button class="example-btn" onclick="setQuery('白酒行业最近有什么新闻？')">📰 行业资讯搜索</button>
</div>
</div>

<script>
function setQuery(text) {
    document.getElementById('queryInput').value = text;
    submitQuery();
}

async function submitQuery() {
    var query = document.getElementById('queryInput').value.trim();
    if (!query) { alert('请输入您的问题'); return; }
    
    var btn = document.getElementById('submitBtn');
    var resultSection = document.getElementById('resultSection');
    var statusBadge = document.getElementById('statusBadge');
    var metricsGrid = document.getElementById('metricsGrid');
    var answerBox = document.getElementById('answerBox');
    var stepsTimeline = document.getElementById('stepsTimeline');
    
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>正在分析...';
    resultSection.className = 'result-section visible';
    statusBadge.className = 'status-badge status-processing';
    statusBadge.innerHTML = '<span class="loading-spinner"></span>处理中...';
    metricsGrid.innerHTML = '';
    answerBox.innerHTML = '<em style="color:#999;">正在调用 AI 引擎...</em>';
    stepsTimeline.innerHTML = '';
    
    var useRag = document.getElementById('useRag').value === 'true';
    var maxIterations = parseInt(document.getElementById('maxIterations').value);
    
    console.log('[DEBUG] 发送请求:', {query: query, useRag: useRag, maxIterations: maxIterations});
    
    try {
        var response = await fetch('/api/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query, use_rag: useRag, max_iterations: maxIterations})
        });
        
        console.log('[DEBUG] 响应状态:', response.status);
        
        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }
        
        var data = await response.json();
        console.log('[DEBUG] 响应数据:', data);
        
        statusBadge.className = data.success ? 'status-badge status-success' : 'status-badge status-error';
        statusBadge.innerHTML = data.success ? '✅ 完成' : '❌ 失败';
        
        var metricsHTML = '';
        metricsHTML += '<div class="metric-card"><div class="metric-label">意图识别</div><div class="metric-value">' + (data.intent || '-') + '</div></div>';
        metricsHTML += '<div class="metric-card"><div class="metric-label">执行步骤</div><div class="metric-value">' + data.step_count + ' 轮</div></div>';
        metricsHTML += '<div class="metric-card"><div class="metric-label">工具调用</div><div class="metric-value">' + data.tool_call_count + ' 次</div></div>';
        metricsHTML += '<div class="metric-card"><div class="metric-label">响应耗时</div><div class="metric-value">' + (data.response_time_ms / 1000).toFixed(1) + 's</div></div>';
        
        if (data.evaluation && data.evaluation.overall_score !== undefined) {
            var score = data.evaluation.overall_score || 0;
            var level = data.evaluation.overall_level || '-';
            var emojis = {'excellent':'🟢','good':'🟡','acceptable':'🟠','poor':'🔴'};
            metricsHTML += '<div class="metric-card"><div class="metric-label">质量评分</div><div class="metric-value">' + score.toFixed(4) + ' ' + (emojis[level.toLowerCase()]||'') + '</div></div>';
        }
        
        metricsGrid.innerHTML = metricsHTML;
        
        if (data.final_answer) {
            answerBox.innerHTML = data.final_answer.replace(/\n/g, '<br>');
        } else if (!data.success) {
            var errs = data.errors || [];
            answerBox.innerHTML = '<span style="color:#c00;">' + (errs.join('<br>') || '处理失败') + '</span>';
        } else {
            answerBox.innerHTML = '<em style="color:#999;">无返回内容</em>';
        }
        
        if (data.execution_steps && data.execution_steps.length > 0) {
            var stepsHTML = '';
            for (var i = 0; i < data.execution_steps.length; i++) {
                var step = data.execution_steps[i];
                var icons = {'thought':'💭','action':'🔧','observation':'👁️','finish':'✅'};
                var labels = {'thought':'思考','action':'行动','observation':'观察','finish':'完成'};
                var icon = icons[step.type] || '❓';
                var label = labels[step.type] || step.type;
                var content = (step.content || step.tool || '').substring(0, 200);
                
                var cls = 'step-item';
                if (step.type === 'thought') cls += ' step-thought';
                else if (step.type === 'action') cls += ' step-action';
                else if (step.type === 'finish') cls += ' step-finish';
                
                stepsHTML += '<div class="' + cls + '">';
                stepsHTML += '<div class="step-icon">' + icon + '</div>';
                stepsHTML += '<div class="step-content">';
                stepsHTML += '<div class="step-type">' + label + ' · Step ' + (i+1) + '</div>';
                stepsHTML += '<div class="step-text">' + content + '</div>';
                stepsHTML += '</div></div>';
            }
            stepsTimeline.innerHTML = stepsHTML;
        }
        
    } catch (err) {
        console.error('[ERROR]', err);
        statusBadge.className = 'status-badge status-error';
        statusBadge.innerHTML = '❌ 请求失败';
        answerBox.innerHTML = '<span style="color:#c00;">错误: ' + err.message + '</span>';
    }
    
    btn.disabled = false;
    btn.innerHTML = '🚀 开始分析';
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """核心查询 API"""
    start_time = time.time()
    query_id = f"q_{uuid.uuid4().hex[:8]}"
    
    print(f"\n[API] 📝 [{query_id}]: {request.query[:60]}...")
    
    try:
        assistant = get_assistant()
        result = assistant.process_query(user_query=request.query)
        
        resp_time = (time.time() - start_time) * 1000
        
        resp = QueryResponse(
            query_id=query_id,
            success=result.get("success", False),
            query=result.get("query", request.query),
            final_answer=result.get("final_answer"),
            slots=result.get("slots"),
            intent=result.get("intent"),
            rewritten_query=result.get("rewritten_query"),
            task_plan=result.get("task_plan"),
            execution_steps=[
                {
                    "type": s.get("type", "thought") if isinstance(s, dict) else str(s),
                    "content": (
                        s.get("thought", "") or s.get("observation", "")
                    )[:200] if isinstance(s, dict) else str(s)[:200],
                    "tool": s.get("action", "") if isinstance(s, dict) else "",
                }
                for s in result.get("execution_steps", [])
                if isinstance(s, dict)
            ],
            step_count=result.get("step_count", 0),
            tool_call_count=result.get("tool_call_count", 0),
            evaluation=result.get("evaluation"),
            response_time_ms=resp_time,
            stop_reason=result.get("stop_reason"),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
        )
        
        print(f"[API] ✅ 完成 ({resp_time:.0f}ms)")
        return JSONResponse(content=resp.dict())
        
    except Exception as e:
        print(f"[API] ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🌐 AI 金融研究助手 — Web 服务")
    print("=" * 60)
    print("  http://localhost:8000      ← Web 界面")
    print("  http://localhost:8000/docs   ← API 文档")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
