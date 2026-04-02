"""
MCP 服务器

将 tools.py 中的三类金融工具注册为 MCP 标准工具，
通过 stdio 传输层暴露给 LangGraph Agent 调用。

启动方式：
    python -m src.mcp.server

Agent 侧通过 StdioServerParameters 连接此服务器：
    见本文件末尾的 get_mcp_tools() 工具函数。
"""

from __future__ import annotations

import asyncio
import json

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

from src.mcp.tools import (
    StockQuoteInput, get_stock_quote,
    NewsInput, get_financial_news,
    FinancialCalcInput, calculate_financial_metric,
)


# ── MCP Server 实例 ───────────────────────────────────────────────────────────

if not _MCP_AVAILABLE:
    raise ImportError(
        "MCP 服务器需要安装 mcp 包，请运行：pip install mcp langchain-mcp-adapters"
    )

app = Server("financial-tools-server")


# ── 工具注册：列出所有可用工具 ─────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    """向 MCP 客户端（Agent）声明本服务器提供的工具列表"""
    return [
        Tool(
            name="get_stock_quote",
            description="获取指定股票的实时行情，包括最新价、涨跌幅、成交量",
            inputSchema=StockQuoteInput.model_json_schema(),
        ),
        Tool(
            name="get_financial_news",
            description="获取与公司或行业相关的最新金融新闻及情感倾向分析",
            inputSchema=NewsInput.model_json_schema(),
        ),
        Tool(
            name="calculate_financial_metric",
            description=(
                "精确计算财务指标（市盈率/市净率/ROE/毛利率/净利润率/"
                "资产负债率/CAGR/EV-EBITDA），使用精确数学运算避免 LLM 幻觉"
            ),
            inputSchema=FinancialCalcInput.model_json_schema(),
        ),
    ]


# ── 工具调用：路由到对应实现函数 ───────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    接收 Agent 的工具调用请求，路由到对应的实现函数，返回 JSON 文本结果。

    所有异常均捕获后以 JSON error 格式返回，让 Agent 可以感知错误并重试。
    """
    try:
        if name == "get_stock_quote":
            validated = StockQuoteInput(**arguments)
            result = get_stock_quote(validated.symbol)

        elif name == "get_financial_news":
            validated = NewsInput(**arguments)
            result = get_financial_news(validated.query, validated.limit)

        elif name == "calculate_financial_metric":
            validated = FinancialCalcInput(**arguments)
            result = calculate_financial_metric(validated.metric, validated.params)

        else:
            raise ValueError(f"未知工具：{name}")

        # Pydantic model → JSON 字符串返回
        return [TextContent(type="text", text=result.model_dump_json(ensure_ascii=False))]

    except Exception as e:
        error_payload = json.dumps({"error": str(e), "tool": name}, ensure_ascii=False)
        return [TextContent(type="text", text=error_payload)]


# ── 供 LangGraph Agent 使用的连接工具函数 ─────────────────────────────────────

def get_mcp_tools():
    """
    以 LangChain Tool 列表的形式返回所有 MCP 工具，供 Agent 节点直接绑定。

    使用 langchain_mcp_adapters 将 MCP 协议工具转换为 LangChain BaseTool，
    Agent 无需关心底层 MCP 通信细节。

    Returns:
        list[BaseTool]  LangChain 工具列表
    """
    import sys
    from pathlib import Path
    from langchain_mcp_adapters.client import MultiServerMCPClient
    try:
        from langchain_mcp_adapters.tools import mcp_to_langchain
    except ImportError:
        mcp_to_langchain = None

    server_script = str(Path(__file__).resolve())

    client = MultiServerMCPClient({
        "financial": {
            "command": sys.executable,
            "args": [server_script],
            "transport": "stdio",
        }
    })

    async def _load():
        await client.__aenter__()
        mcp_tools = await client.get_tools()
        if mcp_to_langchain is None:
            return mcp_tools
        return mcp_to_langchain(mcp_tools)

    return asyncio.run(_load())


# ── 入口：以独立进程运行 MCP Server ──────────────────────────────────────────

async def _main():
    """通过 stdio 传输层启动 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
