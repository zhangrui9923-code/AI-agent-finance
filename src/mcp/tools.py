"""
MCP 金融工具实现层

定义所有工具的 Pydantic Schema 和业务逻辑。
工具分三类：
  1. 实时行情工具 — 获取股价、涨跌幅
  2. 新闻资讯工具 — 获取个股/行业最新新闻
  3. 财务计算器   — 精确计算财务指标（避免 LLM 数学幻觉）

数据源：Alpha Vantage（免费 API，FINANCE_API_KEY 配置在 .env）

【限流说明】Alpha Vantage 免费版每日 25 次请求限制。
当前使用 tenacity 重试（最多3次），但无法防止日限额耗尽。
如需高频率使用，建议申请 Premium API Key 或接入其他数据源。
"""

from __future__ import annotations

import os
import math
import httpx
from datetime import datetime
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic import BaseModel, Field


# ── 通用配置 ──────────────────────────────────────────────────────────────────

FINANCE_API_KEY = os.getenv("FINANCE_API_KEY", "demo")
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

# HTTP 客户端（设置超时，避免 Agent 长时间阻塞）
_http_client = httpx.Client(timeout=15.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 实时行情工具
# ═══════════════════════════════════════════════════════════════════════════════

class StockQuoteInput(BaseModel):
    """获取实时股价的输入参数"""
    symbol: str = Field(..., description="股票代码，A股加交易所后缀，如 600519.SH；美股直接填 AAPL")


class StockQuoteOutput(BaseModel):
    """实时股价输出"""
    symbol: str = Field(..., description="股票代码")
    price: float = Field(..., description="最新价格（元/股）")
    change: float = Field(..., description="涨跌额")
    change_pct: float = Field(..., description="涨跌幅（%）")
    volume: int = Field(..., description="成交量（手）")
    timestamp: str = Field(..., description="行情时间戳")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def get_stock_quote(symbol: str) -> StockQuoteOutput:
    """
    获取实时股价行情

    使用 Alpha Vantage GLOBAL_QUOTE 接口，失败自动重试最多 3 次。

    Args:
        symbol: 股票代码（如 AAPL、600519.SH）

    Returns:
        StockQuoteOutput 行情数据
    """
    # A 股代码转换：600519.SH → 600519（Alpha Vantage 格式）
    av_symbol = symbol.split(".")[0] if "." in symbol else symbol

    resp = _http_client.get(ALPHA_VANTAGE_BASE, params={
        "function": "GLOBAL_QUOTE",
        "symbol": av_symbol,
        "apikey": FINANCE_API_KEY,
    })
    resp.raise_for_status()
    data = resp.json().get("Global Quote", {})

    if not data:
        raise ValueError(f"未获取到 {symbol} 的行情数据，请检查股票代码或 API Key")

    price = float(data.get("05. price", 0))
    change = float(data.get("09. change", 0))
    change_pct_str = data.get("10. change percent", "0%").replace("%", "")

    return StockQuoteOutput(
        symbol=symbol,
        price=price,
        change=change,
        change_pct=float(change_pct_str),
        volume=int(float(data.get("06. volume", 0))),
        timestamp=data.get("07. latest trading day", datetime.now().strftime("%Y-%m-%d")),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 新闻资讯工具
# ═══════════════════════════════════════════════════════════════════════════════

class NewsInput(BaseModel):
    """获取新闻的输入参数"""
    query: str = Field(..., description="搜索关键词，如公司名称、行业名称")
    limit: int = Field(default=5, ge=1, le=20, description="返回新闻条数，默认 5 条")


class NewsItem(BaseModel):
    """单条新闻"""
    title: str = Field(..., description="新闻标题")
    summary: str = Field(..., description="新闻摘要")
    source: str = Field(..., description="新闻来源")
    url: str = Field(..., description="原文链接")
    published_at: str = Field(..., description="发布时间")
    sentiment: Optional[str] = Field(None, description="情感倾向：Bullish / Bearish / Neutral")


class NewsOutput(BaseModel):
    """新闻查询输出"""
    query: str = Field(..., description="原始查询关键词")
    total: int = Field(..., description="返回新闻条数")
    items: list[NewsItem] = Field(..., description="新闻列表")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def get_financial_news(query: str, limit: int = 5) -> NewsOutput:
    """
    获取金融新闻资讯

    使用 Alpha Vantage NEWS_SENTIMENT 接口，按情感得分排序返回最相关新闻。

    Args:
        query: 搜索关键词
        limit: 返回条数

    Returns:
        NewsOutput 新闻列表
    """
    resp = _http_client.get(ALPHA_VANTAGE_BASE, params={
        "function": "NEWS_SENTIMENT",
        "tickers": query,      # Alpha Vantage 用 tickers 参数过滤
        "limit": min(limit * 2, 50),  # 多拉一些，方便后续过滤
        "apikey": FINANCE_API_KEY,
    })
    resp.raise_for_status()
    feed = resp.json().get("feed", [])

    items = []
    for article in feed[:limit]:
        # 取最相关 ticker 的情感分析
        ticker_sentiments = article.get("ticker_sentiment", [])
        sentiment_label = None
        if ticker_sentiments:
            sentiment_label = ticker_sentiments[0].get("ticker_sentiment_label")

        items.append(NewsItem(
            title=article.get("title", ""),
            summary=article.get("summary", ""),
            source=article.get("source", ""),
            url=article.get("url", ""),
            published_at=article.get("time_published", ""),
            sentiment=sentiment_label,
        ))

    return NewsOutput(query=query, total=len(items), items=items)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 财务计算器工具（避免 LLM 数学幻觉）
# ═══════════════════════════════════════════════════════════════════════════════

class FinancialCalcInput(BaseModel):
    """财务指标计算输入"""
    metric: str = Field(
        ...,
        description=(
            "要计算的指标名称，支持：\n"
            "  pe_ratio        — 市盈率 = price / eps\n"
            "  pb_ratio        — 市净率 = price / bvps\n"
            "  roe             — 净资产收益率 = net_income / equity\n"
            "  gross_margin    — 毛利率 = (revenue - cogs) / revenue\n"
            "  net_margin      — 净利润率 = net_income / revenue\n"
            "  debt_ratio      — 资产负债率 = total_debt / total_assets\n"
            "  cagr            — 复合增长率，需提供 start_value/end_value/years\n"
            "  ev_ebitda       — EV/EBITDA = (market_cap + debt - cash) / ebitda"
        )
    )
    params: dict = Field(..., description="计算所需参数键值对，单位统一为元/亿元（保持一致即可）")


class FinancialCalcOutput(BaseModel):
    """财务指标计算输出"""
    metric: str = Field(..., description="指标名称")
    result: float = Field(..., description="计算结果")
    formula: str = Field(..., description="计算公式说明")
    unit: str = Field(..., description="结果单位")


def calculate_financial_metric(metric: str, params: dict) -> FinancialCalcOutput:
    """
    精确计算财务指标

    使用 Python 原生运算代替 LLM 数学计算，消除幻觉风险。

    Args:
        metric: 指标名称（见 FinancialCalcInput.metric 说明）
        params: 计算参数字典

    Returns:
        FinancialCalcOutput 计算结果
    """
    metric = metric.lower().strip()

    calculators = {
        "pe_ratio": _calc_pe,
        "pb_ratio": _calc_pb,
        "roe": _calc_roe,
        "gross_margin": _calc_gross_margin,
        "net_margin": _calc_net_margin,
        "debt_ratio": _calc_debt_ratio,
        "cagr": _calc_cagr,
        "ev_ebitda": _calc_ev_ebitda,
    }

    if metric not in calculators:
        raise ValueError(f"不支持的指标：{metric}，支持列表：{list(calculators.keys())}")

    return calculators[metric](params)


# ── 各指标计算函数 ─────────────────────────────────────────────────────────────

def _calc_pe(p: dict) -> FinancialCalcOutput:
    price, eps = float(p["price"]), float(p["eps"])
    if eps == 0:
        raise ValueError("EPS 为 0，无法计算市盈率")
    return FinancialCalcOutput(
        metric="pe_ratio",
        result=round(price / eps, 2),
        formula="PE = 股价 / 每股收益（EPS）",
        unit="倍",
    )


def _calc_pb(p: dict) -> FinancialCalcOutput:
    price, bvps = float(p["price"]), float(p["bvps"])
    if bvps == 0:
        raise ValueError("每股净资产为 0，无法计算市净率")
    return FinancialCalcOutput(
        metric="pb_ratio",
        result=round(price / bvps, 2),
        formula="PB = 股价 / 每股净资产（BVPS）",
        unit="倍",
    )


def _calc_roe(p: dict) -> FinancialCalcOutput:
    net_income, equity = float(p["net_income"]), float(p["equity"])
    if equity == 0:
        raise ValueError("净资产为 0，无法计算 ROE")
    return FinancialCalcOutput(
        metric="roe",
        result=round(net_income / equity * 100, 2),
        formula="ROE = 净利润 / 净资产 × 100%",
        unit="%",
    )


def _calc_gross_margin(p: dict) -> FinancialCalcOutput:
    revenue, cogs = float(p["revenue"]), float(p["cogs"])
    if revenue == 0:
        raise ValueError("营收为 0，无法计算毛利率")
    return FinancialCalcOutput(
        metric="gross_margin",
        result=round((revenue - cogs) / revenue * 100, 2),
        formula="毛利率 = (营收 - 营业成本) / 营收 × 100%",
        unit="%",
    )


def _calc_net_margin(p: dict) -> FinancialCalcOutput:
    net_income, revenue = float(p["net_income"]), float(p["revenue"])
    if revenue == 0:
        raise ValueError("营收为 0，无法计算净利润率")
    return FinancialCalcOutput(
        metric="net_margin",
        result=round(net_income / revenue * 100, 2),
        formula="净利润率 = 净利润 / 营收 × 100%",
        unit="%",
    )


def _calc_debt_ratio(p: dict) -> FinancialCalcOutput:
    total_debt, total_assets = float(p["total_debt"]), float(p["total_assets"])
    if total_assets == 0:
        raise ValueError("总资产为 0，无法计算资产负债率")
    return FinancialCalcOutput(
        metric="debt_ratio",
        result=round(total_debt / total_assets * 100, 2),
        formula="资产负债率 = 总负债 / 总资产 × 100%",
        unit="%",
    )


def _calc_cagr(p: dict) -> FinancialCalcOutput:
    start, end, years = float(p["start_value"]), float(p["end_value"]), float(p["years"])
    if start <= 0 or years <= 0:
        raise ValueError("起始值和年数必须大于 0")
    cagr = (math.pow(end / start, 1 / years) - 1) * 100
    return FinancialCalcOutput(
        metric="cagr",
        result=round(cagr, 2),
        formula="CAGR = (终值/始值)^(1/年数) - 1",
        unit="%",
    )


def _calc_ev_ebitda(p: dict) -> FinancialCalcOutput:
    market_cap = float(p["market_cap"])
    debt = float(p["debt"])
    cash = float(p["cash"])
    ebitda = float(p["ebitda"])
    if ebitda == 0:
        raise ValueError("EBITDA 为 0，无法计算 EV/EBITDA")
    ev = market_cap + debt - cash
    return FinancialCalcOutput(
        metric="ev_ebitda",
        result=round(ev / ebitda, 2),
        formula="EV/EBITDA = (市值 + 债务 - 现金) / EBITDA",
        unit="倍",
    )
