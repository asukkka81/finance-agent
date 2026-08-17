# sandbox/runtime.py
"""金融分析运行时库 — 沙箱预置函数集.

Agent 生成的分析代码可以直接调用这些函数，无需 import。

提供的函数:
    === 风险指标 ===
    sharpe_ratio(returns, risk_free=0.02)       — 夏普比率
    sortino_ratio(returns, risk_free=0.02)      — 索提诺比率
    max_drawdown(prices)                         — 最大回撤
    calmar_ratio(returns, max_dd)               — 卡尔玛比率
    information_ratio(returns, benchmark)        — 信息比率

    === 收益指标 ===
    annualized_return(returns)                   — 年化收益率
    cumulative_return(returns)                   — 累计收益率
    volatility(returns, annualize=True)          — 波动率
    cagr(start, end, years)                     — 复合年增长率

    === 回归 & 相关性 ===
    beta(stock, market)                          — Beta 系数
    alpha(stock, market, risk_free=0.02)        — Alpha
    correlation(x, y)                            — 相关系数
    r_squared(y_true, y_pred)                   — R² 决定系数

    === 技术指标 ===
    sma(prices, window)                          — 简单移动平均
    ema(prices, window)                          — 指数移动平均
    rsi(prices, period=14)                       — 相对强弱指数
    macd(prices)                                 — MACD 指标

    === 组合分析 ===
    portfolio_return(weights, returns)           — 组合预期收益
    portfolio_volatility(weights, cov_matrix)    — 组合波动率
    efficient_frontier(returns, points=50)       — 有效前沿
    value_at_risk(returns, confidence=0.95)      — VaR 风险价值
    conditional_var(returns, confidence=0.95)    — CVaR 条件风险价值

    === 可视化辅助 ===
    plot_prices(dates, prices, title)            — 价格走势图 (base64 PNG)
    plot_returns(returns, title)                 — 收益分布直方图
    plot_efficient_frontier(ef_data)             — 有效前沿图
"""

import math
import sys
from typing import Optional

import numpy as np

# ================================================================
# 风险指标
# ================================================================


def sharpe_ratio(returns, risk_free: float = 0.02) -> float:
    """夏普比率 — 衡量单位风险所带来的超额回报.

    Args:
        returns: 日/周/月收益率序列 (array-like).
        risk_free: 无风险利率 (年化, 默认 2%).

    Returns:
        年化夏普比率。越高表示风险调整后收益越好。
        通常 > 1.0 为良好，> 2.0 为优秀。
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free / 252  # 日化无风险利率
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def sortino_ratio(returns, risk_free: float = 0.02) -> float:
    """索提诺比率 — 只惩罚下行波动的夏普改进版.

    Args:
        returns: 收益率序列.
        risk_free: 无风险利率 (年化).

    Returns:
        年化索提诺比率。
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(252))


def max_drawdown(prices) -> float:
    """最大回撤 — 投资组合从峰顶到谷底的最大跌幅.

    Args:
        prices: 价格序列 (array-like).

    Returns:
        最大回撤 (小数, 如 -0.25 表示最大跌幅 25%)。
    """
    prices = np.asarray(prices, dtype=float)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return float(np.min(drawdown))


def calmar_ratio(returns, max_dd: float = None) -> float:
    """卡尔玛比率 — 年化收益 / |最大回撤|.

    Args:
        returns: 收益率序列.
        max_dd: 最大回撤 (可选, 不传则自动计算).

    Returns:
        卡尔玛比率。越高越好。
    """
    returns = np.asarray(returns, dtype=float)
    ann_return = annualized_return(returns)
    if max_dd is None:
        max_dd = abs(max_drawdown(np.cumprod(1 + returns)))
    if max_dd == 0:
        return 0.0
    return float(ann_return / abs(max_dd))


def information_ratio(returns, benchmark_returns) -> float:
    """信息比率 — 超额收益 / 跟踪误差.

    Args:
        returns: 组合收益率序列.
        benchmark_returns: 基准收益率序列.

    Returns:
        年化信息比率。
    """
    returns = np.asarray(returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    excess = returns - benchmark
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


# ================================================================
# 收益指标
# ================================================================


def annualized_return(returns) -> float:
    """年化收益率 — 基于日收益序列计算.

    Args:
        returns: 日收益率序列.

    Returns:
        年化收益率 (小数)。
    """
    returns = np.asarray(returns, dtype=float)
    return float(np.mean(returns) * 252)


def cumulative_return(returns) -> float:
    """累计收益率 — 从初始到最终的总体收益.

    Args:
        returns: 收益率序列.

    Returns:
        累计收益率 (小数)。
    """
    returns = np.asarray(returns, dtype=float)
    return float(np.prod(1 + returns) - 1)


def volatility(returns, annualize: bool = True) -> float:
    """波动率 (标准差).

    Args:
        returns: 收益率序列.
        annualize: 是否年化 (默认 True).

    Returns:
        波动率。
    """
    returns = np.asarray(returns, dtype=float)
    vol = float(np.std(returns))
    if annualize:
        vol *= np.sqrt(252)
    return vol


def cagr(start_value: float, end_value: float, years: float) -> float:
    """复合年增长率 (CAGR).

    Args:
        start_value: 初始价值.
        end_value: 最终价值.
        years: 年数.

    Returns:
        CAGR (小数)。
    """
    if start_value <= 0 or years <= 0:
        return 0.0
    return float((end_value / start_value) ** (1 / years) - 1)


# ================================================================
# 回归 & 相关性
# ================================================================


def beta(stock_returns, market_returns) -> float:
    """Beta 系数 — 衡量个股相对市场的系统风险.

    Args:
        stock_returns: 个股收益率序列.
        market_returns: 市场收益率序列.

    Returns:
        Beta 值:
            = 1: 与市场同步
            > 1: 比市场波动更大 (激进型)
            < 1: 比市场波动更小 (防御型)
    """
    stock = np.asarray(stock_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    cov = np.cov(stock, market)[0][1]
    var = np.var(market)
    if var == 0:
        return 0.0
    return float(cov / var)


def alpha(stock_returns, market_returns, risk_free: float = 0.02) -> float:
    """Alpha — 超额收益 (Jensen's Alpha).

    Args:
        stock_returns: 个股收益率序列.
        market_returns: 市场收益率序列.
        risk_free: 无风险利率.

    Returns:
        年化 Alpha。正值表示跑赢市场。
    """
    stock = np.asarray(stock_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    b = beta(stock, market)
    rf_daily = risk_free / 252
    return float((np.mean(stock) - rf_daily - b * (np.mean(market) - rf_daily)) * 252)


def correlation(x, y) -> float:
    """皮尔逊相关系数."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0][1])


def r_squared(y_true, y_pred) -> float:
    """R² 决定系数."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ================================================================
# 技术指标
# ================================================================


def sma(prices, window: int = 20):
    """简单移动平均 (SMA)."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < window:
        return np.full_like(prices, np.nan)
    result = np.full_like(prices, np.nan)
    for i in range(window - 1, len(prices)):
        result[i] = np.mean(prices[i - window + 1 : i + 1])
    return result


def ema(prices, window: int = 20):
    """指数移动平均 (EMA)."""
    prices = np.asarray(prices, dtype=float)
    alpha = 2 / (window + 1)
    result = np.zeros_like(prices)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    return result


def rsi(prices, period: int = 14):
    """相对强弱指数 (RSI).

    Returns:
        RSI 值 (0~100)。> 70 超买，< 30 超卖。
    """
    prices = np.asarray(prices, dtype=float)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    result = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return result

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(prices)):
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

        # Wilder's smoothing
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

    return result


def macd(prices, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 指标.

    Returns:
        (MACD线, 信号线, 柱状图) 三元组。
    """
    prices = np.asarray(prices, dtype=float)
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ================================================================
# 组合分析
# ================================================================


def portfolio_return(weights, returns) -> float:
    """投资组合预期年化收益.

    Args:
        weights: 各资产权重 [w1, w2, ...], sum=1.
        returns: 各资产年化收益率 [r1, r2, ...].

    Returns:
        组合预期年化收益。
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    return float(np.dot(weights, returns))


def portfolio_volatility(weights, cov_matrix) -> float:
    """投资组合年化波动率.

    Args:
        weights: 资产权重.
        cov_matrix: 协方差矩阵 (年化).

    Returns:
        组合年化波动率。
    """
    weights = np.asarray(weights, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return float(np.sqrt(var))


def efficient_frontier(returns, cov_matrix=None, points: int = 50):
    """计算有效前沿 (简化版: 随机权重模拟).

    Args:
        returns: 各资产预期年化收益率.
        cov_matrix: 协方差矩阵 (不传则用 returns 计算模拟).
        points: 前沿点数.

    Returns:
        {
            'returns': [年化收益列表],
            'volatilities': [年化波动率列表],
            'sharpe_ratios': [夏普比率列表],
            'optimal': {'weights': [...], 'return': ..., 'volatility': ...},
        }
    """
    returns = np.asarray(returns, dtype=float)
    n_assets = len(returns)

    if cov_matrix is None:
        # 模拟协方差矩阵
        np.random.seed(42)
        rand_returns = np.random.randn(252, n_assets) * 0.01 + returns / 252
        cov_matrix = np.cov(rand_returns.T) * 252

    front_returns = []
    front_vols = []
    front_sharpes = []
    front_weights = []

    for _ in range(points * 100):
        w = np.random.random(n_assets)
        w = w / np.sum(w)

        port_ret = portfolio_return(w, returns)
        port_vol = portfolio_volatility(w, cov_matrix)
        sharpe = (port_ret - 0.02) / port_vol if port_vol > 0 else 0

        front_returns.append(port_ret)
        front_vols.append(port_vol)
        front_sharpes.append(sharpe)
        front_weights.append(w)

    # 找最优 (最高夏普比率)
    best_idx = int(np.argmax(front_sharpes))

    return {
        "returns": [float(r) for r in front_returns],
        "volatilities": [float(v) for v in front_vols],
        "sharpe_ratios": [float(s) for s in front_sharpes],
        "optimal": {
            "weights": [float(w) for w in front_weights[best_idx]],
            "return": float(front_returns[best_idx]),
            "volatility": float(front_vols[best_idx]),
            "sharpe_ratio": float(front_sharpes[best_idx]),
        },
    }


def value_at_risk(returns, confidence: float = 0.95) -> float:
    """VaR (Value at Risk) — 在给定置信度下最大可能损失.

    Args:
        returns: 收益率序列.
        confidence: 置信度 (默认 95%).

    Returns:
        VaR 值 (正数表示损失)。如 0.03 表示 95% 置信度下日损失不超过 3%。
    """
    returns = np.asarray(returns, dtype=float)
    return float(-np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns, confidence: float = 0.95) -> float:
    """CVaR (Conditional VaR) — 超过 VaR 的平均损失.

    Args:
        returns: 收益率序列.
        confidence: 置信度.

    Returns:
        CVaR 值。
    """
    returns = np.asarray(returns, dtype=float)
    var = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) == 0:
        return var
    return float(-np.mean(tail_losses))


# ================================================================
# 可视化辅助 (生成 base64 PNG 供 Agent 展示)
# ================================================================


def _fig_to_base64(fig) -> str:
    """将 matplotlib figure 转为 base64 PNG."""
    import io
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_prices(dates, prices, title: str = "价格走势", label: str = "Price") -> str:
    """价格走势图 → base64 PNG 字符串.

    Args:
        dates: 日期列表.
        prices: 价格序列.
        title: 图表标题.
        label: 图例标签.

    Returns:
        base64 编码的 PNG 图片字符串。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互后端
        import matplotlib.pyplot as plt
    except ImportError:
        return "[错误] matplotlib 未安装"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, prices, linewidth=1.5, label=label, color="#2563eb")
    ax.fill_between(
        range(len(prices)), prices, min(prices) * 0.95,
        alpha=0.1, color="#2563eb",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期" if isinstance(dates[0], str) else "时间")
    ax.set_ylabel("价格")
    ax.legend()
    ax.grid(True, alpha=0.3)

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def plot_returns(returns, title: str = "收益分布") -> str:
    """收益分布直方图 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "[错误] matplotlib 未安装"

    returns = np.asarray(returns, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 收益曲线
    axes[0].plot(np.cumprod(1 + returns), linewidth=1.5, color="#2563eb")
    axes[0].set_title("累计收益曲线")
    axes[0].grid(True, alpha=0.3)

    # 分布直方图
    axes[1].hist(returns, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("日收益分布")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def plot_efficient_frontier(ef_data: dict) -> str:
    """有效前沿图 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "[错误] matplotlib 未安装"

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        ef_data["volatilities"],
        ef_data["returns"],
        c=ef_data["sharpe_ratios"],
        cmap="YlOrRd",
        alpha=0.6,
        s=10,
    )

    # 标记最优组合
    opt = ef_data["optimal"]
    ax.scatter(
        opt["volatility"], opt["return"],
        color="red", s=200, marker="*",
        label=f"最优组合 (Sharpe={opt['sharpe_ratio']:.2f})",
    )

    ax.set_xlabel("年化波动率")
    ax.set_ylabel("年化收益率")
    ax.set_title("有效前沿 & 最优组合", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label="夏普比率")

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


# ================================================================
# 导出函数列表 (供 Agent 展示可用功能)
# ================================================================


FINANCIAL_FUNCTIONS: dict[str, str] = {
    # 风险指标
    "sharpe_ratio": "sharpe_ratio(returns, risk_free=0.02) — 夏普比率，越高越好",
    "sortino_ratio": "sortino_ratio(returns, risk_free=0.02) — 索提诺比率，只惩罚下行波动",
    "max_drawdown": "max_drawdown(prices) — 最大回撤，峰顶→谷底的最大跌幅",
    "calmar_ratio": "calmar_ratio(returns) — 卡尔玛比率，年化收益/|最大回撤|",
    "information_ratio": "information_ratio(returns, benchmark) — 信息比率，超额收益/跟踪误差",

    # 收益指标
    "annualized_return": "annualized_return(returns) — 年化收益率",
    "cumulative_return": "cumulative_return(returns) — 累计收益率",
    "volatility": "volatility(returns, annualize=True) — 波动率",
    "cagr": "cagr(start, end, years) — 复合年增长率",

    # 回归
    "beta": "beta(stock, market) — Beta 系数，衡量系统风险",
    "alpha": "alpha(stock, market, risk_free=0.02) — Alpha 超额收益",
    "correlation": "correlation(x, y) — 皮尔逊相关系数",

    # 技术指标
    "sma": "sma(prices, window=20) — 简单移动平均",
    "ema": "ema(prices, window=20) — 指数移动平均",
    "rsi": "rsi(prices, period=14) — 相对强弱指数",
    "macd": "macd(prices) — MACD (MACD线, 信号线, 柱状图)",

    # 组合
    "portfolio_return": "portfolio_return(weights, returns) — 组合预期收益",
    "portfolio_volatility": "portfolio_volatility(weights, cov) — 组合波动率",
    "efficient_frontier": "efficient_frontier(returns, points=50) — 有效前沿",
    "value_at_risk": "value_at_risk(returns, confidence=0.95) — VaR 风险价值",
    "conditional_var": "conditional_var(returns, confidence=0.95) — CVaR 条件风险价值",

    # 可视化
    "plot_prices": "plot_prices(dates, prices, title) — 价格走势图 → base64 PNG",
    "plot_returns": "plot_returns(returns, title) — 收益分布图 → base64 PNG",
    "plot_efficient_frontier": "plot_efficient_frontier(ef_data) — 有效前沿图 → base64 PNG",
}
