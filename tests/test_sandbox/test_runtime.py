# tests/test_sandbox/test_runtime.py
"""金融运行时库测试."""

import numpy as np
import pytest


class TestRiskMetrics:
    """测试风险指标函数."""

    def test_sharpe_ratio_positive(self):
        """正收益序列的夏普比率应为正值."""
        from sandbox.runtime import sharpe_ratio

        returns = [0.01, 0.02, 0.015, 0.01, 0.005]
        sharpe = sharpe_ratio(returns)
        assert sharpe > 0, f"Expected positive sharpe, got {sharpe}"

    def test_sharpe_ratio_zero_returns(self):
        """零收益的夏普比率应为 0."""
        from sandbox.runtime import sharpe_ratio

        returns = [0.0, 0.0, 0.0, 0.0]
        sharpe = sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_max_drawdown(self):
        """最大回撤应正确计算."""
        from sandbox.runtime import max_drawdown

        prices = [100, 110, 90, 95, 105, 80, 100]
        mdd = max_drawdown(prices)
        # 从 110 到 80 → (80-110)/110 = -0.2727...
        assert mdd < 0
        assert abs(mdd - (-30/110)) < 0.01

    def test_max_drawdown_all_up(self):
        """持续上涨的最大回撤应接近 0."""
        from sandbox.runtime import max_drawdown

        prices = [100, 101, 102, 103, 104]
        mdd = max_drawdown(prices)
        assert mdd == 0.0

    def test_sortino_ratio(self):
        """索提诺比率计算 — 应 > 0 当有正超额收益."""
        from sandbox.runtime import sortino_ratio

        # 构造有足够下行波动的序列 (mix of gains and losses)
        np.random.seed(123)
        returns = list(np.random.randn(252) * 0.015 + 0.0008)  # ~20% annualized with volatility
        sortino = sortino_ratio(returns)
        assert sortino > 0, f"Expected positive sortino with positive drift, got {sortino}"

    def test_calmar_ratio(self):
        """卡尔玛比率计算."""
        from sandbox.runtime import calmar_ratio

        # 生成有明显波动的收益序列
        np.random.seed(42)
        returns = np.random.randn(252) * 0.02 + 0.0005  # 年化 ~12%, 有波动
        calmar = calmar_ratio(returns)
        # 卡尔玛比率应 > 0 (正收益)
        assert calmar > 0, f"Expected positive calmar, got {calmar}"


class TestReturnMetrics:
    """测试收益指标."""

    def test_annualized_return(self):
        """年化收益率计算."""
        from sandbox.runtime import annualized_return

        # 每天 0.1% → 年化 ≈ 0.001 * 252 = 0.252
        returns = [0.001] * 252
        ann = annualized_return(returns)
        assert abs(ann - 0.252) < 0.01

    def test_cumulative_return(self):
        """累计收益率计算."""
        from sandbox.runtime import cumulative_return

        # 每天涨 1%，100 天
        returns = [0.01] * 100
        cum = cumulative_return(returns)
        expected = (1.01 ** 100) - 1
        assert abs(cum - expected) < 0.01

    def test_cagr(self):
        """CAGR 计算."""
        from sandbox.runtime import cagr

        # 3 年从 100 涨到 133.1 → CAGR = 10%
        c = cagr(100, 133.1, 3)
        assert abs(c - 0.10) < 0.01

    def test_volatility(self):
        """波动率计算."""
        from sandbox.runtime import volatility

        returns = [0.01, -0.01, 0.01, -0.01] * 63  # ~252 天
        vol = volatility(returns, annualize=True)
        assert vol > 0  # 应该有波动


class TestRegressionMetrics:
    """测试回归指标."""

    def test_beta_one(self):
        """跟踪市场的股票 Beta 应≈1."""
        from sandbox.runtime import beta

        np.random.seed(42)
        market = np.random.randn(252) * 0.01
        stock = market + np.random.randn(252) * 0.002  # 高度相关

        b = beta(stock, market)
        assert 0.8 < b < 1.2

    def test_alpha(self):
        """Alpha 计算."""
        from sandbox.runtime import alpha

        np.random.seed(42)
        market = np.random.randn(252) * 0.01
        # 每天跑赢市场 0.05%
        stock = market + 0.0005 + np.random.randn(252) * 0.002

        a = alpha(stock, market)
        # 年化约 0.0005 * 252 = 0.126
        assert a > 0

    def test_correlation(self):
        """相关系数计算."""
        from sandbox.runtime import correlation

        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(correlation(x, y) - 1.0) < 0.001

        z = [5, 4, 3, 2, 1]
        assert abs(correlation(x, z) + 1.0) < 0.001


class TestTechnicalIndicators:
    """测试技术指标."""

    def test_sma(self):
        """简单移动平均."""
        from sandbox.runtime import sma

        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = sma(prices, window=3)
        # 前 2 个为 NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # 第 3 个: (1+2+3)/3 = 2
        assert abs(result[2] - 2.0) < 0.01
        # 最后一个: (8+9+10)/3 = 9
        assert abs(result[-1] - 9.0) < 0.01

    def test_ema(self):
        """指数移动平均."""
        from sandbox.runtime import ema

        prices = [10.0] * 20
        result = ema(prices, window=5)
        assert abs(result[-1] - 10.0) < 0.01

    def test_rsi_range(self):
        """RSI 值应在 0~100 之间."""
        from sandbox.runtime import rsi

        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        result = rsi(prices, period=14)

        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_macd(self):
        """MACD 指标."""
        from sandbox.runtime import macd

        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)
        macd_line, signal_line, histogram = macd(prices)

        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)


class TestPortfolioAnalysis:
    """测试组合分析."""

    def test_portfolio_return(self):
        """组合预期收益."""
        from sandbox.runtime import portfolio_return

        weights = [0.5, 0.3, 0.2]
        returns = [0.15, 0.20, 0.10]
        expected = 0.5 * 0.15 + 0.3 * 0.20 + 0.2 * 0.10
        assert abs(portfolio_return(weights, returns) - expected) < 0.001

    def test_portfolio_volatility(self):
        """组合波动率."""
        from sandbox.runtime import portfolio_volatility

        weights = [0.5, 0.5]
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        vol = portfolio_volatility(weights, cov)
        assert vol > 0

    def test_efficient_frontier(self):
        """有效前沿计算."""
        from sandbox.runtime import efficient_frontier

        returns = [0.15, 0.20, 0.10]
        ef = efficient_frontier(returns)

        assert "returns" in ef
        assert "volatilities" in ef
        assert "optimal" in ef
        assert ef["optimal"]["weights"]

    def test_value_at_risk(self):
        """VaR 计算."""
        from sandbox.runtime import value_at_risk

        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02 - 0.001  # 略有负偏
        var_95 = value_at_risk(returns, confidence=0.95)

        # VaR 应该是正数 (表示损失)
        assert var_95 > 0
        # 95% VaR 应大约在正态分布的 1.645σ 附近
        assert 0.01 < var_95 < 0.10

    def test_conditional_var(self):
        """CVaR 应 ≥ VaR."""
        from sandbox.runtime import value_at_risk, conditional_var

        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02 - 0.001
        var = value_at_risk(returns)
        cvar = conditional_var(returns)

        assert cvar >= var  # CVaR 总是 ≥ VaR
