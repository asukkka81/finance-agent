# tests/test_sandbox/test_sandbox.py
"""Sandbox 端到端测试."""

import pytest

from sandbox.config import SandboxConfig
from sandbox.sandbox import Sandbox, SandboxResult


class TestSandbox:
    """测试沙箱端到端执行."""

    @pytest.fixture
    def sandbox(self):
        config = SandboxConfig(
            max_time_seconds=10,
            max_memory_mb=256,
            max_output_bytes=50000,
        )
        return Sandbox(config)

    # ----- 安全代码执行 -----

    def test_simple_execution(self, sandbox):
        """简单代码应成功执行."""
        result = sandbox.run("x = 1 + 2\nprint(f'result={x}')")
        assert result.success
        assert "result=3" in result.output
        assert result.scan_passed
        assert not result.timed_out

    def test_numpy_calculation(self, sandbox):
        """NumPy 计算应正常工作."""
        result = sandbox.run("""
import numpy as np
data = np.array([1, 2, 3, 4, 5])
print(f"mean={np.mean(data):.2f}")
print(f"std={np.std(data):.2f}")
""")
        assert result.success
        assert "mean=3.00" in result.output
        assert "std=1.41" in result.output

    def test_sharpe_ratio_integration(self, sandbox):
        """沙箱中使用预导入的 sharpe_ratio()."""
        result = sandbox.run("""
returns = [0.01, 0.02, -0.005, 0.015, 0.01]
sharpe = sharpe_ratio(returns)
print(f"sharpe={sharpe:.2f}")
""")
        assert result.success, f"Failed: {result.error}"
        assert "sharpe=" in result.output

    def test_max_drawdown_integration(self, sandbox):
        """沙箱中使用预导入的 max_drawdown()."""
        result = sandbox.run("""
prices = [100, 110, 90, 95, 105, 80, 100]
mdd = max_drawdown(prices)
print(f"mdd={mdd:.4f}")
""")
        assert result.success, f"Failed: {result.error}"
        assert "mdd=" in result.output

    def test_portfolio_analysis(self, sandbox):
        """完整的投资组合分析."""
        result = sandbox.run("""
weights = [0.4, 0.35, 0.25]
returns = [0.15, 0.20, 0.10]
port_ret = portfolio_return(weights, returns)
print(f"组合预期年化收益: {port_ret:.2%}")

cov = [
    [0.04, 0.01, 0.005],
    [0.01, 0.06, 0.015],
    [0.005, 0.015, 0.05],
]
port_vol = portfolio_volatility(weights, cov)
print(f"组合年化波动率: {port_vol:.2%}")
""")
        assert result.success, f"Failed: {result.error}"
        assert "组合预期年化收益" in result.output
        assert "组合年化波动率" in result.output

    def test_rsi_calculation(self, sandbox):
        """RSI 技术指标计算."""
        result = sandbox.run("""
import numpy as np
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 2)
rsi_values = rsi(prices, period=14)
valid = rsi_values[~np.isnan(rsi_values)]
print(f"RSI range: {valid.min():.1f} - {valid.max():.1f}")
print(f"RSI latest: {valid[-1]:.1f}")
""")
        assert result.success, f"Failed: {result.error}"
        assert "RSI range" in result.output

    # ----- 安全拦截 -----

    def test_blocks_os_in_execution(self, sandbox):
        """导入 os 的代码应被拦截 (AST 扫描阶段)."""
        result = sandbox.run("import os\nprint(os.listdir('.'))")
        assert not result.success
        assert not result.scan_passed
        assert len(result.scan_issues) > 0

    def test_blocks_eval_in_execution(self, sandbox):
        """使用 eval 的代码应被拦截."""
        result = sandbox.run("x = eval('1+1')\nprint(x)")
        assert not result.success
        assert not result.scan_passed

    def test_blocks_file_write(self, sandbox):
        """尝试写文件的代码应被拦截."""
        result = sandbox.run("f = open('/tmp/test.txt', 'w')\nf.write('hacked')\nf.close()")
        assert not result.success
        assert not result.scan_passed

    # ----- 错误处理 -----

    def test_syntax_error_handling(self, sandbox):
        """语法错误应优雅处理."""
        result = sandbox.run("this is invalid !!!")
        assert not result.success
        assert not result.scan_passed

    def test_runtime_error_handling(self, sandbox):
        """运行时错误应被捕获."""
        result = sandbox.run("x = 1 / 0\nprint(x)")
        assert not result.success

    def test_code_too_long(self, sandbox):
        """过长代码应被拒绝."""
        long_code = "x = 1\n" * 6000  # 超过 max_code_length
        result = sandbox.run(long_code)
        assert not result.success
        assert "过长" in result.error

    # ----- Result 方法 -----

    def test_result_to_dict(self, sandbox):
        """SandboxResult.to_dict() 应正确序列化."""
        result = sandbox.run("print('hello')")
        d = result.to_dict()
        assert d["success"] is True
        assert "output" in d
        assert "elapsed_ms" in d
        assert d["code_length"] > 0

    def test_result_summary(self, sandbox):
        """SandboxResult.summary 应有意义."""
        result = sandbox.run("print('test output')")
        assert "成功" in result.summary or "exec" in result.summary.lower()

    def test_run_safe_returns_default_on_failure(self, sandbox):
        """run_safe() 失败时应返回默认值."""
        output = sandbox.run_safe("import os", default="FALLBACK")
        assert output == "FALLBACK"

    def test_is_code_safe(self, sandbox):
        """is_code_safe() 快捷方法."""
        assert sandbox.is_code_safe("x = 1 + 2") is True
        assert sandbox.is_code_safe("import os") is False

    # ----- 性能 -----

    def test_execution_is_fast(self, sandbox):
        """简单代码应在合理时间内完成 (含子进程启动 + numpy 导入)."""
        result = sandbox.run("x = sum(range(1000))\nprint(x)")
        assert result.elapsed_ms < 3000  # 子进程启动 + numpy 导入

    def test_multiple_executions(self, sandbox):
        """多次执行应稳定."""
        for _ in range(5):
            result = sandbox.run("print('hello')")
            assert result.success
