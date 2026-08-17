# sandbox/sandbox.py
"""沙箱主类 — AST 扫描 → 安全执行 → 结果收集.

完整的沙箱执行流程:
    1. 代码长度检查
    2. AST 静态安全扫描
    3. 子进程隔离执行
    4. 结果收集 & 格式化
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from sandbox.config import SandboxConfig
from sandbox.executor import SafeExecutor
from sandbox.scanner import CodeScanner

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """沙箱执行完整结果.

    Attributes:
        success: 是否成功执行.
        output: stdout 输出.
        error: stderr 错误信息.
        scan_passed: AST 扫描是否通过.
        scan_issues: 扫描发现的问题列表.
        timed_out: 是否超时.
        elapsed_ms: 耗时 (毫秒).
        exit_code: 进程退出码.
    """

    success: bool = False
    output: str = ""
    error: str = ""
    scan_passed: bool = True
    scan_issues: list[dict] = field(default_factory=list)
    timed_out: bool = False
    elapsed_ms: float = 0.0
    exit_code: int = 0
    code_length: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "scan_passed": self.scan_passed,
            "scan_issues": self.scan_issues,
            "scan_issue_count": len(self.scan_issues),
            "timed_out": self.timed_out,
            "elapsed_ms": self.elapsed_ms,
            "exit_code": self.exit_code,
            "code_length": self.code_length,
        }

    @property
    def summary(self) -> str:
        if not self.success:
            if self.timed_out:
                return f"执行超时 ({self.elapsed_ms:.0f}ms)"
            if not self.scan_passed:
                return f"安全扫描未通过 ({len(self.scan_issues)} 个问题)"
            return f"执行失败 (exit={self.exit_code}): {self.error[:100]}"
        out_preview = self.output[:200].replace("\n", " ")
        return f"执行成功 ({self.elapsed_ms:.0f}ms): {out_preview}..."


class Sandbox:
    """Python 代码执行沙箱.

    三重安全机制:
        1. AST 静态扫描 — 执行前检测危险代码
        2. 子进程隔离 — 独立 Python 进程执行
        3. 资源限制 — 内存/时间/输出上限

    Usage::

        config = SandboxConfig(max_time_seconds=10, max_memory_mb=256)
        sandbox = Sandbox(config)

        result = sandbox.run('''
            returns = [0.01, -0.02, 0.03, 0.015, -0.005]
            sharpe = sharpe_ratio(returns)
            mdd = max_drawdown([100, 101, 99, 102, 105])
            print(f"夏普比率: {{sharpe:.4f}}")
            print(f"最大回撤: {{mdd:.4f}}")
        ''')
        print(result.output)
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.scanner = CodeScanner(self.config)
        self.executor = SafeExecutor(self.config)

    # ================================================================
    # 主执行方法
    # ================================================================

    def run(self, code: str) -> SandboxResult:
        """执行 Python 代码 (完整安全流程).

        Args:
            code: Python 源代码.

        Returns:
            SandboxResult — 包含输出、错误、扫描结果等。
        """
        t0 = time.time()

        # ---- Step 1: 代码长度检查 ----
        if len(code) > self.config.max_code_length:
            return SandboxResult(
                success=False,
                error=f"代码过长 ({len(code)} > {self.config.max_code_length} 字符)",
                code_length=len(code),
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # ---- Step 2: AST 扫描 ----
        if self.config.enable_ast_scan:
            scan_result = self.scanner.scan(code)
            if not scan_result.passed:
                return SandboxResult(
                    success=False,
                    error=f"代码安全扫描未通过: {scan_result.summary}",
                    scan_passed=False,
                    scan_issues=[i.__dict__ for i in scan_result.issues],
                    code_length=len(code),
                    elapsed_ms=(time.time() - t0) * 1000,
                )

        # ---- Step 3: 执行 ----
        exec_result = self.executor.execute(code)

        elapsed = (time.time() - t0) * 1000

        return SandboxResult(
            success=exec_result.success,
            output=exec_result.output,
            error=exec_result.error,
            scan_passed=True,
            timed_out=exec_result.timed_out,
            elapsed_ms=elapsed,
            exit_code=exec_result.exit_code,
            code_length=len(code),
        )

    def run_safe(self, code: str, default: str = "") -> str:
        """安全执行 — 失败时返回默认值 (不抛异常).

        Args:
            code: Python 代码.
            default: 失败时的默认返回值.

        Returns:
            执行输出 (stdout) 或 default。
        """
        result = self.run(code)
        if result.success:
            return result.output
        logger.warning("Sandbox run failed: %s", result.error)
        return default

    # ================================================================
    # 快速检查
    # ================================================================

    def is_code_safe(self, code: str) -> bool:
        """快速检查代码是否安全 (仅扫描，不执行)."""
        return self.scanner.is_safe(code)


# ================================================================
# MCP 工具集成
# ================================================================


def register_sandbox_tool(registry) -> None:
    """将沙箱注册为 MCP 工具.

    Args:
        registry: agent_layer ToolRegistry 实例.
    """
    from agent_layer.mcp.types import ToolRole

    _default_sandbox = Sandbox()

    @registry.register(
        name="execute_python",
        description=(
            "在安全的隔离沙箱中执行 Python 代码。"
            "支持金融计算 (夏普比率/最大回撤/Beta/Alpha/有效前沿/技术指标) "
            "和数据可视化 (价格走势图/收益分布图/有效前沿图)。"
            "沙箱已预导入 numpy、pandas 和金融分析函数。"
            "\n\n"
            "## 可用的金融函数:"
            "\n- sharpe_ratio(returns, risk_free=0.02) — 夏普比率"
            "\n- max_drawdown(prices) — 最大回撤"
            "\n- sortino_ratio(returns) — 索提诺比率"
            "\n- annualized_return(returns) — 年化收益率"
            "\n- volatility(returns) — 波动率"
            "\n- beta(stock, market) — Beta 系数"
            "\n- alpha(stock, market) — Alpha"
            "\n- sma(prices, window) / ema(prices, window) — 移动平均"
            "\n- rsi(prices, period=14) — 相对强弱指数"
            "\n- macd(prices) — MACD 指标"
            "\n- portfolio_return(weights, returns) — 组合收益"
            "\n- portfolio_volatility(weights, cov) — 组合波动率"
            "\n- efficient_frontier(returns) — 有效前沿"
            "\n- value_at_risk(returns) — VaR"
            "\n- plot_prices(dates, prices, title) — 价格走势图 (返回 base64 PNG)"
            "\n- plot_returns(returns, title) — 收益分布图"
            "\n- plot_efficient_frontier(data) — 有效前沿图"
            "\n\n"
            "## 安全限制:"
            "\n- 禁止导入 os/sys/subprocess/socket 等系统模块"
            "\n- 禁止 eval/exec/open 等危险函数"
            "\n- 禁止网络访问和文件操作"
            "\n- 最长执行时间 30 秒，内存上限 512MB"
            "\n\n"
            "## 示例:"
            "\n```python"
            "\nreturns = [0.01, -0.02, 0.03, 0.015, -0.005]"
            "\nsharpe = sharpe_ratio(returns)"
            "\nmdd = max_drawdown([100, 101, 99, 102, 105])"
            "\nprint(f'夏普比率: {sharpe:.4f}')"
            "\nprint(f'最大回撤: {mdd:.4%}')"
            "\n```"
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码。可以使用预导入的金融分析函数。",
                },
            },
            "required": ["code"],
        },
        role=ToolRole.SANDBOX,
    )
    def execute_python(code: str) -> dict:
        """MCP tool handler — 执行 Python 代码."""
        result = _default_sandbox.run(code)
        return result.to_dict()

    logger.info("Registered sandbox tool: execute_python")
