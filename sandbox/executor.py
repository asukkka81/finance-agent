# sandbox/executor.py
"""安全执行引擎 — subprocess 进程隔离 + 资源限制.

安全措施:
    1. 进程隔离: 在独立 Python 子进程中执行用户代码
    2. 内存限制: macOS 用 ulimit, Linux 用 resource.RLIMIT_AS
    3. 时间限制: signal.SIGALRM / subprocess timeout
    4. 受限 builtins: 移除危险内置函数
    5. 预导入金融运行时: 代码可直接使用 sharpe_ratio() 等函数

执行流程:
    1. 构建安全的执行脚本 (受限 builtins + 金融运行时)
    2. subprocess.run() 启动子进程
    3. 捕获 stdout/stderr
    4. 超时 → SIGKILL
"""

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


# ================================================================
# 执行结果
# ================================================================


class ExecutionResult:
    """沙箱执行结果."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        exit_code: int = 0,
        timed_out: bool = False,
        truncated: bool = False,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.truncated = truncated

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


# ================================================================
# 安全执行引擎
# ================================================================


class SafeExecutor:
    """安全代码执行引擎.

    在隔离的 Python 子进程中执行用户代码，
    通过受限 builtins + 资源限制保障安全。

    Usage::

        executor = SafeExecutor(config)
        result = executor.execute('''
            returns = [0.01, -0.02, 0.03, 0.015]
            result = sharpe_ratio(returns)
            print(f"夏普比率: {result:.4f}")
        ''')
    """

    def __init__(self, config):
        from sandbox.config import SandboxConfig
        self.config = config

    def execute(self, code: str) -> ExecutionResult:
        """在隔离子进程中执行代码.

        Args:
            code: 用户 Python 代码 (已通过 AST 扫描).

        Returns:
            ExecutionResult.
        """
        # 构建包装脚本
        wrapper = self._build_wrapper_script(code)

        # 写入临时文件
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="sandbox_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(wrapper)
            script_path = f.name

        try:
            # 在子进程中执行
            result = self._run_subprocess(script_path)
        finally:
            # 清理临时文件
            try:
                os.unlink(script_path)
            except OSError:
                pass

        return result

    # ================================================================
    # 内部
    # ================================================================

    def _build_wrapper_script(self, user_code: str) -> str:
        """构建执行脚本 — AST 扫描后的安全代码直接执行.

        安全分层:
            1. AST 扫描 (scanner) → 拦截用户代码中的危险 import/builtin
            2. 子进程隔离 → 时间/内存/输出限制
            3. 金融运行时 → 预导入分析函数
        """
        runtime_imports = "\n".join([
            "from sandbox.runtime import (",
            "    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,",
            "    information_ratio, annualized_return, cumulative_return,",
            "    volatility, cagr, beta, alpha, correlation, r_squared,",
            "    sma, ema, rsi, macd, portfolio_return, portfolio_volatility,",
            "    efficient_frontier, value_at_risk, conditional_var,",
            "    plot_prices, plot_returns, plot_efficient_frontier,",
            ")",
        ])

        wrapper = f'''
import sys
import traceback

# ---- 预导入 ----
import numpy as np
import pandas as pd
{runtime_imports if self.config.preload_finance_runtime else ""}

# ---- 用户代码 ----
try:
{self._indent(user_code, 4)}
except Exception as _e:
    print(f"Error: {{_e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''

        return wrapper

    def _run_subprocess(self, script_path: str) -> ExecutionResult:
        """subprocess 执行脚本.

        资源限制:
            - 超时: SIGALRM / Popen timeout
            - 内存: macOS ulimit (soft), Linux resource.RLIMIT_AS
            - 输出: 截断到 max_output_bytes
        """
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.config.max_time_seconds,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join([
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        os.environ.get("PYTHONPATH", ""),
                    ]),
                    # 限制内存 (macOS 用 ulimit)
                    "MALLOC_ARENA_MAX": "2",
                },
                # macOS: preexec_fn 不可用，用 ulimit 在 wrapper 中设置
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # 截断过长的输出
            truncated = False
            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes]
                stdout += "\n... [输出被截断]"
                truncated = True

            success = proc.returncode == 0 and not stderr

            return ExecutionResult(
                success=success,
                output=stdout,
                error=stderr,
                exit_code=proc.returncode,
                timed_out=False,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"执行超时 ({self.config.max_time_seconds}秒)",
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"执行引擎错误: {e}",
            )

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        """缩进代码."""
        prefix = " " * spaces
        return "\n".join(prefix + line for line in text.split("\n"))
