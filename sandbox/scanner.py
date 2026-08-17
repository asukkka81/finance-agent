# sandbox/scanner.py
"""AST 静态代码扫描器 — 在执行前检测危险代码.

扫描维度:
    1. 危险导入 (os, sys, subprocess, socket, ...)
    2. 危险内置函数 (eval, exec, open, __import__, ...)
    3. 文件操作 (open, pathlib.Path, ...)
    4. 网络操作 (socket, urllib, http, requests, ...)
    5. 动态代码执行 (exec, eval, compile, ...)

使用 Python 标准库 ast 模块遍历语法树，不执行代码。
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScanIssue:
    """扫描发现的问题."""

    category: str          # dangerous_import / dangerous_builtin / file_op / network / dynamic_exec
    severity: str          # critical / high / medium / low
    line: int
    code_snippet: str      # 触发问题的代码片段
    description: str


@dataclass
class ScanResult:
    """扫描结果."""

    passed: bool = True
    issues: list[ScanIssue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def summary(self) -> str:
        if self.passed:
            return "代码安全扫描通过"
        return f"发现 {len(self.issues)} 个安全问题 (严重: {self.critical_count})"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity,
                    "line": i.line,
                    "code": i.code_snippet,
                    "description": i.description,
                }
                for i in self.issues
            ],
        }


class CodeScanner:
    """AST 代码安全扫描器.

    遍历 Python AST，检测所有不安全的操作模式。
    支持白名单机制 — allowed_imports 中的模块豁免检查。

    Usage::

        scanner = CodeScanner(config)
        result = scanner.scan('''
            import numpy as np
            returns = np.array([0.01, -0.02, 0.03])
            sharpe = np.mean(returns) / np.std(returns)
        ''')
        if result.passed:
            # 安全，可以执行
            ...
    """

    def __init__(self, config=None):
        from sandbox.config import SandboxConfig
        self.config = config or SandboxConfig()

    # ================================================================
    # 主扫描方法
    # ================================================================

    def scan(self, code: str) -> ScanResult:
        """扫描 Python 代码的安全性.

        Args:
            code: Python 源代码字符串.

        Returns:
            ScanResult — passed=True 表示安全，passed=False 表示有问题.
        """
        result = ScanResult()

        # 解析 AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.passed = False
            result.errors.append(f"语法错误: {e}")
            return result

        # 遍历 AST 节点，收集问题
        visitor = _SecurityVisitor(self.config, code)
        visitor.visit(tree)

        result.issues = visitor.issues
        result.passed = len(visitor.issues) == 0

        if not result.passed:
            logger.warning(
                "Code scan: %d issues found", len(result.issues)
            )
            for issue in result.issues:
                logger.debug(
                    "  [%s] L%d: %s", issue.severity, issue.line, issue.description
                )

        return result

    def is_safe(self, code: str) -> bool:
        """快速检查代码是否安全."""
        return self.scan(code).passed


# ================================================================
# AST Visitor
# ================================================================


class _SecurityVisitor(ast.NodeVisitor):
    """AST 安全遍历器.

    对所有敏感节点做安全检查。
    """

    def __init__(self, config, source_code: str):
        self.config = config
        self.source_lines = source_code.split("\n")
        self.issues: list[ScanIssue] = []

    def _add_issue(
        self, category: str, severity: str, node: ast.AST, description: str
    ):
        """记录一个安全问题."""
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = ""
        if 0 < line <= len(self.source_lines):
            snippet = self.source_lines[line - 1].strip()[:80]

        self.issues.append(ScanIssue(
            category=category,
            severity=severity,
            line=line,
            code_snippet=snippet,
            description=description,
        ))

    # ----- 导入检查 -----

    def visit_Import(self, node: ast.Import):
        """检查 import xxx 语句."""
        if self.config.block_dangerous_imports:
            for alias in node.names:
                self._check_import(alias.name, alias.asname, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """检查 from xxx import yyy 语句."""
        if self.config.block_dangerous_imports and node.module:
            self._check_import(node.module, None, node)
        self.generic_visit(node)

    def _check_import(
        self, module_name: str, alias: Optional[str], node: ast.AST
    ):
        """检查单个导入是否安全."""
        # 获取顶级模块名
        top_module = module_name.split(".")[0]

        # 白名单检查
        if module_name in self.config.allowed_imports:
            return
        # 检查前缀匹配 (必须跟 "." 或完全相同)
        for allowed in self.config.allowed_imports:
            if module_name == allowed or module_name.startswith(allowed + "."):
                return

        # 危险模块检查
        if top_module in self.config._dangerous_imports:
            self._add_issue(
                "dangerous_import", "critical", node,
                f"禁止导入危险模块: {module_name}",
            )

        # 网络检测
        if self.config.block_network:
            network_modules = {
                "socket", "http", "urllib", "requests", "httpx",
                "aiohttp", "websockets", "smtplib", "ftplib",
                "urllib3", "curl_cffi",
            }
            if (top_module in network_modules or
                any(module_name.startswith(p + ".") for p in network_modules)):
                self._add_issue(
                    "network", "critical", node,
                    f"禁止网络操作: {module_name}。沙箱环境无网络访问权限",
                )
                return  # 已经标记为危险，不重复添加

    # ----- 函数调用检查 -----

    def visit_Call(self, node: ast.Call):
        """检查函数调用."""
        if self.config.block_dangerous_builtins:
            self._check_dangerous_call(node)
        if self.config.block_file_operations:
            self._check_file_operation(node)
        self.generic_visit(node)

    def _check_dangerous_call(self, node: ast.Call):
        """检查是否调用了危险的内置函数."""
        func_name = self._get_func_name(node.func)
        if func_name and func_name in self.config._dangerous_builtins:
            severity = "critical" if func_name in ("eval", "exec", "__import__") else "high"
            self._add_issue(
                "dangerous_builtin", severity, node,
                f"禁止调用危险函数: {func_name}()",
            )

    def _check_file_operation(self, node: ast.Call):
        """检查文件操作."""
        func_name = self._get_func_name(node.func)

        # open() 调用
        if func_name == "open":
            self._add_issue(
                "file_op", "critical", node,
                "禁止文件操作: open()。沙箱中不允许读写文件系统",
            )

        # 文件相关方法调用
        file_methods = {".read", ".write", ".readlines", ".writelines"}
        if func_name and any(func_name.endswith(m) for m in file_methods):
            self._add_issue(
                "file_op", "high", node,
                f"疑似文件操作: {func_name}()",
            )

    # ----- 属性访问检查 -----

    def visit_Attribute(self, node: ast.Attribute):
        """检查属性访问 — 如 os.system, subprocess.call."""
        if self.config.block_dangerous_imports:
            obj_name = self._get_name(node.value)
            attr_name = node.attr

            if obj_name in ("os", "sys", "subprocess", "shutil"):
                self._add_issue(
                    "dangerous_import", "critical", node,
                    f"禁止访问系统模块属性: {obj_name}.{attr_name}",
                )

        self.generic_visit(node)

    # ================================================================
    # 辅助
    # ================================================================

    def _get_func_name(self, node) -> Optional[str]:
        """从 Call 节点提取函数名."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _get_name(self, node) -> Optional[str]:
        """从 AST 节点提取名称."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._get_name(node.value)
        return None
