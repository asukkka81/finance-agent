# tests/test_sandbox/test_scanner.py
"""AST 安全扫描器测试."""

import pytest

from sandbox.config import SandboxConfig
from sandbox.scanner import CodeScanner, ScanResult


class TestCodeScanner:
    """测试 AST 安全扫描."""

    @pytest.fixture
    def scanner(self):
        return CodeScanner(SandboxConfig())

    # ----- 安全代码应通过 -----

    def test_safe_math_code(self, scanner):
        """安全的数学运算应通过."""
        result = scanner.scan("""
x = 1 + 2
y = x * 3
print(f"结果: {y}")
""")
        assert result.passed

    def test_safe_numpy_import(self, scanner):
        """导入 numpy 应通过 (在白名单中)."""
        result = scanner.scan("""
import numpy as np
data = np.array([1, 2, 3])
print(np.mean(data))
""")
        assert result.passed

    def test_safe_pandas_import(self, scanner):
        """导入 pandas 应通过."""
        result = scanner.scan("""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.describe())
""")
        assert result.passed

    def test_safe_matplotlib(self, scanner):
        """导入 matplotlib 应通过."""
        result = scanner.scan("""
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
""")
        assert result.passed

    def test_safe_scipy(self, scanner):
        """导入 scipy.stats 应通过."""
        result = scanner.scan("""
from scipy.stats import norm
print(norm.ppf(0.95))
""")
        assert result.passed

    # ----- 危险代码应被拦截 -----

    def test_blocks_os_import(self, scanner):
        """导入 os 应被拦截."""
        result = scanner.scan("import os\nos.system('ls')")
        assert not result.passed
        assert any(i.category == "dangerous_import" for i in result.issues)

    def test_blocks_sys_import(self, scanner):
        """导入 sys 应被拦截."""
        result = scanner.scan("import sys\nsys.exit(0)")
        assert not result.passed

    def test_blocks_subprocess(self, scanner):
        """导入 subprocess 应被拦截."""
        result = scanner.scan("import subprocess\nsubprocess.run(['ls'])")
        assert not result.passed

    def test_blocks_socket_import(self, scanner):
        """导入 socket 应被拦截 (网络隔离)."""
        result = scanner.scan("import socket\ns = socket.socket()")
        assert not result.passed
        assert any(i.category == "network" for i in result.issues)

    def test_blocks_requests_import(self, scanner):
        """导入 requests 应被拦截."""
        result = scanner.scan("import requests\nrequests.get('http://example.com')")
        assert not result.passed

    def test_blocks_eval(self, scanner):
        """eval() 应被拦截."""
        result = scanner.scan("eval('1+1')")
        assert not result.passed
        assert any("eval" in i.description for i in result.issues)

    def test_blocks_exec(self, scanner):
        """exec() 应被拦截."""
        result = scanner.scan("exec('print(1)')")
        assert not result.passed

    def test_blocks_open(self, scanner):
        """open() 应被拦截 (文件操作)."""
        result = scanner.scan("f = open('/etc/passwd', 'r')")
        assert not result.passed

    def test_blocks_dunder_import(self, scanner):
        """__import__() 应被拦截."""
        result = scanner.scan("m = __import__('os')")
        assert not result.passed

    def test_blocks_shutil(self, scanner):
        """导入 shutil 应被拦截."""
        result = scanner.scan("import shutil\nshutil.rmtree('/')")
        assert not result.passed

    def test_blocks_pickle(self, scanner):
        """导入 pickle 应被拦截 (代码注入风险)."""
        result = scanner.scan("import pickle\npickle.loads(b'...')")
        assert not result.passed

    def test_syntax_error(self, scanner):
        """语法错误应被捕获."""
        result = scanner.scan("this is not valid python {{{")
        assert not result.passed
        assert len(result.errors) > 0

    def test_is_safe_helper(self, scanner):
        """is_safe() 快捷方法."""
        assert scanner.is_safe("x = 1 + 2") is True
        assert scanner.is_safe("import os") is False

    def test_multiple_issues(self, scanner):
        """一条代码中的多个问题都应被报告."""
        result = scanner.scan("""
import os
import sys
eval('1+1')
open('/tmp/test')
""")
        assert not result.passed
        assert len(result.issues) >= 4

    def test_scan_result_to_dict(self, scanner):
        """ScanResult.to_dict() 应正确序列化."""
        result = scanner.scan("import os")
        d = result.to_dict()
        assert d["passed"] is False
        assert d["issue_count"] > 0
        assert "issues" in d
        assert d["issues"][0]["category"] == "dangerous_import"
