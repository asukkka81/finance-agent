# sandbox/config.py
"""沙箱配置 — 安全参数 & 资源限制."""

from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """沙箱执行配置.

    安全措施:
        1. AST 静态扫描 — 检测危险调用 (import os, eval, open...)
        2. 内存限制 — RLIMIT_AS 限制虚拟内存 (MB)
        3. 时间限制 — SIGALRM 超时 (秒)
        4. 输出限制 — stdout/stderr 最大字节数
        5. 网络隔离 — 禁用 socket/urllib/requests 导入
    """

    # --- 资源限制 ---
    max_memory_mb: int = 512         # 最大内存 (MB)
    max_time_seconds: int = 30       # 最大执行时间 (秒)
    max_output_bytes: int = 100_000  # stdout 最大输出 (字节)
    max_code_length: int = 10_000    # 代码最大长度 (字符)

    # --- AST 扫描 ---
    enable_ast_scan: bool = True
    block_dangerous_imports: bool = True
    block_dangerous_builtins: bool = True
    block_file_operations: bool = True
    block_network: bool = True

    # --- 白名单 ---
    allowed_imports: list[str] = field(default_factory=lambda: [
        # 数据处理
        "numpy",
        "pandas",
        "math",
        "statistics",
        "decimal",
        "fractions",
        "random",
        "itertools",
        "collections",
        "functools",
        "operator",
        "datetime",
        "dateutil",
        "json",
        "csv",
        "re",
        "typing",
        "dataclasses",
        "enum",
        "copy",
        "hashlib",
        "base64",
        "textwrap",
        "string",
        "heapq",
        "bisect",
        "array",
        # 可视化
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        # 科学计算 (受限)
        "scipy.stats",
        "scipy.optimize",
        "numpy.linalg",
        "numpy.random",
    ])

    # --- 金融运行时预导入 ---
    preload_finance_runtime: bool = True

    # --- 危险模式 & 关键字 ---
    _dangerous_imports: set = field(default_factory=lambda: {
        "os", "sys", "subprocess", "shutil", "ctypes", "multiprocessing",
        "socket", "http", "urllib", "requests", "httpx", "aiohttp",
        "pathlib", "glob", "fnmatch", "tempfile", "io",
        "importlib", "pkgutil", "pkg_resources", "setuptools",
        "signal", "threading", "concurrent.futures", "asyncio",
        "code", "codeop", "compileall", "py_compile",
        "pty", "fcntl", "termios", "tty",
        "smtplib", "ftplib", "telnetlib", "poplib", "imaplib",
        "pickle", "shelve", "marshal",
        "webbrowser", "antigravity",
    })

    _dangerous_builtins: set = field(default_factory=lambda: {
        "eval", "exec", "compile", "__import__", "open",
        "input", "breakpoint", "memoryview",
        "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "hasattr",
        "isinstance", "issubclass", "type",
        "super",
    })
