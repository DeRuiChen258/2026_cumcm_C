"""兼容用的旧入口，真正的绘图程序在 code/report/visualize.py。

留着它是为了让以前的命令还能用：

    python3 -m code.report.figures      # 等价于 visualize --set main
"""
from __future__ import annotations

from code.report.visualize import main as _main


def main() -> None:
    _main(["--set", "main"])


if __name__ == "__main__":
    main()
