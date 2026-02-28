from __future__ import annotations

import argparse

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="memcp")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start MemCP MCP server")

    args = parser.parse_args()

    if args.command == "run":
        run()


if __name__ == "__main__":
    main()
