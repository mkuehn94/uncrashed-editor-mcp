"""python -m uncrashed_mcp: the CLI with arguments; without arguments the MCP stdio server when stdin is piped
(existing MCP client registrations), CLI help when it is a terminal. serve() is the `uncrashed-mcp` script."""

from __future__ import annotations

import sys


def serve() -> int:
    """Run the MCP stdio server; one line instead of a traceback when the optional `mcp` package is missing."""
    try:
        from .server import main as run
    except ModuleNotFoundError as exc:
        if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
            raise
        print("error: the MCP server needs the 'mcp' package: pip install 'uncrashed-mcp[mcp]'", file=sys.stderr)
        return 1
    run()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    stdin = sys.stdin  # None when the launcher closed fd 0
    if argv or (stdin is not None and stdin.isatty()):
        from .cli import main as cli_main

        if argv:
            return cli_main(argv)
        code = cli_main(["help"])
        print("run `uncrashed mcp` for the MCP server")
        return code
    if stdin is None:
        print("error: stdin is closed: the MCP server needs it (the command line is `uncrashed help`)", file=sys.stderr)
        return 1
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
