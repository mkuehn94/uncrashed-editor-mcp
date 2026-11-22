"""Optional MCP adapter: every operation of api.OPERATIONS as an MCP tool (needs the `mcp` extra).

Tool names, parameters and descriptions come from the api functions; this module only converts results and
exceptions to the reply formats of the MCP tools.
"""

from __future__ import annotations

import functools
import inspect
import json

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image

from . import api
from .api import INSTRUCTIONS, VERBS  # noqa: F401  (both stay importable from this module)

mcp = MCPServer("uncrashed", instructions=INSTRUCTIONS)

_TEXT_ERRORS = ("agent_exec", "agent_batch")  # these report failures as 'error <message>', all others as JSON


def _dump(payload) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _adapt(op: api.Op):
    capture = op.name == "capture_view"

    def tool(*args, **kwargs):
        try:
            value = api.call(op.name, *args, **kwargs)
        except Exception as exc:
            if op.name in _TEXT_ERRORS:
                return f"error {exc}"
            error = _dump({"error": str(exc)})
            return [error] if capture else error
        if isinstance(value, api.Text):
            return value.text
        if capture:
            return [_dump(value), Image(path=value["image"])]
        return _dump(value)

    functools.update_wrapper(tool, op.fn)  # name, qualname and the RAW docstring (the MCP description)
    del tool.__wrapped__  # or inspect.signature() would follow it back to the api return annotation
    signature = inspect.signature(op.fn, eval_str=True)
    params = [p for p in signature.parameters.values() if not p.name.startswith("_")]
    returns = list if capture else str
    tool.__signature__ = signature.replace(parameters=params, return_annotation=returns)
    tool.__annotations__ = {**{p.name: p.annotation for p in params}, "return": returns}
    return tool


for _op in api.OPERATIONS.values():
    globals()[_op.name] = _tool = _adapt(_op)
    mcp.add_tool(_tool, structured_output=False if _op.name == "capture_view" else None)
del _op, _tool


def main() -> None:
    mcp.run(transport="stdio")
