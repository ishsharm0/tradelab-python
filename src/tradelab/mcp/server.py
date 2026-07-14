"""Low-level MCP SDK server and strict text-result boundary."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .tools import McpDependencies, ToolDefinition, build_tools

_MAX_SAFE_INTEGER = (1 << 53) - 1


def _clean_text(value: object) -> str:
    return "".join(
        "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character for character in str(value)
    )


def _json_safe(value: object, seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool)):
        return _clean_text(value) if isinstance(value, str) else value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("integer exceeds the portable JSON safe range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result contains a non-finite number")
        return value
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("result contains a reference cycle")
        seen.add(identity)
        try:
            output: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("result object keys must be strings")
                output[_clean_text(key)] = _json_safe(item, seen)
            return output
        finally:
            seen.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in seen:
            raise ValueError("result contains a reference cycle")
        seen.add(identity)
        try:
            return [_json_safe(item, seen) for item in value]
        finally:
            seen.remove(identity)
    raise ValueError(f"result contains unsupported value {type(value).__name__}")


def _result(value: object) -> CallToolResult:
    safe = _json_safe(value)
    payload = json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False)
    return CallToolResult(content=[TextContent(type="text", text=payload)], isError=False)


def _error(error: object) -> CallToolResult:
    message = _clean_text(error)
    return CallToolResult(
        content=[TextContent(type="text", text=f"Error: {message}")],
        isError=True,
    )


async def invoke_tool(
    tools: Mapping[str, ToolDefinition], name: str, arguments: Mapping[str, Any] | None
) -> CallToolResult:
    """Invoke one handler and convert every outcome to strict MCP text content."""
    definition = tools.get(name)
    if definition is None:
        return _error(f'Unknown tool "{name}"')
    try:
        return _result(await definition.handler(arguments or {}))
    except Exception as error:
        return _error(error)


def _package_version() -> str:
    try:
        return version("tradelab-python")
    except PackageNotFoundError:
        return "0.0.0"


def create_server(
    *,
    tools: Mapping[str, ToolDefinition] | None = None,
    dependencies: McpDependencies | None = None,
) -> Server:
    """Create a configured low-level SDK server without starting I/O."""
    catalog = dict(tools) if tools is not None else build_tools(dependencies)
    server = Server("tradelab", version=_package_version())

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=definition.name,
                description=definition.description,
                inputSchema=definition.input_schema,
            )
            for definition in catalog.values()
        ]

    @server.call_tool(validate_input=True)  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        return await invoke_tool(catalog, name, arguments)

    return server


async def run_stdio_server(server: Server | None = None) -> None:
    """Serve MCP messages over the process standard streams."""
    application = server or create_server()
    async with stdio_server() as (read_stream, write_stream):
        await application.run(
            read_stream,
            write_stream,
            application.create_initialization_options(),
        )


def entrypoint() -> None:
    """Console-script entrypoint for ``tradelab-mcp``."""
    asyncio.run(run_stdio_server())


__all__ = ["create_server", "entrypoint", "invoke_tool", "run_stdio_server"]
