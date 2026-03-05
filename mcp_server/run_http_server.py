"""
HTTP runner for the MATPOWER FastMCP server.

Starts the FastMCP HTTP transport so clients can call tools via JSON-RPC
at an endpoint like http://127.0.0.1:3929/mcp with methods such as
{"method": "tools/call", "params": {"name": "wls_from_path", ...}}.

Usage:
  python -m mcp_server.run_http_server --port 3929 --host 127.0.0.1 --path /mcp

Notes:
- On first request, the server will start a MATLAB engine process.
- Ensure MATLAB has MATPOWER on the path (or set MATPOWER_PATH env var).
- This process should be kept running while clients send requests.
"""

from __future__ import annotations

import asyncio
import argparse

from mcp_server.matpower_server import mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3929)
    ap.add_argument("--path", default="/mcp")
    ap.add_argument(
        "--transport",
        default="http",
        choices=["http", "streamable-http", "sse"],
        help="HTTP transport flavor (default http)",
    )
    args = ap.parse_args()

    # FastMCP handles uvicorn internally
    asyncio.run(
        mcp.run_http_async(
            host=args.host,
            port=args.port,
            path=args.path,
            transport=args.transport,
            json_response=True,
            stateless_http=True,
        )
    )


if __name__ == "__main__":
    main()
