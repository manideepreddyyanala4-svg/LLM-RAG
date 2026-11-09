"""Small real MCP client for discovery and a safe search call."""
import asyncio
import os
import sys
from mcp import Client, StdioServerParameters


async def main():
    """Start the local server through stdio and print tools and search results."""
    token = os.environ.get("EVIDENCEOPS_TOKEN", "")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not token:
        print("Set EVIDENCEOPS_TOKEN first")
        return
    environment = os.environ.copy()
    environment.pop("EVIDENCEOPS_APPROVAL_TOKEN", None)
    if environment.get("EVIDENCEOPS_MCP_ROLE") != "operator":
        environment.pop("EVIDENCEOPS_OPERATOR_TOKEN", None)
    settings = StdioServerParameters(command=sys.executable, args=["mcp_server.py"], env=environment)
    try:
        async with Client(settings) as client:
            tools = await asyncio.wait_for(client.list_tools(), timeout=10)
            print("Available tools:", [tool.name for tool in tools.tools])
            result = await asyncio.wait_for(client.call_tool("search_approved_documents",
                                            {"question": "calibration records"},
                                            read_timeout_seconds=10), timeout=12)
            if result.is_error:
                print("MCP search failed:", result.content)
            else:
                print("Search result:", result.structured_content or result.content)
    except TimeoutError:
        print("MCP server did not respond before the timeout")
    except OSError as error:
        print("Cannot start the local MCP server:", error)


if __name__ == "__main__":
    asyncio.run(main())
