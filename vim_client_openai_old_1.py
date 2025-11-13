# client_neovim.py
import asyncio, os, sys, json
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEBUG = True

class NeovimMCPClient:
    def __init__(self, socket_path: str = "/tmp/nvim", allow_shell: bool = "false"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.socket_path = socket_path
        self.allow_shell = allow_shell.lower()

    async def connect(self):
        env = {
            **os.environ,
            "NVIM_SOCKET_PATH": self.socket_path,
            "ALLOW_SHELL_COMMANDS": "true" if self.allow_shell == "true" else "false",
        }
        params = StdioServerParameters(
            command="npx",
            args=["-y", "mcp-neovim-server"],
            env=env
        )
        stdio = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        tools = await self.session.list_tools()
        if DEBUG:
            print("Connected. Tools:", [t.name for t in (tools.tools or [])])

    async def open_file(self, filename: str):
        # Uses tool: vim_file_open
        return await self.session.call_tool("vim_file_open", {"filename": filename})

    async def append_hello_world(self):
        """
        Append 'Hello World' at end using a safe Vim command.
        Equivalent to: :normal GoHello World
        """
        cmd = "normal GoHello World"
        return await self.session.call_tool("vim_command", {"command": cmd})

    async def save(self, filename: Optional[str] = None):
        # Uses tool: vim_buffer_save
        args = {"filename": filename} if filename else {}
        return await self.session.call_tool("vim_buffer_save", args)

    async def status(self):
        return await self.session.call_tool("vim_status", {})

    async def buffer_with_numbers(self, filename: Optional[str] = None):
        args = {"filename": filename} if filename else {}
        return await self.session.call_tool("vim_buffer", args)

    async def run_demo(self, filename: str):
        await self.open_file(filename)
        await self.append_hello_world()
        await self.save()

        stat = await self.status()
        buf  = await self.buffer_with_numbers(filename)

        def _content_to_text(content_list: List[Any]) -> str:
            parts: List[str] = []
            for item in content_list or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        #print("\n--- vim_status ---")
        #print(_content_to_text(stat.content))

        #print("\n--- vim_buffer (numbered) ---")
        #print(_content_to_text(buf.content))


    async def close(self):
        await self.exit_stack.aclose()

async def main():
    socket = os.getenv("NVIM_SOCKET_PATH", "/tmp/nvim")
    allow_shell = os.getenv("ALLOW_SHELL_COMMANDS", "false")
    target_file = sys.argv[1] if len(sys.argv) > 1 else "my_nvim_file.txt"

    client = NeovimMCPClient(socket, allow_shell)
    try:
        await client.connect()
        await client.run_demo(target_file)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
