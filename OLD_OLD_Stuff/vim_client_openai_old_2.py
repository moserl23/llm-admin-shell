# client_neovim.py (local MCP server, unix socket only)
import asyncio
import os
import sys
import subprocess
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEBUG = True

def _unix_socket_exists(path: str) -> bool:
    try:
        return subprocess.run(["bash", "-lc", f'[[ -S "{path}" ]]'], check=False).returncode == 0
    except Exception:
        return False

class NeovimMCPClient:
    def __init__(self, socket_path: str = "/tmp/nvim", allow_shell: str = "false"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.socket_path = socket_path
        self.allow_shell = allow_shell.lower()

    async def connect(self):
        # Preflight: ensure local UNIX socket exists (e.g. created by socat bridge)
        if not _unix_socket_exists(self.socket_path):
            raise RuntimeError(
                f"Expected UNIX socket at {self.socket_path}.\n"
                f"Start your tunnel + bridge first (example):\n"
                f"  ssh -f -N -L 6666:127.0.0.1:6666 server\n"
                f"  socat UNIX-LISTEN:{self.socket_path},fork,unlink-early TCP:127.0.0.1:6666"
            )

        env = {
            **os.environ,
            "NVIM_SOCKET_PATH": self.socket_path,
            "ALLOW_SHELL_COMMANDS": "true" if self.allow_shell == "true" else "false",
        }
        params = StdioServerParameters(
            command="npx",
            args=["-y", "mcp-neovim-server"],
            env=env,
        )

        stdio = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        if DEBUG:
            tools = await self.session.list_tools()
            print("Connected. Tools:", [t.name for t in (tools.tools or [])])

    async def open_file(self, filename: str):
        return await self.session.call_tool("vim_file_open", {"filename": filename})

    async def append_hello_world(self):
        return await self.session.call_tool(
            "vim_command", {"command": "normal GoHello World"}
        )

    async def save(self, filename: Optional[str] = None):
        args = {"filename": filename} if filename else {}
        return await self.session.call_tool("vim_buffer_save", args)

    async def run_demo(self, filename: str):
        await self.open_file(filename)
        await self.append_hello_world()
        await self.save()

    async def close(self):
        await self.exit_stack.aclose()

async def main():
    socket = os.getenv("NVIM_SOCKET_PATH", "/tmp/nvim")  # local UNIX socket (from socat)
    allow_shell = os.getenv("ALLOW_SHELL_COMMANDS", "false")
    target_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/my_nvim_file.txt"

    client = NeovimMCPClient(socket_path=socket, allow_shell=allow_shell)
    try:
        await client.connect()
        await client.run_demo(target_file)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())



