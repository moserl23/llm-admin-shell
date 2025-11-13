# remote_edit_over_ssh.py
from pynvim import attach
import time

# Replace 'server' with your SSH host (or user@host)
with attach('child', argv=['ssh', 'server', 'nvim', '--embed', 'my_nvim_file.txt']) as nvim:
    nvim.current.buffer.append("Hello World")
    nvim.command('write')  # saves on the server
    nvim.exec_lua('vim.schedule(function() vim.cmd("qa!") end)')
    time.sleep(0.1)        # tiny grace period for a clean shutdown
