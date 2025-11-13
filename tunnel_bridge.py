import subprocess
import time
import atexit

SSH_HOST = "server"
SSH_PORT = 22
REMOTE_PORT = 6666
LOCAL_PORT = 6666
SOCKET_PATH = "/tmp/nvim"

def start_ssh_tunnel():
    """
    Start an SSH port-forward in the background:
      local 127.0.0.1:6666 → remote 127.0.0.1:6666
    """
    cmd = [
        "ssh",
        "-N",  # no command, just forwarding
        "-L", f"{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}",
        SSH_HOST,
    ]
    proc = subprocess.Popen(cmd)
    atexit.register(proc.terminate)
    time.sleep(1.0)  # give SSH a moment to establish
    return proc

def start_socat_bridge():
    """
    Bridge a local UNIX socket (/tmp/nvim) to TCP localhost:6666
    """
    cmd = [
        "socat",
        f"UNIX-LISTEN:{SOCKET_PATH},fork,unlink-early",
        f"TCP:127.0.0.1:{LOCAL_PORT}",
    ]
    proc = subprocess.Popen(cmd)
    atexit.register(proc.terminate)
    time.sleep(1.0)  # give socat time to bind the socket
    return proc

def setup_remote_nvim_bridge():
    ssh_proc = start_ssh_tunnel()
    socat_proc = start_socat_bridge()
    print(f"[ok] SSH tunnel + socat bridge running (socket: {SOCKET_PATH})")
    return ssh_proc, socat_proc
