# helpers.py
import re
import pexpect


# read relevant files
with open("InContextLearning/examples.txt", "r", encoding="utf-8") as file:
    examples_content = file.read()
with open("InContextLearning/cheatsheet.txt", "r", encoding="utf-8") as file:
    cheatsheet_content = file.read()
with open("InContextLearning/fileEditingRoutine.txt", "r", encoding="utf-8") as file:
    fileEditingRoutine = file.read()
    

class ShellSession:
    def __init__(self, sentinel="<<<READY>>> "):
        self.sentinel = sentinel
        self.child = pexpect.spawn("/bin/bash", ["-i"], encoding="utf-8", timeout=30)


    def connect_root_setSentinel(self) -> None:
        '''
        Establish an SSH connection to the server, escalate to a root shell via sudo -i,
        and set a unique sentinel prompt (PS1) for reliable command synchronization
        in the pexpect session.
        '''
        # connecting to server
        cmd = "ssh server"
        self.child.sendline(cmd)
        self.child.expect(r"[$#] ")
        self.child.expect(r"[$#] ")
        # open root-shell
        cmd = "sudo -i"
        self.child.sendline(cmd)
        self.child.expect(r"[$#] ")
        # setting environment variable
        cmd = f"PS1='{self.sentinel}'"
        self.child.sendline(cmd)
        self.child.expect(self.sentinel)
        self.child.expect(self.sentinel)
    

    def run_cmd(self, cmd: str) -> str:
        '''
        Execute a shell command via the active pexpect child process, 
        wait for the sentinel prompt, and return the cleaned output.
        '''
        self.child.sendline(cmd)
        try:
            self.child.expect(self.sentinel, timeout=25)
        except pexpect.TIMEOUT:
            print("Timeout was reached!")
            self.child.send('\x03')  # Ctrl-C
            self.child.expect(self.sentinel, timeout=5)
        raw = clean(self.child.before)
        # remove command from output
        parts = raw.splitlines()
        if parts and parts[0].strip() == cmd.strip():
            parts = parts[1:]
        return "\n".join(parts).strip()


    def close(self):
        try:
            self.child.close(force=True)
        except Exception:
            pass



def clean(text: str) -> str:
    '''
    Clean a string by removing unwanted terminal control sequences and
    non-printable characters.
    '''
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    text = re.sub(r'\x1D', '', text)
    return text


# Command Safety
DANGEROUS = (
    "rm -rf /", "mkfs", "dd if=", "shutdown", "reboot", "init 0", "halt",
    "pvcreate", "vgremove", "lvremove", "wipefs", "cryptsetup", "chattr +i",
)
def is_safe_command(cmd: str) -> bool:
    low = cmd.strip().lower()
    return not any(token in low for token in DANGEROUS)

def init_env_and_log_offsets(session):
    # set environment variables to simplify terminal output
    session.run_cmd("export SYSTEMD_URLIFY=0; export SYSTEMD_PAGER=; export SYSTEMD_COLORS=0")

    # set environment variable to extract new logs
    session.run_cmd('POS_nextcloud=$(stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    session.run_cmd('POS_audit=$(stat -c %s /var/log/audit/audit.log)')
    session.run_cmd('POS_syslog=$(stat -c %s /var/log/syslog)')

def read_new_logs(session):
    logs = session.run_cmd('tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
    with open("LOGS/LLM_nextcloud.log", "w", encoding="utf-8") as f:
        f.write(logs)
    logs = session.run_cmd('tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
    with open("LOGS/LLM_audit.log", "w", encoding="utf-8") as f:
        f.write(logs)
    logs = session.run_cmd('tail -c +$((POS_syslog+1)) /var/log/syslog')
    with open("LOGS/LLM_syslog.log", "w", encoding="utf-8") as f:
        f.write(logs)

