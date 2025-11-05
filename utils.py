# helpers.py
import re
import pexpect
import time
import random


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

    def start_vim(self, filename):
        # send a command
        self.child.sendline(f"sudo vim {filename}")
        # wait for vim start
        time.sleep(2.0)

    def print_file_vim(self):
        # deactivate interactive features and print content of file
        self.child.send(":set nomore nonu norelativenumber\r")
        self.child.send(":echo '<<<BEGIN>>>' | 1,$print | echo '<<<END>>>'\r")

        # now capture between the markers (two captures per marker are necessary)
        self.child.expect("<<<BEGIN>>>", timeout=5)
        self.child.expect("<<<END>>>", timeout=5)
        self.child.expect("<<<BEGIN>>>", timeout=5)
        self.child.expect("<<<END>>>", timeout=5)
        raw = self.child.before

        ANSI_RE = re.compile(
            r"(?:\x1B[@-Z\\-_]"                # ESC Fe
            r"|\x1B\[[0-?]*[ -/]*[@-~]"        # ESC [ ... CSI
            r"|\x1B\][^\x07]*\x07"             # OSC ... BEL
            r"|\x1B[P^_].*?\x1B\\)"            # DCS/PM/APC ... ST
        )

        def strip_tty(s: str) -> str:
            s = ANSI_RE.sub("", s)
            s = s.replace("\r", "")
            return s
        
        # clean output
        text = strip_tty(raw).strip()
        return text

    def edit_file_vim(self, keystrokes):
        
        for seq in keystrokes:
            self.child.send(seq)
            time.sleep(random.uniform(0.05, 0.25))  # human-like delays

    def end_vim(self):
        # make sure the edit is written and vim quits
        self.child.send(":wq\r")

        # now wait for your normal shell sentinel
        try:
            self.child.expect(self.sentinel, timeout=30)
        except pexpect.TIMEOUT:
            print("timeout --> try vim_escape_hatch")
            self._vim_escape_hatch()

    def _vim_escape_hatch(self, wait=5):
        """
        Minimal recovery if we're still inside Vim:
        ESC to leave any mode → clear hit-enter → force quit.
        """
        # leave insert/visual/cmdline
        self.child.send("\x1b")      # ESC
        # clear any "Press ENTER" / -- More --
        self.child.send("\r")        # ENTER
        # try a force quit-all
        self.child.send(":qa!\r")
        try:
            self.child.expect(self.sentinel, timeout=wait)
            return
        except pexpect.TIMEOUT:
            pass

        # plan B: single force quit
        self.child.send(":q!\r")
        try:
            self.child.expect(self.sentinel, timeout=wait)
            return
        except pexpect.TIMEOUT:
            pass

        # last resort: Ctrl-C then force quit again
        self.child.send("\x03")      # Ctrl-C
        self.child.send("\r")        # clear any prompt
        self.child.send(":qa!\r")
        self.child.expect(self.sentinel, timeout=wait)

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

