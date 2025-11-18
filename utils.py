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

    # --- Class-level (static) members ---
    ANSI_RE = re.compile(
        r"(?:\x1B[@-Z\\-_]"                # ESC Fe
        r"|\x1B\[[0-?]*[ -/]*[@-~]"        # ESC [ ... CSI
        r"|\x1B\][^\x07]*\x07"             # OSC ... BEL
        r"|\x1B[P^_].*?\x1B\\)"            # DCS/PM/APC ... ST
    )

    @staticmethod
    def strip_tty(s: str) -> str:
        """Remove ANSI escape sequences and carriage returns."""
        s = ShellSession.ANSI_RE.sub("", s)
        s = s.replace("\r", "")
        return s

    # --- Instance methods ---
    def __init__(self, sentinel="<<<READY>>> "):
        self.sentinel = sentinel
        self.child = pexpect.spawn("/bin/bash", ["-i"], encoding="utf-8", timeout=30)

    def start_vim(self, filename: str) -> None:
        # send a command
        self.child.sendline(f"sudo vim {filename}")
        # wait for vim start
        time.sleep(2.0)

    def grep_vim_debug(self, pattern: str, radius: int = 3):
        # clear any pending hit-enter and leave modes
        #self.child.send("\x1b")   # ESC
        #self.child.send("\r")     # ENTER

        
        
        if not pattern.startswith(r'\v'):
            pattern = r'\v\c' + pattern
        pat_vim = pattern.replace("'", "''")



        pat_vim = r"\v\c" + r"\w*\s*option"

        cmd = (
            r":set nomore nonu norelativenumber | "
            r"let v:errmsg='' | "
            r"let g:__px_msg='' | "
            r"echo '<<<BEGIN>>>' | "
            r"try | "
            r"  redir => g:__px_msg | "
            r"  execute 'g/" + pat_vim + r"/ "
            r"     execute printf(''%d,%dnumber'', "
            r"       max([1,line(''.'')-" + str(radius) + r"]), "
            r"       min([line(''$''),line(''.'')+" + str(radius) + r"]))' | "
            r"  redir END | "
            #r"  echo 'STATUS:OK' | "
            r"catch /.*/ | "
            r"  silent! redir END | "           # ensure redir is closed even on error
            r"  echo 'STATUS:EXC' | "
            r"  echo 'EXC:' . v:exception | "
            r"  echo 'ERR:' . v:errmsg | "
            r"endtry | "
            #r"echo 'MSGS:' . g:__px_msg | "
            #r"echo 'PAT:' . '" + pat_vim + r"' | "
            r"echo '<<<END>>>'"
        )

        self.child.send(cmd); self.child.send("\r")

        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>", timeout=15)
        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>", timeout=15)
        raw = self.child.before
        self.child.send("\r")

        return ShellSession.strip_tty(raw).strip()


    def grep_vim(self, pattern: str, radius: int = 3):
        # Clear any pending prompt / mode
        self.child.send("\x1b"); self.child.send("\r")

        # Canonicalize Vim regex flags
        if not pattern.startswith(r'\v'):
            pattern = r'\v\c' + pattern

        # Escape for single-quoted Vim string + delimiter '/'
        pat_vim = pattern.replace("'", "''").replace('/', r'\/')

        cmd = (
            r":silent! set nomore nonu norelativenumber | "
            r"echo '<<<BEGIN>>>' | "
            r"try | "
            r"  redir => g:__px | "
            r"  execute 'g/" + pat_vim + r"/ "
            r"     let s = max([1, line(''.'')-" + str(radius) + r"]) | "
            r"     let e = min([line(''$''), line(''.'')+" + str(radius) + r"]) | "
            r"     for l in range(s, e) | "
            r"       echo getline(l) | "
            r"     endfor' | "
            r"  redir END | "
            r"  echo g:__px | "
            r"catch /.*/ | "
            r"  silent! redir END | "
            r"  echo v:exception | "
            r"finally | "
            r"  echo '<<<END>>>' | "
            r"endtry"
        )

        self.child.send(cmd); self.child.send("\r")

        # Expect one pair only
        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>",   timeout=15)
        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>",   timeout=15)

        raw = self.child.before

        # Clean up and return
        self.child.send("\r")
        return ShellSession.strip_tty(raw).strip()


    def overwrite_vim(self, updated_content):
        self.child.send("\x1b"); self.child.send("\r")
        self.child.send(":0,$d\r")   # delete all lines
        self.child.send("i")         # insert mode
        self.child.send(updated_content)
        self.child.send("\x1b")

    def print_file_vim(self):
        # Clear any pending prompt / mode
        self.child.send("\x1b"); self.child.send("\r")
        # Deactivate interactive features and print content of file
        self.child.send(":set nomore nonu norelativenumber\r")
        #self.child.send(":echo '<<<BEGIN>>>' | silent! 1,$print | echo '<<<END>>>'\r")
        self.child.send(
        ":echo '<<<BEGIN>>>' | "
            "try | 1,$print | "
            r"catch /^Vim\%((\a\+)\)\=:E749/ | "
            "endtry | "
            "echo '<<<END>>>'\r"
        )

        # now capture between the markers (two captures per marker are necessary)
        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>", timeout=15)
        self.child.expect("<<<BEGIN>>>", timeout=15)
        self.child.expect("<<<END>>>", timeout=15)
        raw = self.child.before

        # Clear any lingering hit-enter prompt safely
        self.child.send("\r")
        
        # clean output
        return ShellSession.strip_tty(raw).strip()

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
        '''
        Safely close the child process, ignoring any errors.
        '''
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
    '''
    Check whether a command is safe to run by ensuring it doesn't contain
    any known dangerous operations like file deletion or system shutdown.
    '''
    low = cmd.strip().lower()
    return not any(token in low for token in DANGEROUS)

def init_env_and_log_offsets(session):
    '''
    Initialize environment settings and record current log file sizes as offsets
    for tracking new entries in Nextcloud, audit, and syslog logs.
    '''
    # set environment variables to simplify terminal output
    session.run_cmd("export SYSTEMD_URLIFY=0; export SYSTEMD_PAGER=; export SYSTEMD_COLORS=0")

    # set environment variable to extract new logs
    session.run_cmd('POS_nextcloud=$(stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    session.run_cmd('POS_audit=$(stat -c %s /var/log/audit/audit.log)')
    session.run_cmd('POS_syslog=$(stat -c %s /var/log/syslog)')

def read_new_logs(session):
    '''
    Read new entries from Nextcloud, audit, and syslog files since the last saved positions
    and write them to corresponding files in the LOGS/ directory.
    '''
    logs = session.run_cmd('tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
    with open("LOGS/LLM_nextcloud.log", "w", encoding="utf-8") as f:
        f.write(logs)
    logs = session.run_cmd('tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
    with open("LOGS/LLM_audit.log", "w", encoding="utf-8") as f:
        f.write(logs)
    logs = session.run_cmd('tail -c +$((POS_syslog+1)) /var/log/syslog')
    with open("LOGS/LLM_syslog.log", "w", encoding="utf-8") as f:
        f.write(logs)

