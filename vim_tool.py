# vim_tools.py
# Extended Vim tool with a richer token language and stronger safety.
import re
import time
from typing import List, Tuple, Iterable, Optional, Callable, Any
from langchain_core.tools import tool

# ============================================================
# Keystroke translation: tokens -> exact bytes Vim expects
# ============================================================

# Special key tokens -> actual bytes
SPECIAL_MAP = {
    "<ESC>": "\x1b",    # Escape
    "<C-[>": "\x1b",    # Alternative ESC
    "<CR>": "\r",       # Enter / Carriage return
    "<ENTER>": "\r",
    "<BS>": "\x7f",     # Backspace (DEL on many terminals)
    "<DEL>": "\x1b[3~",
    "<TAB>": "\t",

    # A few common Ctrl-* keys (extend as needed)
    "<C-c>": "\x03",
    "<C-d>": "\x04",
    "<C-u>": "\x15",
    "<C-w>": "\x17",
    "<C-a>": "\x01",
    "<C-e>": "\x05",
    "<C-k>": "\x0b",
    "<C-y>": "\x19",
}

# Shortcuts that quit Vim (blocked)
FORBIDDEN_NORMAL = {"ZZ", "ZQ"}

# Ex-commands that we never allow the model to run (saving/quitting/shell)
FORBIDDEN_EX_PREFIXES = ("w", "wq", "q", "q!", "x")
FORBIDDEN_EX_EXACT = {""}  # nothing here yet; placeholder if you want to block exact cmds
FORBIDDEN_EX_SHELL_BANG = True  # ":!..." is forbidden

# Whitelist prefixes for safer ex-commands (optional, loosen/tighten as needed)
SAFE_EX_PREFIXES = (
    "set",      # :set number, :set nowrap, etc.
    "e", "edit",# :e file
    "bnext", "bprev", "bfirst", "blast",
    "%s", "s",  # substitutions (:%s/old/new/g or :1,$s/foo/bar/g)
    "g",        # :g/pat/cmd (use with care)
    "help", "reg", "map", "nmap", "vmap", "omap",
)
# We also allow numeric ranges like "1,10s/foo/bar/g" or "%s/foo/bar/g". This is done in _is_allowed_ex.


# ============================================================
# Parsing helpers
# ============================================================

FENCE_RE = re.compile(r"^```[\w-]*\s*$")  # ``` or ```lang
WAIT_RE = re.compile(r"^<WAIT\s+(\d+)ms>$")  # <WAIT 150ms>

TYPE_OPEN = "<TYPE>"
TYPE_CLOSE = "</TYPE>"

def _strip_code_fences(s: str) -> str:
    """
    Remove a single leading/trailing Markdown code fence if present.
    Handles ``` and ```lang fences.
    """
    s = s.strip()
    lines = s.splitlines()

    if not lines:
        return s

    # Leading fence?
    if FENCE_RE.match(lines[0].strip()):
        # Find the closing fence (first line that matches)
        for i in range(1, len(lines)):
            if FENCE_RE.match(lines[i].strip()):
                # Extract lines in between
                return "\n".join(lines[1:i]).strip()
        # No closing fence found -> drop just the first fence line
        return "\n".join(lines[1:]).strip()

    # No leading fence — return as-is
    return s

def _unescape_type_text(text: str) -> str:
    """
    Support lightweight escapes inside <TYPE>...</TYPE>:
      \\n -> newline
      \\t -> tab
      \\r -> carriage return
      \\\\ -> backslash
    """
    return (
        text
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
    )

def _strip_inline_comment(line: str) -> str:
    return line
    # Alternative
    # Remove any '#' that appears after some text, unless inside <TYPE>...</TYPE>
    if "<TYPE>" in line and "</TYPE>" in line:
        return line  # don’t strip comments from inside literal text
    return line.split("#", 1)[0].strip()

def _parse_keystrokes(raw: str) -> List[str]:
    """
    Parse the LLM's output into a list of tokens. The language is:

    - One token per line; empty lines are ignored.
    - Full-line comments starting with '#' are ignored.
    - Normal-mode tokens: i I a A o O dd yy p P x X u gg G 0 $ yw dw cw ci" ci' ci) ci] di" di' etc.
    - Special keys: <ESC> <CR> <ENTER> <TAB> <BS> <DEL> <C-c> <C-d> <C-u> <C-w> <C-a> <C-e> ...
    - Insert literal text: <TYPE>your text (\\n, \\t, \\r, \\\\ escapes allowed)</TYPE>
    - Delay token: <WAIT 150ms>  (inserts a sleep)
    - Safe ex-commands (no saving/quitting/shell): :%s/old/new/g, :1,$s/foo/bar/g, :set ..., :e file, etc.

    Returns the list of tokens exactly as lines (no translation here).
    """
    s = _strip_code_fences(raw or "")
    tokens: List[str] = []
    for ln in s.splitlines():
        line = ln.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        line = _strip_inline_comment(line)
        if line:
            tokens.append(line)
    return tokens


# ============================================================
# Validation & classification
# ============================================================

def _is_wait_token(t: str) -> Optional[int]:
    """
    If t is a <WAIT Nms> token, return N (milliseconds). Else None.
    """
    m = WAIT_RE.match(t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _is_forbidden_ex(cmd: str) -> bool:
    """
    Block writes/quits and optionally any :! shell escape.
    """
    if FORBIDDEN_EX_SHELL_BANG and cmd.startswith("!"):
        return True

    if cmd in FORBIDDEN_EX_EXACT:
        return True

    # Block common write/quit forms and variants with args (e.g., ":w file")
    for pfx in FORBIDDEN_EX_PREFIXES:
        if cmd == pfx or cmd.startswith(pfx + " "):
            return True

    return False

def _is_allowed_ex(cmd: str) -> bool:
    """
    Allow typical safe commands, plus range-based :s calls.
    This is intentionally permissive-but-safe; tighten if needed.
    """
    if _is_forbidden_ex(cmd):
        return False

    # Accept numbers/ranges like "1,10s/foo/bar/g", "%s/foo/bar/g", ".,$s/..", etc.
    if cmd[:1].isdigit() or cmd.startswith("%") or cmd.startswith(".,"):
        return True

    # Accept common safe prefixes
    return any(cmd.startswith(pfx) for pfx in SAFE_EX_PREFIXES)

def _ensure_nonempty_changes(tokens: List[str]) -> bool:
    """
    Basic sanity: ensure there's at least *some* action token.
    """
    return any(t for t in tokens)


# ============================================================
# Translation: tokens -> text to send (with optional delays)
# ============================================================

def _to_vim_inputs(tokens: List[str]) -> List[Tuple[str, float]]:
    """
    Convert tokens to (payload, delay_after) tuples.
    - For NORMAL tokens, returns (text, 0.0)
    - For EX commands (:...), returns (":cmd\\r", 0.0)
    - For <TYPE>...</TYPE>, returns (unescaped literal text, 0.0)
    - For <WAIT Nms>, returns ("", N/1000.0)
    - For specials (<ESC>, <CR>, etc.), returns (byte, 0.0)
    """
    out: List[Tuple[str, float]] = []

    for t in tokens:
        # 1) Waits: <WAIT 150ms>
        ms = _is_wait_token(t)
        if ms is not None:
            out.append(("", max(ms, 0) / 1000.0))
            continue

        # 2) TYPE blocks
        if t.startswith(TYPE_OPEN) and t.endswith(TYPE_CLOSE):
            text = t[len(TYPE_OPEN):-len(TYPE_CLOSE)]
            text = _unescape_type_text(text)
            out.append((text, 0.0))
            continue

        # 3) Special keys
        if t in SPECIAL_MAP:
            out.append((SPECIAL_MAP[t], 0.0))
            continue

        # 4) Ex-commands (":...")
        if t.startswith(":"):
            ex = t[1:].strip()
            if not ex:
                # empty ":" — ignore
                continue
            if _is_forbidden_ex(ex):
                # silently skip unsafe commands
                continue
            if not _is_allowed_ex(ex):
                # mildly strict: skip unknown ex-commands to be safe
                continue
            out.append((":" + ex + "\r", 0.0))
            continue

        # 5) Forbid quit shortcuts
        if t in FORBIDDEN_NORMAL:
            continue

        # 6) Otherwise, treat as a normal-mode keystroke token.
        out.append((t, 0.0))

    return out


def _coalesce_tokens(tokens: List[str]) -> List[str]:
    """
    Merge common multi-part sequences so they execute correctly:
      - <TYPE> ... </TYPE>         ->  single <TYPE>...joined-with-\n...</TYPE>
      - / or ? + chunks ... + <CR> ->  '/pattern', '<CR>'
      - f/t/r/F/T/R + {char}       ->  'f=', 'ta', 'rX', etc.
      - g + g                      ->  'gg'
      - operator+textobj           ->  'ci"', 'di)', 'ciw', 'cw', etc.
    """
    out: List[str] = []
    i = 0
    n = len(tokens)

    def _is_special(tok: str) -> bool:
        return tok in ("<ESC>", "<TAB>", "<BS>", "<DEL>", "<CR>", "<ENTER>") or tok.startswith(":")

    while i < n:
        t = tokens[i]

        # --- NEW: fix stray '/<CR>' or '?<CR>' emitted as a single token ---
        if t in ("/<CR>", "?<CR>"):
            out.append("<CR>")
            i += 1
            continue

        # --- NEW: split '/pattern<CR>' or '?pattern<CR>' into two tokens ---
        if (t.startswith("/") or t.startswith("?")) and t.endswith("<CR>") and len(t) > len("<CR>"):
            out.append(t[:-len("<CR>")])  # '/pattern'
            out.append("<CR>")
            i += 1
            continue

        # 0) Coalesce TYPE blocks possibly spread across multiple tokens
        if t.startswith(TYPE_OPEN):
            content_parts: List[str] = [t[len(TYPE_OPEN):]] if len(t) > len(TYPE_OPEN) else []
            j = i + 1
            closed = False
            while j < n:
                nxt = tokens[j]
                if nxt == TYPE_CLOSE:
                    closed = True
                    i = j + 1
                    break
                if nxt.endswith(TYPE_CLOSE) and len(nxt) > len(TYPE_CLOSE):
                    before = nxt[: -len(TYPE_CLOSE)]
                    content_parts.append(before)
                    closed = True
                    i = j + 1
                    break
                content_parts.append(nxt)
                j += 1
            if not closed:
                i = j
            merged = TYPE_OPEN + "".join(content_parts) + TYPE_CLOSE
            out.append(merged)
            continue

        # 1) Coalesce searches: /...<CR> or ?...<CR>
        if t in ("/", "?"):
            j = i + 1
            parts: List[str] = []
            while j < n:
                nxt = tokens[j]
                if nxt in ("<CR>", "<ENTER>"):
                    out.append(t + "".join(parts))
                    out.append(nxt)
                    i = j + 1
                    break
                if _is_special(nxt):
                    out.append(t + "".join(parts))
                    i = j
                    break
                parts.append(nxt)
                j += 1
            else:
                out.append(t + "".join(parts))
                i = n
            continue

        # 2) Coalesce already-merged search tokens like '/^enabled'
        if (t.startswith("/") or t.startswith("?")) and len(t) > 1:
            out.append(t)
            i += 1
            continue

        # 3) Coalesce f/t/r (and uppercase) + {char}
        if t in ("f", "t", "r", "F", "T", "R") and i + 1 < n and len(tokens[i + 1]) == 1:
            out.append(t + tokens[i + 1])
            i += 2
            continue

        # 4) Coalesce g + g  -> 'gg'
        if t == "g" and i + 1 < n and tokens[i + 1] == "g":
            out.append("gg")
            i += 2
            continue

        # 5) Coalesce operator + text object
        if t in ("c", "d", "y") and i + 1 < n:
            t2 = tokens[i + 1]
            if t2 in ("i", "a") and i + 2 < n and tokens[i + 2] in ('"', "'", ")", "]", "}", "w"):
                out.append(t + t2 + tokens[i + 2])
                i += 3
                continue
            if t2 == "w":
                out.append(t + t2)
                i += 2
                continue

        # default: pass through
        out.append(t)
        i += 1

    return out

# ============================================================
# Factory: bind tool to a specific ShellSession and LLM
# ============================================================

def make_use_vim(session: Any, planner: Any):
    """
    Dependency-injected tool factory. Pass your existing ShellSession and LLM.
    Returns a @tool callable you can register in LangChain/LangGraph.
    The resulting tool:
      - Opens the file in Vim
      - Reads current content
      - Asks the LLM to produce tokens in safe mini-language
      - Translates tokens into exact input bytes for Vim
      - Sends them (with optional waits)
      - Verifies the result by reading content again
      - Saves & quits via session.end_vim()
    """

    @tool
    def use_vim(filename: str, query: str) -> str:
        """
        Edit or modify a file using Vim-like keystrokes planned by an LLM.

        Args:
            filename (str): Full absolute path of the file to open or edit.
            query (str): Natural language description of the desired edits.

        Returns:
            str: Status message indicating success or failure of the edit.
        """
        print("------------------------- Entered use-vim -------------------------")
        print("vim-filename:", filename)
        print("vim-query:", query)

        # 1) Open file in Vim
        session.start_vim(filename)

        # Guard: ensure Normal mode at start
        try:
            session.edit_file_vim([SPECIAL_MAP["<ESC>"]])
        except Exception:
            # non-fatal
            pass

        # 2) Read current content
        try:
            before = session.print_file_vim()
        except Exception as e:
            try:
                session.end_vim()
            finally:
                return f"Failure to read file before editing: {e}"

        ### DEBUG
        #print("File content before editing:")
        #print(before)

        # 3) Ask the LLM for keystrokes in our expanded language (NO save/quit)
        prompt = rf"""
You are a Vim expert. Output EXACTLY one token per line. No prose, no fences.

TOKENS
- NORMAL: i I a A o O x X u dd yy p P gg G 0 $ yw dw cw ciw diw ci" ci' ci) ci] ci}}
- SPECIAL: <ESC> <CR> <TAB> <BS> <DEL> <C-c> <C-d> <C-u> <C-w> <C-a> <C-e>
- TYPE: <TYPE>text (\\n, \\t, \\r, \\\\ allowed)</TYPE>
- WAIT: <WAIT 150ms>
- EX: :s/..../..../  :%s/..../..../g  :g/.../ d  :set ...  :e {filename}

RULES
- Searches MUST be atomic: '/regex' or '?regex' on one line, then <CR> on next. (Do NOT split chars.)
- Do NOT use multi-line search patterns (\n); prefer line-range Ex commands instead.
- TYPE blocks MUST be a single token: put the entire multi-line inside one <TYPE>...</TYPE> using \n.
- Prefer Ex substitutions for key/values (e.g., :s/^\s*enabled\s*=.*/enabled = true/). Avoid f=, cf, ciw unless needed.
- NO :w, :wq, :q, :q!, :x, ZZ, ZQ. NO :! shell.

EXAMPLES
/^\[server\.tls\]/
<CR>
j
:s/^\s*enabled\s*=.*/enabled = true/

USER REQUEST
{query}

CURRENT FILE CONTENT
{before}
""".strip()

        try:
            result = planner.invoke(prompt)
            tokens = _parse_keystrokes(getattr(result, "content", ""))
            # print("raw tokens:", tokens)

            if not _ensure_nonempty_changes(tokens):
                try:
                    session.end_vim()
                finally:
                    return "Failure: no actionable tokens were produced."

            # 4) Translate tokens to (payload, delay) stream
            tokens = _coalesce_tokens(tokens)
            ops = _to_vim_inputs(tokens)
            # print("translated ops:", ops)

            # 5) Send to Vim (respect delays)
            try:
                for payload, delay_s in ops:
                    if payload:
                        session.edit_file_vim([payload])
                    if delay_s > 0:
                        time.sleep(delay_s)
                # Safety: ensure Normal mode before save/quit
                session.edit_file_vim([SPECIAL_MAP["<ESC>"]])
            except Exception as e:
                try:
                    session.end_vim()
                finally:
                    return f"Failure to send keystrokes: {e}"

            # 6) Read after content (optional but useful for logs)
            try:
                after = session.print_file_vim()
            except Exception as e:
                try:
                    session.end_vim()
                finally:
                    return f"Failure to read file after editing: {e}"

            #print("File content after editing:")
            #print(after)

            # 7) Save & quit via your session helper (centralized, reliable)
            session.end_vim()
            return "Vim edits applied."
        except Exception as e:
            # Last-resort escape so we don't trap the shell in Vim
            try:
                session._vim_escape_hatch(wait=3)
            except Exception:
                pass
            return f"Failure to edit file: {e}"

    return use_vim
