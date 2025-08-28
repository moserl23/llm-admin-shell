#!/usr/bin/env python3
import pexpect
from openai import OpenAI
from config import API_KEY
import json
import time
import random
import re


# read relevant files
with open("examples.txt", "r", encoding="utf-8") as file:
    examples_content = file.read()
with open("cheatsheet.txt", "r", encoding="utf-8") as file:
    cheatsheet_content = file.read()
with open("fileEditingRoutine.txt", "r", encoding="utf-8") as file:
    fileEditingRoutine = file.read()


# system prompt
SYSTEM_PROMPT = f"""
You are a Linux administrator for Ubuntu 24.04 running a LAMP stack with Nextcloud (PHP 8.3.6).

Connection context:
- You are connected via SSH.
- You are already in a root login shell (sudo -i).
- The following environment variables are always set:
    export SYSTEMD_URLIFY=0
    export SYSTEMD_PAGER=
    export SYSTEMD_COLORS=0    

Assumptions:
- Nextcloud is installed at /var/www/nextcloud
- occ commands are run via:
/usr/bin/php /var/www/nextcloud/occ
- Available tools: curl, net-tools, top (must be used non-interactively)

Logs are found here:
- ErrorLog /var/log/apache2/nextcloud.local-error.log
- CustomLog /var/log/apache2/nextcloud.local-access.log

Rules:
1. Output JSON in exactly one of these two forms:
   a) A single command to run:
      {{"cmd":"<command>"}}

   b) Stop signal when you believe the issue is resolved or further commands are not helpful:
      {{"stop": true, "reason": "<short reason (max 120 chars)>"}}
2. Never produce explanations, only the JSON.
3. No interactive prompts. Never use interactive/full-screen tools.
4. Keep command output short - when accessing large log files or content-heavy files, always use filters like grep, tail, or head.
5. Prefer safe read-only checks before destructive actions.

Reference:
Examples - Admin Routines:
{examples_content}

Cheat Sheet - occ Commands:
{cheatsheet_content}

File Editing (non-interactive):
{fileEditingRoutine}

"""

# user prompt
USER_PROMPT = (
    "Return ONLY JSON. "
    "Either {'cmd':'<command>'} OR {'stop': true, 'reason':'<short reason>'}. "
    "Given the recent command output, propose the next single command. "
    "If the problem appears fixed or more commands are unlikely to help, return the stop JSON."
)

# Start a bash-shell in interactive mode
child = pexpect.spawn("/bin/bash", ["-i"], encoding="utf-8", timeout=30)
# Initialize an LLM-Client
client = OpenAI(api_key=API_KEY)
# Define Sentinel for bash-shell
SENTINEL = "<<<READY>>> "



def connect_root_setSentinel():
    # connecting to server
    cmd = "ssh server"
    child.sendline(cmd)
    child.expect(r"[$#] ")
    child.expect(r"[$#] ")
    # open root-shell
    cmd = "sudo -i"
    child.sendline(cmd)
    child.expect(r"[$#] ")
    # setting environment variable
    cmd = f"PS1='{SENTINEL}'"
    child.sendline(cmd)
    child.expect(SENTINEL)
    child.expect(SENTINEL)



def clean(text):
    # Remove ANSI escape sequences
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    # Remove strange control character
    text = re.sub(r'\x1D', '', text)
    return text


def run_cmd(cmd):
    child.sendline(cmd)
    try:
        child.expect(SENTINEL, timeout=15)
    except pexpect.TIMEOUT:
        print("Timeout was reached!")
        child.send('\x03')  # Ctrl-C
        child.expect(SENTINEL, timeout=5)
    raw = clean(child.before)
    # remove command from output
    parts = raw.splitlines()
    if parts and parts[0].strip() == cmd.strip():
        parts = parts[1:]
    return "\n".join(parts).strip()


def summarize_transcript(text: str, model: str, target_chars: int = 1800) -> str:
    """
    Summarize a transcript of a system administrator interacting with a bash shell
    into a compact plain-text form.
    """
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Summarize the following transcript of a system administrator "
                    "interacting with a bash shell. "
                    "Keep it concise, factual, and focused on the most important actions/results."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,                  # deterministic summaries
        max_tokens=min(600, target_chars) # cap to prevent runaway outputs
    )

    return chat.choices[0].message.content.strip()


def summarize_terminal(text: str, model: str, target_chars: int = 1800) -> str:
    """
    Summarize a bash-terminal output.
    """
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Summarize the following bash-terminal output. "
                    "Keep it concise, factual, and focused on the most important results."
                ),
            },
            {"role": "user", "content": text[-10000:]},
        ],
        temperature=0.0,                  # deterministic summaries
        max_tokens=min(600, target_chars) # cap to prevent runaway outputs
    )

    return chat.choices[0].message.content.strip()

def summarize_final(text: str, model: str, target_chars: int = 1800) -> str:
    """
    Summarize a bash-terminal output.
    """
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Summarize the following bash-terminal interaction. "
                    "Keep it concise, factual, and focused on the most important results. Use \n appropriately to ensure proper output formatting."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,                  # deterministic summaries
        max_tokens=min(600, target_chars) # cap to prevent runaway outputs
    )

    return "\n".join(chat.choices[0].message.content.strip().split("."))




def ask_LLM(prev_output, issue, model, temperatur=0.1):
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + f" Issue: {issue}" + f" Recent output: {prev_output[:6000]} "}
        ],
        response_format={"type": "json_object"},
        temperature=temperatur, # float [0.0–2.0]
                                # Controls randomness in token choice.
                                # Lower = more deterministic/repetitive (good for commands).
                                # Higher = more creative/diverse outputs.
                                # Example: 0.0 → always same command; 1.0 → varied suggestions.

        # top_p=1.0,            # float [0.0–1.0]  (a.k.a nucleus sampling)
                                # Alternative to temperature.
                                # Model considers only the top-p fraction of probability mass.
                                # Example: 0.9 → use smallest set of tokens with 90% total probability.
                                # Usually tune EITHER top_p OR temperature, not both.

        # frequency_penalty=0,  # float [-2.0–2.0]
                                # Penalizes repeated tokens in output.
                                # Higher values = less repetition (useful for text, less for commands).
                                # Example: if model repeats 'tail -f', increase this slightly.

        # presence_penalty=0,   # float [-2.0–2.0]
                                # Encourages introducing new tokens not seen yet.
                                # Higher values = more variety; lower/negative = more repetition.
                                # Useful in brainstorming, but usually keep 0 for deterministic tasks.

        # max_tokens=64         # int [1–model limit]
                                # Maximum number of tokens in the response (output only).
                                # Safeguard against overly long responses.
                                # Example: 64 tokens ≈ one short command with some args.
    )
    return json.loads(chat.choices[0].message.content)



def append_result(transcript, cmd, result, max_len, model):
    if len(result) > max_len:
        try:
            # keep the *end* too, often more useful for logs
            summary = summarize_terminal(cmd + "\n" + result, model)
            tail = result[-500:]
            result = f"[summary]\n{summary}\n---[recent tail]---\n{tail}"
        except Exception:
            result = "[clipped]\n" + result[-1000:]
    entry = f"\n$ {cmd}\n{result}\n"
    return transcript + entry


if __name__ == "__main__":

    # Settings
    NUMBER_OF_INTERACTIONS = 40
    TEMPERATUR = 1
    MIN_SLEEP = 0.1
    MAX_SLEEP = 0.2   
    MAX_LENGTH = 2000
    MAX_TRANSCRIPT = 6000
    
    ISSUE = "Nextcloud is returning an HTTP 500 (Internal Server Error)."

    model_admin_agent = "gpt-5"
    model_transcript_summarizer = "gpt-4.1-mini"
    model_terminal_summarizer = "gpt-4.1-mini"
    
    commands_list = []

    # connect server, login-shell as root and  configure sentinel
    connect_root_setSentinel()
    
    # set environment variables to simplify terminal output
    run_cmd("export SYSTEMD_URLIFY=0; export SYSTEMD_PAGER=; export SYSTEMD_COLORS=0")

    # set environment variable to extract new logs
    run_cmd('POS_nextcloud=$(stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    run_cmd('POS_audit=$(stat -c %s /var/log/audit/audit.log)')



    try:
        # Chain of Interactions
        output = ""
        for _ in range(NUMBER_OF_INTERACTIONS):
            decision = ask_LLM(output, issue=ISSUE, model=model_admin_agent, temperatur=TEMPERATUR)

            # Stop condition from the LLM
            if decision.get("stop"):
                msg = decision.get("reason", "LLM indicated resolution.")
                print(f"# Stopping: {msg}")
                break

            # Otherwise, run one command
            cmd = decision.get("cmd")
            if not cmd:
                print("# No 'cmd' provided; stopping to avoid undefined behavior.")
                break

            # log the command
            commands_list.append(cmd)
            # execute the command in the interactive terminal
            result = run_cmd(cmd)
            # clip terminal output if it is too big
            output = append_result(output, cmd, result, MAX_LENGTH, model_terminal_summarizer)
            # summarize current transcript if it is too long
            if len(output) > MAX_TRANSCRIPT:
                summary = summarize_transcript(output, model=model_transcript_summarizer)
                tail = output[-1500:]
                output = summary + "\n\n---[recent tail]---\n" + tail

            # continuously update output.txt
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(output)

            # simulate thinking
            delay = random.uniform(MIN_SLEEP, MAX_SLEEP)
            print(f"# sleep {delay:.2f}s")
            time.sleep(delay)
    finally:
        # write commands
        with open("commands.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(commands_list))
        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(summarize_final(output + "\nList of commands: " + "\n".join(commands_list), model_transcript_summarizer))
        # write output to file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        # extract new logs and write to file
        logs = run_cmd('tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
        with open("logs_nextcloud.txt", "w", encoding="utf-8") as f:
            f.write(logs)
        logs = run_cmd('tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
        with open("logs_audit.txt", "w", encoding="utf-8") as f:
            f.write(logs)
        try: child.close(force=True)
        except Exception: pass


