#!/usr/bin/env python3
import pexpect
from openai import OpenAI
from config import API_KEY
import json
import time
import random
import re
import math
from utils import *


################################################ system-prompt and user-prompt ################################################


# system prompt
SYSTEM_PROMPT = f"""
Reference:
Examples - Admin Routines:
{examples_content}

Cheat Sheet - occ Commands:
{cheatsheet_content}

File Editing (non-interactive):
{fileEditingRoutine}

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

   b) Stop signal when the issue is resolved or the task is completed, or if further commands would not be useful:
      {{"stop": true, "reason": "<short reason (max 120 chars)>"}}
2. Never produce explanations, only the JSON.
3. No interactive prompts. Never use interactive/full-screen tools.
4. Keep command output short - when accessing large log files or content-heavy files, always use filters like grep, tail, or head.
5. Prefer safe read-only checks before destructive actions.
"""

# user prompt
USER_PROMPT = (
    "Return ONLY JSON. "
    "Either {\"cmd\":\"<command>\"} OR {\"stop\": true, \"reason\":\"<short reason>\"}. "
    "Given the recent command output, propose the next single command. "
    "If verification shows the problem is resolved, or if further commands are unlikely to help, return the stop JSON."
)

################################################ LLM-Client ################################################
client = OpenAI(api_key=API_KEY)


################################################ API-callers ################################################
def summarize_transcript(text: str, model: str, target_tokens: int = 600) -> str:
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
        max_tokens=target_tokens           # cap to prevent runaway outputs
    )

    return chat.choices[0].message.content.strip()

def summarize_terminal(
    text: str,
    model: str,
    target_tokens: int = 600,
    context_window: int = 10_000
) -> str:    
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
            {"role": "user", "content": text[-context_window:]},
        ],
        temperature=0.0,                  # deterministic summaries
        max_tokens=target_tokens           # cap to prevent runaway outputs
    )

    return chat.choices[0].message.content.strip()

def summarize_final(text: str, model: str, target_tokens: int = 600) -> str:
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
                    "Summarize the following bash-terminal interaction for a final protocol. "
                    "Keep it concise, factual, and focused on the most important results. Use \n appropriately to ensure proper output formatting."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,                  # deterministic summaries
        max_tokens=target_tokens           # cap to prevent runaway outputs
    )

    return chat.choices[0].message.content.strip()


def ask_LLM(
    prev_output: str,
    issue: str,
    model: str,
    temperature: float = 1
) -> dict:
    '''
    Query the LLM with the current issue context and recent output.
    '''
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + f" Issue/Task: {issue}" + f" Recent output: {prev_output} "}
        ],
        response_format={"type": "json_object"},
        temperature=temperature
    )
    return json.loads(chat.choices[0].message.content)

def ask_LLM_for_verification(issue: str, model: str) -> dict:
    """
    Ask the LLM for ONE safe, read-only verification command.
    Returns a JSON dict like {"cmd": "<command>"}.
    """
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    'Return ONLY JSON: {"cmd":"<command>"}\n'
                    "Propose ONE safe, read-only verification command to check whether the issue persists or the task has been solved.\n"
                    f"Issue/Task: {issue}"
                ),
            },
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(chat.choices[0].message.content)




################################################ main ################################################
if __name__ == "__main__":

    # =====================Settings-START=====================
    ### LLM
    NUMBER_OF_INTERACTIONS = 3
    TEMPERATURE = 1 # this has to be exactly 1 for gpt-5 and gpt-5-mini

    ### Verification
    FREQUENCY_VERIFICATION = 3
  
    ### Length thresholds for different content
    MAX_LENGTH_TERMINAL_OUTPUT = 2000
    MAX_LENGTH_TRANSCRIPT = 6000
    TAIL_LENGTH_TERMINAL = 500
    TAIL_LENGTH_TRANSCRIPT = 1500
    TERMINAL_CONTEXT_WINDOW = 10_000

    ### time delay
    BASE_DELAY = 0.2      # minimum thinking time (s)
    PER_CHAR = 0.01       # extra seconds per character
    VARIABILITY = 0.2     # fraction of mean used as stddev
    MAX_DELAY = 5.0       # hard cap
    
    ### Issue or Task for the Agent
    ISSUE = "Nextcloud is returning an HTTP 500 (Internal Server Error)."

    ### model selection
    MODEL_ADMIN_AGENT = "gpt-5-mini"
    MODEL_TRANSCRIPT_SUMMARIZER = "gpt-4.1-mini"
    MODEL_TERMINAL_SUMMARIZER = "gpt-4.1-mini"
    # =====================Settings-END=====================
    
    # protocol commands used
    commands_list = []

    # connect server, login-shell as root and configure sentinel
    session = ShellSession()
    session.connect_root_setSentinel()
    
    # set environment variables to simplify terminal output
    session.run_cmd("export SYSTEMD_URLIFY=0; export SYSTEMD_PAGER=; export SYSTEMD_COLORS=0")

    # set environment variable to extract new logs
    session.run_cmd('POS_nextcloud=$(stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    session.run_cmd('POS_audit=$(stat -c %s /var/log/audit/audit.log)')

    # generate verification command
    verify_cmd = ask_LLM_for_verification(ISSUE, MODEL_ADMIN_AGENT)["cmd"]

    try:
        # Chain of Interactions
        output, raw_output = "", ""
        for i in range(NUMBER_OF_INTERACTIONS):
            if i % FREQUENCY_VERIFICATION == 0:
                cmd = verify_cmd
            else:
                decision = ask_LLM(output, issue=ISSUE, model=MODEL_ADMIN_AGENT, temperature=TEMPERATURE)
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

            # simulate thinking dependent on command length
            mean_delay = BASE_DELAY + PER_CHAR * len(cmd)
            stddev = mean_delay * VARIABILITY
            delay = max(BASE_DELAY, min(random.gauss(mean_delay, stddev), MAX_DELAY))
            print(f"# sleep {delay:.2f}s (cmd length {len(cmd)})")
            time.sleep(delay)

            # execute the command in the interactive terminal
            result = session.run_cmd(cmd)
            # update raw output
            raw_output += f"\n$ {cmd}\n{result}\n"
            # summarize terminal output if it is too long
            if len(result) > MAX_LENGTH_TERMINAL_OUTPUT:
                try:
                    summary = summarize_terminal(cmd + "\n" + result, model=MODEL_TERMINAL_SUMMARIZER, context_window=TERMINAL_CONTEXT_WINDOW)
                    tail = result[-TAIL_LENGTH_TERMINAL:]
                    result = (
                        "[TERMINAL SUMMARY]\n"
                        f"{summary}\n"
                        "[TERMINAL TAIL]\n"
                        f"{tail}"
                    )
                except Exception as e:
                    # Fallback: no summary, just clip the tail
                    print(f"# Terminal-Summarizer failed: {e}")
                    result = (
                        "[clipped]\n"
                        f"{result[-TAIL_LENGTH_TERMINAL:]}"
                    )
            output += f"\n$ {cmd}\n{result}\n"
            # summarize current transcript if it is too long
            if len(output) > MAX_LENGTH_TRANSCRIPT:
                summary = summarize_transcript(output, model=MODEL_TRANSCRIPT_SUMMARIZER)
                tail = output[-TAIL_LENGTH_TRANSCRIPT:]
                output = summary + "\n\n---[recent tail]---\n" + tail

            # continuously update output.txt
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(output)

    finally:
        # write commands
        with open("commands.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(commands_list))
        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(summarize_final(output + "\nList of commands: " + "\n".join(commands_list), MODEL_TRANSCRIPT_SUMMARIZER))
        # write output to file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        # write raw_output to file
        with open("raw_output.txt", "w", encoding="utf-8") as f:
            f.write(raw_output)
        # extract new logs and write to file
        logs = session.run_cmd('tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
        with open("LLM_nextcloud.log", "w", encoding="utf-8") as f:
            f.write(logs)
        logs = session.run_cmd('tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
        with open("LLM_audit.log", "w", encoding="utf-8") as f:
            f.write(logs)
        session.close()

