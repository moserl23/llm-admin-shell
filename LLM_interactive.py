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

# Model
MODEL = "gpt-5"

# Issue Text
ISSUE = "Nextcloud web UI returns 503"

# system prompt
SYSTEM_PROMPT = f"""
You are a Linux administrator for Ubuntu 24.04 running a LAMP stack with Nextcloud (PHP 8.3.6). 
The sudo password is: 'changeme'. Always run sudo non-interactively, e.g.:
    echo changeme | sudo -S <command>

Assumptions:
- Nextcloud is installed at /var/www/nextcloud
- occ commands are run via:
/usr/bin/php /var/www/nextcloud/occ
- Available tools: curl, net-tools, top (must be used non-interactively)

Logs are found here:
- ErrorLog /var/log/apache2/nextclouds.local-error.log
- CustomLog /var/log/apache2/nextcloud.local-access.log

Rules:
1. Output EXACTLY one command, as JSON only:
   {{"cmd":"<command>"}}
2. Never produce explanations, only the JSON.
3. No interactive prompts. Never use interactive/full-screen tools.
4. Keep command output short - when accessing large log files or content-heavy files, always use filters like grep, tail, or head.

Reference:
Examples - Admin Routines:
{examples_content}

Cheat Sheet - occ Commands:
{cheatsheet_content}

File Editing (non-interactive):
{fileEditingRoutine}

"""

# user prompt
USER_PROMPT = f"""Issue: {ISSUE}.
Return ONLY JSON: {{'cmd':'<command>'}}. NO EXPLAINATIONS.
Given the recent command output, propose the next single command.
"""

# Start a bash-shell in interactive mode
child = pexpect.spawn("/bin/bash", ["-i"], encoding="utf-8", timeout=30)
# Initialize an LLM-Client
client = OpenAI(api_key=API_KEY)
# Define Sentinel for bash-shell
SENTINEL = "<<<READY>>> "


def connect_setSentinel():
    # connecting to server
    cmd = "ssh server"
    child.sendline(cmd)
    child.expect(r"[$#] ")
    child.expect(r"[$#] ")
    # setting environment variable
    cmd = f"PS1='{SENTINEL}'"
    child.sendline(cmd)
    child.expect(SENTINEL)
    child.expect(SENTINEL)

def clean(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def run_cmd(cmd):
    child.sendline(cmd)
    child.expect(SENTINEL)
    output = child.before.strip()
    return clean(output)


def ask_LLM(prev_output):
    chat = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + f" Recent output: {prev_output[:6000]} "}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(chat.choices[0].message.content)["cmd"].strip()


def clip_middle(text, max_chars, marker = "\n---[output clipped]---\n"):
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(marker)
    if keep <= 0:  # degenerate case
        return text[:max_chars]
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


if __name__ == "__main__":


    # Settings
    NUMBER_OF_INTERACTIONS = 10
    MIN_SLEEP = 0.8
    MAX_SLEEP = 3.0   
    MAX_LENGTH = 2000

    # connect server and configure sentinel
    connect_setSentinel()

    # set environment variable to extract new logs
    run_cmd('POS_nextcloud=$(echo "changeme" | sudo -S stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    run_cmd('POS_audit=$(echo "changeme" | sudo -S stat -c %s /var/log/audit/audit.log )')

    # Chain of Interactions
    output = ""
    for _ in range(NUMBER_OF_INTERACTIONS):
        # execution
        output += clip_middle(run_cmd(ask_LLM(output)), MAX_LENGTH)

        # simulate thinking
        delay = random.uniform(MIN_SLEEP, MAX_SLEEP)
        print(f"# sleep {delay:.2f}s")
        time.sleep(delay)


    # write output to file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(output)

    # extract new logs and write to file
    logs = run_cmd('echo "changeme" | sudo -S tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
    with open("logs_nextcloud.txt", "w", encoding="utf-8") as f:
        f.write(logs)
    logs = run_cmd('echo "changeme" | sudo -S tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
    with open("logs_audit.txt", "w", encoding="utf-8") as f:
        f.write(logs)




