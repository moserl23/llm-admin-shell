#!/usr/bin/env python3
from config import API_KEY
from utils import *
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from typing import Callable
from functools import partial, wraps
import time, random
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from enum import Enum


def summarize_terminal(
    text: str,
    model: str = "gpt-4.1",
    target_tokens: int = 600,
    max_terminal_output: int = 4000,     # threshold to decide when to summarize
    context_window: int = 10_000,
    tail_chars: int = 400                # always keep this much raw tail
) -> str:
    if len(text) <= max_terminal_output:
        return text

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=target_tokens, api_key=API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following bash-terminal output. "
                   "Keep it concise, factual, and highlight errors, warnings, and key results."),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": text[-context_window:]})
    return summary + "\n\n--- Tail ---\n" + text[-tail_chars:]


#### Define Tools ####

# ---------- Schemas ----------
class ValidationInput(BaseModel): command: str
class TerminationInput(BaseModel): rationale: str
class PlanningInput(BaseModel): hypothesis: str
class CommandInput(BaseModel): command: str
# COT
class PlanHypothesesInput(BaseModel):
    content: str
class PlanEvaluateInput(BaseModel):
    content: str
class PlanActionsInput(BaseModel):
    content: str

class Phase(str, Enum):
    PLAN_EVALUATE   = "plan_evaluate"
    PLAN_ACTIONS    = "plan_actions"
    OPS             = "ops"
stage = {"phase": Phase.OPS}
# ---------- Delay Wrapper ---------- 
def with_random_delay(func, mean=3.0, stddev=0.5, min_delay=0.5, max_delay=5.0):
    @wraps(func)
    def wrapper(*args, **kwargs):
        d = random.normalvariate(mean, stddev)
        d = max(min_delay, min(d, max_delay))
        print(f"[Delay] Sleeping for {d:.2f} seconds before running tool...")
        time.sleep(d)
        return func(*args, **kwargs)
    return wrapper

# ---------- Tool factories ----------
def make_validation_tool(session, summarize_terminal: Callable[[str], str], is_safe_command: Callable[[str], bool],
                         delay_mean, delay_std, delay_min, delay_max):
    def emit_validation_command(command: str) -> str:
        if stage["phase"] == Phase.PLAN_EVALUATE:
            return f"ERROR: Planning not complete. Call plan_evaluate next."
        if stage["phase"] == Phase.PLAN_ACTIONS:
            return f"ERROR: Planning not complete. Call plan_actions next."
        if not is_safe_command(command):
            return f"VALIDATION_COMMAND: {command}\nOUTPUT:\n[blocked: command deemed unsafe]"
        try:
            output = summarize_terminal(session.run_cmd(command))
        except Exception as e:
            return f"VALIDATION_COMMAND: {command}\nOUTPUT:\n[error: {e}]"
        return f"VALIDATION_COMMAND: {command}\nOUTPUT:\n{output}"

    delayed = with_random_delay(emit_validation_command, delay_mean, delay_std, delay_min, delay_max)
    return StructuredTool.from_function(
        func=delayed,
        name="validation",
        description="Run ONE concrete validation command and return its output.",
        args_schema=ValidationInput,
    )

def make_next_command_tool(session, summarize_terminal: Callable[[str], str], is_safe_command: Callable[[str], bool],
                           delay_mean, delay_std, delay_min, delay_max):
    def emit_next_command(command: str) -> str:
        if stage["phase"] == Phase.PLAN_EVALUATE:
            return f"ERROR: Planning not complete. Call plan_evaluate next."
        if stage["phase"] == Phase.PLAN_ACTIONS:
            return f"ERROR: Planning not complete. Call plan_actions next."
        if not is_safe_command(command):
            return f"NEXT_COMMAND: {command}\nOUTPUT:\n[blocked: command deemed unsafe]"
        try:
            output = summarize_terminal(session.run_cmd(command))
        except Exception as e:
            return f"NEXT_COMMAND: {command}\nOUTPUT:\n[error: {e}]"
        return f"NEXT_COMMAND: {command}\nOUTPUT:\n{output}"

    delayed = with_random_delay(emit_next_command, delay_mean, delay_std, delay_min, delay_max)
    return StructuredTool.from_function(
        func=delayed,
        name="next_command",
        description="Run ONE diagnostic/repair command and return its output.",
        args_schema=CommandInput,
    )

def make_termination_tool():
    def terminate(rationale: str) -> str:
        return f"TERMINATE: {rationale}"

    return StructuredTool.from_function(
        func=terminate,
        name="terminate",
        description="Terminate if success is proven by a passed validation. Provide a brief rationale.",
        args_schema=TerminationInput,
    )

# currently not used
def make_planning_tool():
    def emit_plan(hypothesis: str) -> str:
        return f"PLAN:\n{hypothesis}"
    return StructuredTool.from_function(
        func=emit_plan,
        name="plan",
        description="Propose the next investigative hypotheses or a short plan (3–6 bullets).",
        args_schema=PlanningInput,
    )

def make_plan_hypotheses_tool():
    def emit(content: str) -> str:
        if stage["phase"] == Phase.PLAN_EVALUATE:
            return f"ERROR: Incorrect planning step. Call plan_evaluate next."
        if stage["phase"] == Phase.PLAN_ACTIONS:
            return f"ERROR: Incorrect planning step. Call plan_actions next."
        stage["phase"] = Phase.PLAN_EVALUATE
        return "PLAN_HYPOTHESES:\n" + content
    return StructuredTool.from_function(func=emit, name="plan_hypotheses",
        description="Stage 1/3. Create assumptions and 2–4 ranked hypotheses.",
        args_schema=PlanHypothesesInput)

def make_plan_evaluate_tool():
    def emit(content: str) -> str:
        if stage["phase"] == Phase.OPS:
            return f"ERROR: Incorrect planning step. Start with plan_hypotheses."
        if stage["phase"] == Phase.PLAN_ACTIONS:
            return f"ERROR: Incorrect planning step. Call plan_actions next."
        stage["phase"] = Phase.PLAN_ACTIONS
        return "PLAN_EVALUATE:\n" + content
    return StructuredTool.from_function(func=emit, name="plan_evaluate",
        description="Stage 2/3 (after plan_hypotheses). Identify the MOST PROBABLE hypothesis from the previous step and justify it.",
        args_schema=PlanEvaluateInput)

def make_plan_actions_tool():
    def emit(content: str) -> str:
        if stage["phase"] == Phase.OPS:
            return f"ERROR: Incorrect planning step. Start with plan_hypotheses."
        if stage["phase"] == Phase.PLAN_EVALUATE:
            return f"ERROR: Incorrect planning step. Call plan_evaluate next."
        stage["phase"] = Phase.OPS
        return "PLAN_ACTIONS:\n" + content
    return StructuredTool.from_function(func=emit, name="plan_actions",
        description="Stage 3/3 (after plan_evaluate). Produce a short, safe action plan of concrete steps. Keep the plan at a high level and abstract — describe what type of checks or fixes should be done.",
        args_schema=PlanActionsInput)


# --------------------------- main ---------------------------
if __name__ == "__main__":


    # =====================Settings-START=====================
    # sumarize
    MODEL_TERMINAL_SUMMARIZER = "gpt-4.1"
    MAX_LENGTH_TERMINAL_OUTPUT = 2000
    TERMINAL_CONTEXT_WINDOW = 10_000
    TAIL_LENGTH_TERMINAL = 500

    # time
    VARIABILITY = 0.2
    MEAN_DELAY = 3
    MIN_DELAY = 0
    MAX_DELAY = 5

    # admin agent
    MODEL_ADMIN_AGENT = "gpt-4.1-mini"
    NUMBER_OF_INTERACTIONS = 30
    TEMPERATURE = 0
    ISSUE = "Nextcloud is returning an HTTP 500 (Internal Server Error)."
    # =====================Settings-END======================

    session = ShellSession()
    session.connect_root_setSentinel()
    # set environment variable for cleaner output
    session.run_cmd("export SYSTEMD_URLIFY=0; export SYSTEMD_PAGER=; export SYSTEMD_COLORS=0")
    # set environment variable to extract new logs
    session.run_cmd('POS_nextcloud=$(stat -c %s /var/www/nextcloud/data/nextcloud.log)')
    session.run_cmd('POS_audit=$(stat -c %s /var/log/audit/audit.log)')

    # initialization
    result = {"output": "", "intermediate_steps": []}
    try:

        summarize_parametrized = partial(summarize_terminal, model=MODEL_TERMINAL_SUMMARIZER, max_terminal_output=MAX_LENGTH_TERMINAL_OUTPUT, context_window=TERMINAL_CONTEXT_WINDOW, tail_chars=TAIL_LENGTH_TERMINAL)
        time_param_dict = {"delay_mean":MEAN_DELAY, "delay_std":VARIABILITY*MEAN_DELAY, "delay_min":MIN_DELAY, "delay_max":MAX_DELAY}

        # 1) Build tools *after* session exists
        validation_tool  = make_validation_tool(session, summarize_parametrized, is_safe_command, **time_param_dict)
        command_tool     = make_next_command_tool(session, summarize_parametrized, is_safe_command, **time_param_dict)
        planning_tool    = make_planning_tool() #  currently not used
        termination_tool = make_termination_tool()
        plan_hypotheses_tool = make_plan_hypotheses_tool()
        plan_evaluate_tool   = make_plan_evaluate_tool()
        plan_actions_tool    = make_plan_actions_tool()
        tools = [
            plan_hypotheses_tool,
            plan_evaluate_tool,
            plan_actions_tool,
            validation_tool,
            termination_tool,
            #planning_tool,
            command_tool,
        ]


        # 2) LLM and prompt, force LLM to be non-streaming
        base_llm = ChatOpenAI(model=MODEL_ADMIN_AGENT, temperature=TEMPERATURE, api_key=API_KEY)
        nonstream_llm = RunnableLambda(lambda msgs, **kw: base_llm.invoke(msgs, **kw))
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system",
            f"Reference:\n"
            f"Examples - Admin Routines:\n{examples_content}\n\n"
            f"Cheat Sheet - occ Commands:\n{cheatsheet_content}\n\n"
            f"File Editing (non-interactive):\n{fileEditingRoutine}\n\n"
            "You are an operations remediator for Ubuntu 24.04 running a LAMP stack with Nextcloud (PHP 8.3.6).\n\n"

            "=== Guidelines ===\n"
            "- Prefer read-only checks first; avoid destructive or state-changing operations.\n"
            "- Keep commands ONE line only.\n"
            "- Keep output concise: when inspecting logs or large files, always use filters such as grep, tail, or head.\n"
            "- Never use interactive or full-screen tools (no less, vi, interactive top, etc.).\n"
            "- You are connected via SSH in a root shell (sudo -i).\n"
            "- Environment variables are always set:\n"
            "    export SYSTEMD_URLIFY=0\n"
            "    export SYSTEMD_PAGER=\n"
            "    export SYSTEMD_COLORS=0\n\n"

            "=== Nextcloud specifics ===\n"
            "- Installed at: /var/www/nextcloud\n"
            "- occ commands must be run as: /usr/bin/php /var/www/nextcloud/occ\n\n"

            "=== Logs ===\n"
            "- Apache error log: /var/log/apache2/nextcloud.local-error.log\n"
            "- Apache access log: /var/log/apache2/nextcloud.local-access.log\n\n"

            "=== Available system tools ===\n"
            "- curl (HTTP checks)\n"
            "- net-tools (network diagnostics)\n"
            "- top (non-interactive only, e.g. top -b -n 1)\n\n"

            "=== Agent tools ===\n"
            "- validation: run ONE concrete health check (e.g., curl/status, systemd health) and return output.\n"
            "- next_command: run ONE low-risk diagnostic/repair command and return output.\n"
            "- plan: use the tools plan_hypotheses -> plan_evaluate -> plan_actions.\n"
            "- terminate: stop once a passed validation proves the issue is resolved.\n"
             "=== Planning discipline ===\n"
            "- Tools are HARD-GATED. You MUST call plan_hypotheses → plan_evaluate → plan_actions in order.\n"
            "- validation / next_command / terminate are unavailable until planning is complete. "
            "Out-of-order calls will return an ERROR string you must correct.\n",
            ),
            ("human", "{input}"),
            # (optional) keep conversation state; harmless if you don't use it:
            MessagesPlaceholder(variable_name="chat_history"),
            # REQUIRED by create_openai_tools_agent:
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 3) Build agent *after* tools exist
        agent_runnable = create_openai_tools_agent(nonstream_llm, tools, prompt_tmpl)
        agent = AgentExecutor(
            agent=agent_runnable,
            tools=tools,
            verbose=True,
            max_iterations=NUMBER_OF_INTERACTIONS,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        # 4) Run
        prompt = f""" Issue: {ISSUE} \n """
        result = agent.invoke(
            {"input": prompt, "chat_history": []}
        )


    finally:
        # Logs / transcript (optional)
        with open("final_report.txt", "w", encoding="utf-8") as f:
            f.write(result.get("output", ""))
        
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("\n===== STEPS =====")
            for i, (action, observation) in enumerate(result.get("intermediate_steps", []), 1):
                f.write(f"\n--- Step {i} ---")
                f.write(f"Obs:\n{observation}")
        # extract new logs and write to file
        logs = session.run_cmd('tail -c +$((POS_nextcloud+1)) /var/www/nextcloud/data/nextcloud.log')
        with open("LLM_nextcloud.log", "w", encoding="utf-8") as f:
            f.write(logs)
        logs = session.run_cmd('tail -c +$((POS_audit+1)) /var/log/audit/audit.log')
        with open("LLM_audit.log", "w", encoding="utf-8") as f:
            f.write(logs)
        session.close()
    
