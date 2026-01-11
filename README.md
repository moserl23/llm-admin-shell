# Nextcloud Admin LLM Agent

An LLM-based automation agent for troubleshooting and fixing Nextcloud installations using shell access, browser automation, and safe file editing.

---

## Requirements

- Python 3.10+
- Node.js (required for browser_agent.py)
- SSH key-based access

---

## Setup

### Install Dependencies

pip install -r requirements.txt

### Environment Configuration

Create a .env file in the repository root (used for all environment variables):

OPENAI_API_KEY=your_key_here

### Run the Agent

python LLM_Agent.py

---

## Configuration

Edit AgentConfig in LLM_Agent.py to define:

- problem_prompt (system / task prompt)
- Model selection
- Hyperparameters

---

## IMPORTANT SSH ASSUMPTIONS

- ssh final_arena_server works without password prompts
- sudo -i works without a password
- The agent runs commands as root via an interactive shell

---

## Notes

- browser_agent.py requires Node.js and npx
- File edits are handled by a dedicated Vim edit agent
