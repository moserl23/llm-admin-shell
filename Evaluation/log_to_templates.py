from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import re

# -----------------------------------------
# High-quality preprocessing (your version)
# -----------------------------------------

AUDIT_MSG_RE = re.compile(r"msg=audit\([^)]+\):")
LONG_HEX_RE  = re.compile(r"\b[0-9a-fA-F]{12,}\b")
HEX_RE       = re.compile(r"\b0x[0-9a-fA-F]+\b")
PATH_RE      = re.compile(r'(?P<q>["\'])(/[^"\']+)(?P=q)')
IP_RE        = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
NUM_RE       = re.compile(r"\b\d+\b")

def preprocess(line: str) -> str:
    s = line.strip()
    if not s:
        return s
    s = AUDIT_MSG_RE.sub("msg=audit(<AUDIT_META>):", s)
    s = PATH_RE.sub('"/PATH"', s)
    s = LONG_HEX_RE.sub("<HEX>", s)
    s = HEX_RE.sub("<HEX>", s)
    s = IP_RE.sub("<IP>", s)
    s = NUM_RE.sub("<NUM>", s)
    return s


# --------------------------------------------------
# Convert logs → templates using Drain3
# --------------------------------------------------

def logs_to_templates(log_lines):
    cfg = TemplateMinerConfig()
    cfg.load("drain3.ini")
    miner = TemplateMiner(config=cfg, persistence_handler=None)

    cluster_ids = []
    for line in log_lines:
        msg = preprocess(line)
        result = miner.add_log_message(msg)
        cluster_ids.append(result["cluster_id"])

    # Get FINAL templates after all logs are processed
    final_templates = {c.cluster_id: c.get_template()
                       for c in miner.drain.clusters}

    # Map each log to its final template
    template_output = [final_templates[cid] for cid in cluster_ids]

    return template_output


# --------------------------------------------------
# Main Example
# --------------------------------------------------

if __name__ == "__main__":

    ai_logs = [
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003e syscall=59 success=yes comm="php" exe="/usr/bin/php8.3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.310:55983): argc=5 a0="curl" a1="-s" a2="-o" a3="out.txt" a4="https://example.com/data"',
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003e syscall=59 success=no comm="php" exe="/usr/bin/php8.3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.310:55983): argc=5 a0="curl" a1="-s" a2="-o" a3="out.txt" a4="google"',
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003e syscall=59 success=no exe="/usr/bin/php8.3"',
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003e syscall=59 success=no comm="php"',
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003ed syscall=59 success=no comm="php" exe="/usr/bin/php8.3"',
        # ...
    ]

    templates = logs_to_templates(ai_logs)

    print("\n==== TEMPLATE OUTPUT ====\n")
    for original, tmpl in zip(ai_logs, templates):
        print("ORIGINAL :", original)
        print("TEMPLATE :", tmpl)
        print()


