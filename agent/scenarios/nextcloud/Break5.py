"""Nextcloud scenario for an unrealistically low PHP memory limit.

The site remains reachable, but ordinary requests can fail once the PHP worker
exhausts memory.
"""

from ...utils import ShellSession


def config(session):
    """Lower Apache PHP memory enough to trigger runtime failures.

    The override is written as a dedicated drop-in so the mutation stays
    explicit and easy to revert.
    """
    session.run_cmd("""sudo tee /etc/php/8.3/apache2/conf.d/99-memory-limit.ini >/dev/null <<'EOF'
; Custom PHP overrides
memory_limit = 8M
EOF""")
    session.run_cmd("sudo systemctl reload apache2")


def fix(session):
    """Restore a reasonable PHP memory limit for the local deployment.

    Reusing the same override file keeps the fix idempotent across repeated
    scenario runs.
    """
    session.run_cmd("""sudo tee /etc/php/8.3/apache2/conf.d/99-memory-limit.ini >/dev/null <<'EOF'
memory_limit = 512M
EOF""")
    session.run_cmd("sudo systemctl reload apache2")

























# ---- Scenario entry point ----
# Symptom: Nextcloud behaves incorrectly because PHP runs out of memory.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()
    
    config(session)  # call this to break
    #fix(session)       # call this to fix
    
    session.close()
