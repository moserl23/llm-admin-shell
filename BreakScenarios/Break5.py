import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


def config(session):
    session.run_cmd("""sudo tee /etc/php/8.3/apache2/conf.d/99-memory-limit.ini >/dev/null <<'EOF'
; Custom PHP overrides
memory_limit = 8M
EOF""")
    session.run_cmd("sudo systemctl reload apache2")


def fix(session):
    session.run_cmd("""sudo tee /etc/php/8.3/apache2/conf.d/99-memory-limit.ini >/dev/null <<'EOF'
memory_limit = 512M
EOF""")
    session.run_cmd("sudo systemctl reload apache2")


if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()
    
    #config(session)  # call this to break
    fix(session)       # call this to fix
    
    session.close()
