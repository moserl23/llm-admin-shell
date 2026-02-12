import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

VERSION_FILE = "/var/www/wordpress/wp-includes/version.php"


def config(session):
    """
    BREAK:
    Delete a core WordPress file.
    Result: fatal error / 500 (frontend + admin).
    """
    # Keep a backup so the fix is deterministic
    session.run_cmd(rf"sudo test -f {VERSION_FILE}.bak || sudo cp -a {VERSION_FILE} {VERSION_FILE}.bak")
    session.run_cmd(rf"sudo rm -f {VERSION_FILE}")

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """
    FIX:
    Restore the deleted core file from backup.
    """
    session.run_cmd(rf"sudo test -f {VERSION_FILE}.bak && sudo cp -a {VERSION_FILE}.bak {VERSION_FILE}")
    session.run_cmd(rf"sudo chown www-data:www-data {VERSION_FILE} || true")
    session.run_cmd(rf"sudo chmod 644 {VERSION_FILE} || true")

    session.run_cmd("sudo systemctl reload apache2 || true")


# Problem: WordPress is dead (fatal error / 500)

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    #config(session)   # break
    fix(session)     # fix

    session.close()
