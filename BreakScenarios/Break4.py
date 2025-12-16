import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


LIVE_FILE = "/var/www/nextcloud/apps/files/lib/Controller/ViewController.php"
BACKUP_FILE = "/var/backups/nextcloud/www/apps/files/lib/Controller/ViewController.php"


def config(session):
    """
    BREAK:
    Delete a post-login critical asset. Login works,
    but the Files app breaks after login (JS 404).
    """
    # Remove the JS file
    session.run_cmd(f"sudo rm -f {LIVE_FILE}")

    # Reload Apache to ensure cache/template invalidation
    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """
    FIX:
    Restore the missing JS file from backup and validate.
    """

    # Ensure the backup exists
    session.run_cmd(
        f"test -f {BACKUP_FILE} || "
        f"(echo 'Backup file missing! Cannot restore.' && exit 1)"
    )

    # Restore the file
    session.run_cmd(f"sudo cp {BACKUP_FILE} {LIVE_FILE}")

    # Correct permissions
    session.run_cmd(f"sudo chown www-data:www-data {LIVE_FILE}")
    session.run_cmd(f"sudo chmod 644 {LIVE_FILE}")

    # Reload Apache
    session.run_cmd("sudo systemctl reload apache2 || true")

    # OCC status to confirm Nextcloud loads properly again
    session.run_cmd("sudo -u www-data php /var/www/nextcloud/occ status || true")


if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    #config(session)  # to break
    fix(session)     # to fix

    session.close()
