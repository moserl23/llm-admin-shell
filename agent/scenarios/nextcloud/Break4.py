"""Nextcloud scenario for a missing Files app controller.

The break simulates a partial deploy or corrupted installation: login can still
succeed, but the Files app itself fails afterward.
"""

from ...utils import ShellSession


LIVE_FILE = "/var/www/nextcloud/apps/files/lib/Controller/ViewController.php"
BACKUP_FILE = "/var/backups/nextcloud/www/apps/files/lib/Controller/ViewController.php"


def config(session):
    """Remove a core controller from the Files app.

    The rest of Nextcloud stays intact, but opening the Files view should fail
    because the live application tree is now incomplete.
    """
    session.run_cmd(f"sudo rm -f {LIVE_FILE}")

    # Reload Apache so stale opcode or template state does not mask the break.
    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore the missing controller from the known-good backup.

    Ownership is reset after the copy so the restored file matches the shipped
    application tree.
    """
    # Fail early if the backup is unavailable; restoring a different file would
    # make the scenario non-deterministic.
    session.run_cmd(
        f"test -f {BACKUP_FILE} || "
        f"(echo 'Backup file missing! Cannot restore.' && exit 1)"
    )

    session.run_cmd(f"sudo cp {BACKUP_FILE} {LIVE_FILE}")

    session.run_cmd(f"sudo chown www-data:www-data {LIVE_FILE}")
    session.run_cmd(f"sudo chmod 644 {LIVE_FILE}")

    session.run_cmd("sudo systemctl reload apache2 || true")

    # ``occ status`` is a cheap sanity check that the application boots again.
    session.run_cmd("sudo -u www-data php /var/www/nextcloud/occ status || true")



























# ---- Scenario entry point ----
# Symptom: the Files section in Nextcloud fails after login.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # to break
    #fix(session)     # to fix

    session.close()
