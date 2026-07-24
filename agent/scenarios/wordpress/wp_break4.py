"""WordPress scenario for a missing core file.

Deleting a file from ``wp-includes`` forces WordPress to fail during
bootstrap, while a backup keeps restoration deterministic.
"""

from ...utils import ShellSession

VERSION_FILE = "/var/www/wordpress/wp-includes/version.php"


def config(session):
    """Delete a core file that WordPress needs during bootstrap.

    The failure should affect both frontend and admin requests because the
    missing file is loaded early in the request lifecycle.
    """
    # Preserve a golden copy so the repair path is deterministic.
    session.run_cmd(rf"sudo test -f {VERSION_FILE}.bak || sudo cp -a {VERSION_FILE} {VERSION_FILE}.bak")
    session.run_cmd(rf"sudo rm -f {VERSION_FILE}")

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore the deleted core file from the saved backup.

    Ownership and mode are reset afterward so the replacement matches the
    expected WordPress installation state.
    """
    session.run_cmd(rf"sudo test -f {VERSION_FILE}.bak && sudo cp -a {VERSION_FILE}.bak {VERSION_FILE}")
    session.run_cmd(rf"sudo chown www-data:www-data {VERSION_FILE} || true")
    session.run_cmd(rf"sudo chmod 644 {VERSION_FILE} || true")

    session.run_cmd("sudo systemctl reload apache2 || true")




























# ---- Scenario entry point ----
# Symptom: WordPress fails with a fatal error or HTTP 500.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # break
    #fix(session)     # fix

    session.close()
