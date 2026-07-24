"""WordPress scenario for a syntax error in ``wp-config.php``.

The injected parse error happens before WordPress can bootstrap, so every
request returns a server error.
"""

from ...utils import ShellSession

WP_CONFIG = "/var/www/wordpress/wp-config.php"


def config(session):
    """Inject a deterministic PHP syntax error into ``wp-config.php``.

    The bad line is inserted near the top of the file so PHP fails before any
    WordPress logic can run.
    """
    # Keep a backup so the exact original file can be restored.
    session.run_cmd(rf"sudo test -f {WP_CONFIG}.bak || sudo cp -a {WP_CONFIG} {WP_CONFIG}.bak")

    # Insert the malformed statement immediately after ``<?php`` so the parse
    # error is triggered on every request.
    session.run_cmd(
        rf"""sudo sed -i "2i\define('WP_DEBUG', true;" {WP_CONFIG}"""
    )

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore ``wp-config.php`` from the preserved backup.

    Using the backup avoids trying to surgically remove the injected line from
    a file that may already have changed elsewhere.
    """
    session.run_cmd(rf"sudo test -f {WP_CONFIG}.bak && sudo cp -a {WP_CONFIG}.bak {WP_CONFIG}")
    session.run_cmd("sudo systemctl reload apache2 || true")















# ---- Scenario entry point ----
# Symptom: WordPress returns HTTP 500.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # break
    #fix(session)     # fix

    session.close()
