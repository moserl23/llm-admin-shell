"""WordPress scenario for invalid database credentials.

The credentials are edited directly in ``wp-config.php``, which produces the
expected connection failure without relying on WP-CLI or database access.
"""

from ...utils import ShellSession

WP_CONFIG = "/var/www/wordpress/wp-config.php"

# Known-good credentials for the local lab deployment.
GOOD_DB_USER = "wp_user"
GOOD_DB_PASS = "WpApp_2025!"

# Plausible alternatives used to trigger the failure.
BAD_DB_USER = "wp_admin"
BAD_DB_PASS = "WpSecure_2025!"


def config(session):
    """Replace the configured database credentials with plausible wrong ones.

    Editing ``wp-config.php`` directly keeps the break independent of whether
    WordPress itself can still bootstrap.
    """
    # Patch the config file instead of using application tooling so the failure
    # path does not depend on a healthy WordPress runtime.
    session.run_cmd(
        rf'''sudo sed -i -E \
        "s/(define\(\s*'DB_USER'\s*,\s*)'[^']*'/\1'{BAD_DB_USER}'/" \
        {WP_CONFIG}'''
    )

    session.run_cmd(
        rf'''sudo sed -i -E \
        "s/(define\(\s*'DB_PASSWORD'\s*,\s*)'[^']*'/\1'{BAD_DB_PASS}'/" \
        {WP_CONFIG}'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore the known-good database credentials in ``wp-config.php``.

    The same direct edit path keeps the repair available even while the site is
    still returning a database-connection error.
    """
    session.run_cmd(
        rf'''sudo sed -i -E \
        "s/(define\(\s*'DB_USER'\s*,\s*)'[^']*'/\1'{GOOD_DB_USER}'/" \
        {WP_CONFIG}'''
    )

    session.run_cmd(
        rf'''sudo sed -i -E \
        "s/(define\(\s*'DB_PASSWORD'\s*,\s*)'[^']*'/\1'{GOOD_DB_PASS}'/" \
        {WP_CONFIG}'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")












































# ---- Scenario entry point ----
# Symptom: WordPress reports a database connection failure.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # call this to break
    #fix(session)     # call this to fix

    session.close()



