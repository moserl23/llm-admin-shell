import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

WP_CONFIG = "/var/www/wordpress/wp-config.php"

# Correct credentials (from your server)
GOOD_DB_USER = "wp_user"
GOOD_DB_PASS = "WpApp_2025!"

# Wrong-but-plausible credentials (for the break)
BAD_DB_USER = "wp_admin"
BAD_DB_PASS = "WpSecure_2025!"


def config(session):
    """
    BREAK:
    Change DB credentials in wp-config.php so WordPress shows
    "Error establishing a database connection".
    """
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
    """
    FIX:
    Restore the correct DB credentials in wp-config.php.
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


# Problem: WordPress shows "Error establishing a database connection"

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    #config(session)   # call this to break
    fix(session)     # call this to fix

    session.close()
