import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

WP_CONFIG = "/var/www/wordpress/wp-config.php"


def config(session):
    """
    BREAK:
    Introduce a PHP syntax error into wp-config.php.
    Result: parse error / 500 on every request.
    """
    # Backup first for deterministic restore
    session.run_cmd(rf"sudo test -f {WP_CONFIG}.bak || sudo cp -a {WP_CONFIG} {WP_CONFIG}.bak")

    # Add an invalid PHP line near the top (after <?php)
    session.run_cmd(
        rf'''sudo sed -i '2i\this_is_not_valid_php(;' {WP_CONFIG}'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """
    FIX:
    Restore wp-config.php from backup.
    """
    session.run_cmd(rf"sudo test -f {WP_CONFIG}.bak && sudo cp -a {WP_CONFIG}.bak {WP_CONFIG}")
    session.run_cmd("sudo systemctl reload apache2 || true")


# Problem: WordPress is dead (500)

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    #config(session)   # break
    fix(session)     # fix

    session.close()
