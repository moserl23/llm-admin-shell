import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


def config(session):
    """
    BREAK:
    WordPress base URL path mismatch.

    Front page loads normally.
    All internal links point to a non-existing sub-path (/blobdoesnotexist),
    which causes real HTTP 404 errors.
    """

    # Keep correct siteurl so WP loads
    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local'
        WHERE option_name='siteurl';
        "'''
    )

    # Break homepage base path
    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local/site'
        WHERE option_name='home';
        "'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """
    FIX:
    Restore correct WordPress base URL.
    """

    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local'
        WHERE option_name IN ('siteurl','home');
        "'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")








# Problem:
# WordPress front page loads, but navigation links return HTTP 404 (non-existing path).


if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # call this to break
    #fix(session)    # call this to fix

    session.close()