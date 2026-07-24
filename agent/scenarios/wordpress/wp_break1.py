"""WordPress scenario for a ``siteurl``/``home`` mismatch.

Core still boots from the correct ``siteurl``, but generated navigation links
use the wrong base path and lead to real 404s.
"""

from ...utils import ShellSession


def config(session):
    """Create a mismatch between WordPress bootstrap and navigation URLs.

    The front page remains reachable because ``siteurl`` stays correct, while
    internal links are generated from a broken ``home`` value.
    """

    # Keep ``siteurl`` correct so WordPress can still bootstrap normally.
    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local'
        WHERE option_name='siteurl';
        "'''
    )

    # Break only ``home`` so navigation fails without taking down the front page.
    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local/site'
        WHERE option_name='home';
        "'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore a consistent base URL for WordPress.

    Resetting both options together avoids partial repairs that still leave
    generated links incorrect.
    """

    session.run_cmd(
        r'''sudo mysql -e "
        UPDATE wordpress.wp_options
        SET option_value='http://wordpress.local'
        WHERE option_name IN ('siteurl','home');
        "'''
    )

    session.run_cmd("sudo systemctl reload apache2 || true")








# ---- Scenario entry point ----
# Symptom: the front page loads, but internal navigation returns 404s.


if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # call this to break
    #fix(session)    # call this to fix

    session.close()
