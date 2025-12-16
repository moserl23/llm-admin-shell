import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

VHOST = "/etc/apache2/sites-enabled/nextcloud.conf"


def config(session):
    """
    BREAK:
    Remove the DocumentRoot directive from the Nextcloud vHost.
    Apache will fall back to /var/www/html.
    """
    session.run_cmd(
        rf'''sudo sed -i -E \
        "/^[[:space:]]*DocumentRoot[[:space:]]+/d" \
        {VHOST}'''
    )

    session.run_cmd("sudo systemctl reload apache2")


def fix(session):
    """
    FIX:
    Restore the correct DocumentRoot for Nextcloud.
    """
    session.run_cmd(
        rf'''sudo sed -i -E \
        "/^[[:space:]]*DocumentRoot[[:space:]]+/d" \
        {VHOST}'''
    )

    session.run_cmd(
        rf'''sudo sed -i \
        "1i DocumentRoot /var/www/nextcloud" \
        {VHOST}'''
    )

    session.run_cmd("sudo systemctl reload apache2")






















# Problem: Nextcloud doesnot look right!

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # call this to break
    #fix(session)      # call this to fix

    session.close()
