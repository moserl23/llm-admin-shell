

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

VHOST = "/etc/apache2/sites-enabled/nextcloud.conf"


def config(session):
    """
    BREAK:
    Disable .htaccess processing for Nextcloud by
    switching AllowOverride All -> None.
    """
    session.run_cmd(
        f"sudo sed -i 's/AllowOverride All/AllowOverride None/' {VHOST}"
    )
    session.run_cmd("sudo systemctl reload apache2")


def fix(session):
    """
    FIX:
    Re-enable .htaccess processing for Nextcloud by
    switching AllowOverride None -> All.
    """
    session.run_cmd(
        f"sudo sed -i 's/AllowOverride None/AllowOverride All/' {VHOST}"
    )
    session.run_cmd("sudo systemctl reload apache2")



















# Problem: Wrong file URLs are not correctly handled by Nextcloud.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    #config(session)
    fix(session)    

    session.close()
