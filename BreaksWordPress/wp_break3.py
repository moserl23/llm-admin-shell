import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

UPLOADS = "/var/www/wordpress/wp-content/uploads"


def config(session):
    """
    BREAK:
    Make Media uploads fail by breaking permissions on wp-content/uploads.
    WordPress frontend still works, but uploading images/files fails.
    """
    # Ensure uploads directory exists
    session.run_cmd(rf"sudo mkdir -p {UPLOADS}")

    # Break: root owns it and webserver can't write
    session.run_cmd(rf"sudo chown -R root:root {UPLOADS}")
    session.run_cmd(rf"sudo chmod -R 755 {UPLOADS}")

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """
    FIX:
    Restore correct ownership/permissions so uploads work again.
    """
    session.run_cmd(rf"sudo chown -R www-data:www-data {UPLOADS}")
    session.run_cmd(rf"sudo find {UPLOADS} -type d -exec chmod 755 {{}} \;")
    session.run_cmd(rf"sudo find {UPLOADS} -type f -exec chmod 644 {{}} \;")

    session.run_cmd("sudo systemctl reload apache2 || true")





















# Problem: Media upload fails in WordPress admin

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # call this to break
    #fix(session)     # call this to fix

    session.close()
