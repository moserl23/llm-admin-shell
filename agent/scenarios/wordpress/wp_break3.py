"""WordPress scenario for a non-writable uploads directory.

The public site stays reachable, but the admin can no longer persist new media
to ``wp-content/uploads``.
"""

from ...utils import ShellSession

UPLOADS = "/var/www/wordpress/wp-content/uploads"


def config(session):
    """Remove write access from the uploads directory.

    Existing media remains readable, but new uploads fail because the web
    server no longer owns the target tree.
    """
    session.run_cmd(rf"sudo mkdir -p {UPLOADS}")

    # Root ownership preserves readability while blocking Apache from creating
    # or modifying files in the uploads tree.
    session.run_cmd(rf"sudo chown -R root:root {UPLOADS}")
    session.run_cmd(rf"sudo chmod -R 755 {UPLOADS}")

    session.run_cmd("sudo systemctl reload apache2 || true")


def fix(session):
    """Restore the standard uploads ownership and file modes.

    Directories and files are reset separately so the tree matches the usual
    WordPress permission pattern.
    """
    session.run_cmd(rf"sudo chown -R www-data:www-data {UPLOADS}")
    session.run_cmd(rf"sudo find {UPLOADS} -type d -exec chmod 755 {{}} \;")
    session.run_cmd(rf"sudo find {UPLOADS} -type f -exec chmod 644 {{}} \;")

    session.run_cmd("sudo systemctl reload apache2 || true")





















# ---- Scenario entry point ----
# Symptom: media uploads fail in the WordPress admin.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)   # call this to break
    #fix(session)     # call this to fix

    session.close()
