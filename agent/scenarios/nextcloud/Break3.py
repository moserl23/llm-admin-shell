"""Nextcloud scenario for a non-writable data directory.

The web server can still read the storage tree, but write operations such as
uploads or deletes fail.
"""

from ...utils import ShellSession


def config(session):
    """Remove write access from the data directory used by Nextcloud.

    The directory remains readable, so the instance still loads while
    state-changing file operations start to fail.
    """
    session.run_cmd(r'sudo chown -R www-data:www-data /var/www/nextcloud/data')
    # Mode 550 preserves read and traversal access for ``www-data`` while
    # blocking writes, which isolates the failure to file operations.
    session.run_cmd(r'sudo chmod -R 550 /var/www/nextcloud/data')
    session.run_cmd(r'sudo systemctl reload apache2')


def fix(session):
    """Restore writable permissions on the Nextcloud data directory.

    This returns the storage area to the standard ownership and mode expected
    by the local installation.
    """
    session.run_cmd(r'sudo chown -R www-data:www-data /var/www/nextcloud/data')
    session.run_cmd(r'sudo chmod -R 750 /var/www/nextcloud/data')
    session.run_cmd(r'sudo systemctl reload apache2')























# ---- Scenario entry point ----
# Symptom: uploads and deletes in Nextcloud fail.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # call this to break
    #fix(session)       # call this to fix
    
    session.close()
