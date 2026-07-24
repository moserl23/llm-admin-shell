"""Nextcloud scenario for a database credential mismatch.

The break edits ``config.php`` directly so the failure persists even after the
application can no longer use ``occ``.
"""

from ...utils import ShellSession


def config(session):
    """Replace the stored database credentials with plausible wrong values.

    The edit is applied directly to ``config.php`` because once database access
    fails, ``occ`` is no longer a reliable recovery path.
    """
    # Direct file edits keep the scenario reproducible even when Nextcloud
    # cannot bootstrap far enough to run its CLI.
    session.run_cmd(r'''sudo sed -i -E "s/('dbuser'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'nc_user',/" /var/www/nextcloud/config/config.php''')
    session.run_cmd(r'''sudo sed -i -E "s/('dbpassword'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'NcApp_2025',/" /var/www/nextcloud/config/config.php''')

    # Reloading Apache is not strictly required, but it makes the failure
    # visible on the next request without waiting for process reuse.
    session.run_cmd(r'sudo systemctl reload apache2 || true')


def fix(session):
    """Restore the known-good database credentials in ``config.php``.

    The fix also bypasses ``occ`` so it remains usable even while the
    application is still unable to connect to MySQL.
    """
    session.run_cmd(r'''sudo sed -i -E "s/('dbuser'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'nextcloud',/" /var/www/nextcloud/config/config.php''')
    session.run_cmd(r'''sudo sed -i -E "s/('dbpassword'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'passw0rd',/" /var/www/nextcloud/config/config.php''')

    # Use the same direct edit path here so recovery does not depend on the app
    # already being healthy.
    session.run_cmd(r'sudo systemctl reload apache2 || true')

    return
    ''' OLD '''
    # Create the new DB user and grant privileges on the REAL DB name
    session.run_cmd(r'''sudo mysql -e "CREATE USER IF NOT EXISTS 'nc_user2'@'localhost' IDENTIFIED BY 'password'; GRANT ALL PRIVILEGES ON nextclouddb.* TO 'nc_user2'@'localhost'; FLUSH PRIVILEGES;"''')

    # Update config.php directly to the new working creds
    session.run_cmd(r'''sudo sed -i -E "s/('dbuser'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'nc_user2',/" /var/www/nextcloud/config/config.php''')
    session.run_cmd(r'''sudo sed -i -E "s/('dbpassword'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'password',/" /var/www/nextcloud/config/config.php''')

    # Optionally toggle maintenance mode around the fix (not strictly required)
    session.run_cmd(r'sudo -u www-data php /var/www/nextcloud/occ maintenance:mode --on || true')
    session.run_cmd(r'sudo systemctl reload apache2 || true')
    session.run_cmd(r'sudo -u www-data php /var/www/nextcloud/occ maintenance:mode --off || true')

    # Verify (now that DB works, occ should run fine)
    session.run_cmd(r'sudo -u www-data php /var/www/nextcloud/occ status')
    



























# ---- Scenario entry point ----
# Symptom: Nextcloud returns an internal server error.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()
    
    config(session)  # call this to break
    #fix(session)       # call this to fix
    
    session.close()
