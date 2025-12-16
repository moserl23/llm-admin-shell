import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


def config(session):
    # BREAK: edit config.php directly (don’t use occ here)
    session.run_cmd(r'''sudo sed -i -E "s/('dbuser'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'nc_user',/" /var/www/nextcloud/config/config.php''')
    session.run_cmd(r'''sudo sed -i -E "s/('dbpassword'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'NcApp_2025',/" /var/www/nextcloud/config/config.php''')

    # No need to reload apache for config.php changes, but harmless if you want:
    session.run_cmd(r'sudo systemctl reload apache2 || true')


def fix(session):
    # BREAK: edit config.php directly (don’t use occ here)
    session.run_cmd(r'''sudo sed -i -E "s/('dbuser'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'nextcloud',/" /var/www/nextcloud/config/config.php''')
    session.run_cmd(r'''sudo sed -i -E "s/('dbpassword'[[:space:]]*=>[[:space:]]*)'[^']*',/\1'passw0rd',/" /var/www/nextcloud/config/config.php''')

    # No need to reload apache for config.php changes, but harmless if you want:
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
    



























# Problem: Internal Server Error 500

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()
    
    config(session)  # call this to break
    #fix(session)       # call this to fix
    
    session.close()


