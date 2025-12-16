import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession

def config(session):
    session.run_cmd(r'sudo chown -R www-data:www-data /var/www/nextcloud/data')
    session.run_cmd(r'sudo chmod -R 550 /var/www/nextcloud/data')
    # reload apache
    session.run_cmd(r'sudo systemctl reload apache2')


def fix(session):
    session.run_cmd(r'sudo chown -R www-data:www-data /var/www/nextcloud/data')
    session.run_cmd(r'sudo chmod -R 750 /var/www/nextcloud/data')
    session.run_cmd(r'sudo systemctl reload apache2')



if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # call this to break
    #fix(session)       # call this to fix
    
    session.close()


