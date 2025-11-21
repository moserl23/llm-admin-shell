import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession



def config(session):
    # 1. Tell Nextcloud to use Redis
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set memcache.locking --value="\OC\Memcache\Redis"')
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set memcache.local --value="\OC\Memcache\Redis"')
    session.run_cmd('runuser -u www-data -- php /var/www/nextcloud/occ config:system:set redis host --value="/var/run/redis/redis-server.sock"')
    session.run_cmd('runuser -u www-data -- php /var/www/nextcloud/occ config:system:set redis port --value=0 --type=integer')
    session.run_cmd('runuser -u www-data -- php /var/www/nextcloud/occ config:system:set redis timeout --value=1.5 --type=double')

    # 2. Disable PHP Redis extension
    session.run_cmd('phpdismod -v 8.3 -s apache2 redis')
    session.run_cmd('systemctl reload apache2')

    # 3. Stop Redis server
    session.run_cmd('systemctl stop redis-server')


def fix(session):
    session.run_cmd(r'apt-get update && apt-get install -y php8.3-redis redis-server')
    session.run_cmd(r'phpenmod -v 8.3 -s apache2 redis')
    session.run_cmd(r'systemctl enable --now redis-server')
    session.run_cmd(r'systemctl restart apache2')



if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    
    #config(session)

    fix(session)
    
    session.close()

