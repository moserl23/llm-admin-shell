"""Nextcloud scenario for a broken Redis dependency.

Nextcloud is configured to require Redis for locking and local caching, then
the PHP extension and service are removed so requests fail at runtime.
"""

from ...utils import ShellSession



def config(session):
    """Force Nextcloud to depend on Redis, then remove Redis at runtime.

    Both the PHP extension and the daemon are disabled so the break does not
    depend on a leftover local Redis process.
    """
    # Make Redis mandatory first; otherwise disabling it may not surface as a
    # user-visible failure on this installation.
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set memcache.locking --value="\OC\Memcache\Redis"')
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set memcache.local --value="\OC\Memcache\Redis"')
    session.run_cmd(
        r'sudo -u www-data -- php /var/www/nextcloud/occ '
        r'config:system:set redis host --value="127.0.0.1"'
    )
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set redis port --value=6379 --type=integer')
    session.run_cmd(r'runuser -u www-data -- php /var/www/nextcloud/occ config:system:set redis timeout --value=1.5 --type=double')

    # Break both the client library and the backing service to avoid a partial
    # success path through an already-running Redis instance.
    session.run_cmd('phpdismod -v 8.3 -s apache2 redis')
    session.run_cmd('systemctl reload apache2')

    session.run_cmd('systemctl stop redis-server')


def fix(session):
    """Re-enable the Redis integration expected by Nextcloud.

    The PHP extension is restored before the web server restart so Apache
    workers come back with Redis support available.
    """
    session.run_cmd(r'phpenmod -v 8.3 -s apache2 redis')
    session.run_cmd(r'systemctl enable --now redis-server')
    session.run_cmd(r'systemctl restart apache2')



















# ---- Scenario entry point ----
# Symptom: Nextcloud returns an internal server error.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()
    
    config(session)
    #fix(session)
    
    session.close()
