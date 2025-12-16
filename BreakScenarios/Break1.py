import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


def config(session):
    # BREAK: wipe trusted_domains, add sneaky wrong one
    session.run_cmd(
        r"for i in $(seq 0 10); do "
        r"sudo -u www-data php /var/www/nextcloud/occ config:system:delete trusted_domains $i || true; "
        r"done"
    )

    # Slightly modified, wrong domain
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 0 --value="nextcloud.local."'
    )

    # Second one also slightly modified!
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 1 --value="nextclouds.local"'
    )

    # Third misleading domain
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 2 --value="localhost"'
    )

    # Fourth misleading domain
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 3 --value="127.0.0.1"'
    )

    # harmless overwrite settings
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set overwrite.cli.url --value="http://nextcloud.local"'
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set overwriteprotocol --value="http"'
    )

    session.run_cmd(r'systemctl reload apache2 || true')


def fix(session):
    # FIX: restore correct trusted_domains
    session.run_cmd(
        r"for i in $(seq 0 10); do "
        r"sudo -u www-data php /var/www/nextcloud/occ config:system:delete trusted_domains $i || true; "
        r"done"
    )

    # Correct primary domain
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 0 --value="nextcloud.local"'
    )

    # Restore secondary domain
    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 1 --value="nextclouds.local"'
    )

    session.run_cmd(r'systemctl reload apache2 || true')
    session.run_cmd(r'sudo -u www-data php /var/www/nextcloud/occ status')

































# Problem: Login is not possible.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # to break
    #fix(session)      # to fix
    
    session.close()
