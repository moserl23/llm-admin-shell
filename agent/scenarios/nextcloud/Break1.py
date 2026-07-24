"""Nextcloud scenario for trusted-domain drift.

The break keeps the configuration plausible while removing the expected
canonical host, which blocks normal access without taking the service down.
"""

from ...utils import ShellSession


def config(session):
    """Introduce a subtle trusted-domain misconfiguration.

    The list is rebuilt with plausible but incorrect entries so the instance
    looks configured, yet requests no longer match the intended host.
    """
    # Rebuild the list from scratch so the resulting state is deterministic.
    session.run_cmd(
        r"for i in $(seq 0 10); do "
        r"sudo -u www-data php /var/www/nextcloud/occ config:system:delete trusted_domains $i || true; "
        r"done"
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 0 --value="nextcloud.local."'
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 1 --value="nextclouds.local"'
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 2 --value="localhost"'
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 3 --value="127.0.0.1"'
    )

    # Keep the surrounding URL settings looking reasonable so the failure
    # points back to trusted-domain validation rather than the web server.
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
    """Restore the expected trusted-domain list.

    The list is recreated from scratch so repeated runs do not leave stale
    entries behind.
    """
    # Clear the array first to avoid preserving misleading domains.
    session.run_cmd(
        r"for i in $(seq 0 10); do "
        r"sudo -u www-data php /var/www/nextcloud/occ config:system:delete trusted_domains $i || true; "
        r"done"
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 0 --value="nextcloud.local"'
    )

    session.run_cmd(
        r'sudo -u www-data php /var/www/nextcloud/occ '
        r'config:system:set trusted_domains 1 --value="nextclouds.local"'
    )

    session.run_cmd(r'systemctl reload apache2 || true')
    session.run_cmd(r'sudo -u www-data php /var/www/nextcloud/occ status')

































# ---- Scenario entry point ----
# Symptom: login or normal access is no longer possible.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # to break
    #fix(session)      # to fix
    
    session.close()
