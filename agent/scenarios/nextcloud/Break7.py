"""Nextcloud scenario for an Apache vhost pointing at the wrong root.

Removing the explicit ``DocumentRoot`` makes Apache fall back to the default
web root, so the site appears present but clearly wrong.
"""

from ...utils import ShellSession

VHOST = "/etc/apache2/sites-enabled/nextcloud.conf"


def config(session):
    """Remove the vhost's explicit ``DocumentRoot``.

    Apache then serves its default root instead of the Nextcloud tree, which
    produces a misleading but reproducible presentation failure.
    """
    # Delete the directive rather than editing it in place so repeated runs
    # remain idempotent.
    session.run_cmd(
        rf'''sudo sed -i -E \
        "/^[[:space:]]*DocumentRoot[[:space:]]+/d" \
        {VHOST}'''
    )

    session.run_cmd("sudo systemctl reload apache2")


def fix(session):
    """Restore the expected vhost ``DocumentRoot`` for Nextcloud.

    Existing directives are removed first so the repair does not accumulate
    conflicting roots across repeated runs.
    """
    session.run_cmd(
        rf'''sudo sed -i -E \
        "/^[[:space:]]*DocumentRoot[[:space:]]+/d" \
        {VHOST}'''
    )

    session.run_cmd(
        rf'''sudo sed -i \
        "1i DocumentRoot /var/www/nextcloud" \
        {VHOST}'''
    )

    session.run_cmd("sudo systemctl reload apache2")






















# ---- Scenario entry point ----
# Symptom: Nextcloud resolves to the wrong document root and looks broken.

if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    session.deactivate_history()

    config(session)  # call this to break
    #fix(session)      # call this to fix

    session.close()
