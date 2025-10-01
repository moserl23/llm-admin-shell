import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ShellSession


def config(session):

    session.run_cmd(r'cd /root/expired-cert-test/ca')

    # 1) Create a CSR from the CURRENT key Apache uses
    session.run_cmd(r'''openssl req -new \
        -key /etc/ssl/private/nextcloud.local.key \
        -out /root/expired-cert-test/nextcloud.local.csr \
        -subj "/CN=nextcloud.local" \
        -addext "subjectAltName=DNS:nextcloud.local"''')

    # 2) Sign it with your existing CA config, with past dates (expired)
    session.run_cmd(r'''openssl ca -config /root/expired-cert-test/ca/openssl-ca.cnf \
        -in  /root/expired-cert-test/nextcloud.local.csr \
        -out /root/expired-cert-test/nextcloud.local.expired.crt \
        -startdate 20240901000000Z \
        -enddate   20250901000000Z \
        -batch''')

    # 3) Install the expired cert to the exact path Apache uses and reload
    session.run_cmd(r'''install -m 644 -o root -g root \
        /root/expired-cert-test/nextcloud.local.expired.crt \
        /etc/ssl/certs/nextcloud.local.crt''')

    session.run_cmd(r'''systemctl reload apache2''')


def fix(session):
    # new certificate/key pair using mkcert
    session.run_cmd(r'cd /root')
    session.run_cmd(r'mkcert nextcloud.local')
    session.run_cmd(r'sudo install -m 640 -o root -g ssl-cert nextcloud.local-key.pem /etc/ssl/private/nextcloud.local.key')
    session.run_cmd(r'sudo install -m 644 -o root -g root  nextcloud.local.pem     /etc/ssl/certs/nextcloud.local.crt')
    session.run_cmd(r'sudo systemctl reload apache2')



if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    #config(session)  # call this to break
    fix(session)       # call this to fix
    session.close()


