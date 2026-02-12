import regex as re


tail_f_re = re.compile(
    r'\ba0="tail"\b(?=.*\ba\d+="-(?:c|C)"(?=\s|$))',
    re.IGNORECASE
)

tail_f_re = re.compile(r'\ba0="tail"\b.*\ba\d+="-(?:c|C)"', re.IGNORECASE)


tail_f_re = re.compile(
    r'(?<!\S)a0="tail"(?=\s|$)(?=.*(?<!\S)a\d+="-(?:c|C)"(?=\s|$))',
    re.IGNORECASE
)


ln = 'type=EXECVE msg=audit(1765969282.460:2423): argc=4 a0="tail" a1="-c" a2="+10418437" a3="/var/www/nextcloud/data/nextcloud.log"'
if bool(tail_f_re.search(ln)):  # True
    print("Success!")
else:
    print("Failure!")