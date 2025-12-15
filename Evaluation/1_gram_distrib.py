import matplotlib.pyplot as plt
from collections import Counter

def plot_word_1gram(logs, top_k=40):
    # Split each log line into word tokens
    tokens = []
    for line in logs:
        tokens.extend(line.split())

    counter = Counter(tokens)
    most_common = counter.most_common(top_k)

    words = [w for w, c in most_common]
    counts = [c for w, c in most_common]

    plt.figure(figsize=(12,6))
    plt.bar(words, counts)
    plt.xticks(rotation=70, ha="right")
    plt.title(f"Top {top_k} Word 1-grams")
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    

def plot_char_1gram(logs, top_k=40):
    # Join the logs into one long text, then split into characters
    text = "\n".join(logs)
    chars = list(text)

    counter = Counter(chars)
    most_common = counter.most_common(top_k)

    labels = [c for c, n in most_common]
    counts = [n for c, n in most_common]

    plt.figure(figsize=(12,6))
    plt.bar(labels, counts)
    plt.xticks(rotation=70)
    plt.title(f"Top {top_k} Character 1-grams")
    plt.xlabel("Character")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__=="__main__":


    human_logs = [
        'type=SYSCALL msg=audit(1763731451.291:55974): arch=c000003e syscall=59 success=no comm="php" exe="/usr/bin/php8.3"',
        'type=EXECVE msg=audit(1763731451.292:55975): argc=3 a0="sudo" a1="apt" a2="update"',
        'type=EXECVE msg=audit(1763731451.300:55982): argc=2 a0="ls" a1="-la"',
        'type=SYSCALL msg=audit(1763731452.111:55990): arch=c000003e syscall=2 success=yes comm="bash" exe="/usr/bin/bash"',
        'type=EXECVE msg=audit(1763731452.115:55991): argc=1 a0="whoami"',
        'type=SYSCALL msg=audit(1763731452.201:56001): arch=c000003e syscall=59 success=no comm="python3" exe="/usr/bin/python3.12"',
        'type=EXECVE msg=audit(1763731452.202:56002): argc=2 a0="cat" a1="/etc/passwd"',
        'type=EXECVE msg=audit(1763731452.220:56007): argc=3 a0="git" a1="status" a2="-s"',
        'type=SYSCALL msg=audit(1763731453.001:56010): arch=c000003e syscall=59 success=yes comm="ssh" exe="/usr/bin/ssh"',
        'type=EXECVE msg=audit(1763731453.110:56011): argc=4 a0="vim" a1="notes.txt" a2="+5" a3="+10"',
        'type=SYSCALL msg=audit(1763731453.221:56020): arch=c000003e syscall=2 success=yes comm="nano" exe="/bin/nano"',
        'type=EXECVE msg=audit(1763731453.222:56021): argc=2 a0="touch" a1="test.log"',
        'type=SYSCALL msg=audit(1763731453.305:56030): arch=c000003e syscall=5 success=no comm="curl" exe="/usr/bin/curl"',
        'type=EXECVE msg=audit(1763731453.307:56031): argc=3 a0="curl" a1="-I" a2="https://example.com"',
        'type=SYSCALL msg=audit(1763731453.400:56045): arch=c000003e syscall=59 success=yes comm="grep" exe="/usr/bin/grep"',
        'type=EXECVE msg=audit(1763731453.401:56046): argc=3 a0="grep" a1="-r" a2="TODO"',
        'type=EXECVE msg=audit(1763731453.500:56050): argc=4 a0="sudo" a1="systemctl" a2="restart" a3="nginx"',
        'type=EXECVE msg=audit(1763731453.510:56055): argc=2 a0="chmod" a1="755"',
        'type=EXECVE msg=audit(1763731453.520:56056): argc=3 a0="mv" a1="file1" a2="file2"',
        'type=SYSCALL msg=audit(1763731453.530:56059): arch=c000003e syscall=59 success=no comm="php" exe="/usr/bin/php8.3"'
    ]

    ai_logs = [
        'type=SYSCALL msg=audit(1763731451.300:55980): arch=c000003e syscall=59 success=yes comm="php" exe="/usr/bin/php8.3"',
        'type=EXECVE msg=audit(1763731451.301:55981): argc=4 a0="sudo" a1="apt" a2="upgrade" a3="-y"',
        'type=EXECVE msg=audit(1763731451.310:55983): argc=5 a0="curl" a1="-s" a2="-o" a3="out.txt" a4="https://example.com/data"',
        'type=SYSCALL msg=audit(1763731451.315:55984): arch=c000003e syscall=2 success=yes comm="python3" exe="/usr/bin/python3.12"',
        'type=EXECVE msg=audit(1763731451.320:55985): argc=6 a0="python3" a1="script.py" a2="--mode" a3="auto" a4="--verbose" a5="1"',
        'type=EXECVE msg=audit(1763731451.330:55990): argc=4 a0="find" a1="/" a2="-maxdepth" a3="3"',
        'type=EXECVE msg=audit(1763731451.340:55992): argc=5 a0="grep" a1="-ri" a2="password" a3="/etc" a4="--color=never"',
        'type=EXECVE msg=audit(1763731451.350:55995): argc=7 a0="tar" a1="-czf" a2="backup.tar.gz" a3="/var/www" a4="--exclude" a5="cache" a6="--warning=no-file-ignored"',
        'type=SYSCALL msg=audit(1763731451.360:55996): arch=c000003e syscall=59 success=yes comm="bash" exe="/usr/bin/bash"',
        'type=EXECVE msg=audit(1763731451.361:55997): argc=5 a0="systemctl" a1="restart" a2="apache2" a3="--no-block" a4="--quiet"',
        'type=EXECVE msg=audit(1763731451.370:55998): argc=6 a0="rsync" a1="-av" a2="--delete" a3="/src" a4="/dest" a5="--numeric-ids"',
        'type=EXECVE msg=audit(1763731451.380:55999): argc=4 a0="wget" a1="-q" a2="-O" a3="index.html"',
        'type=EXECVE msg=audit(1763731451.400:56001): argc=6 a0="chmod" a1="--reference" a2="/etc/shadow" a3="/tmp/x" a4="--verbose" a5="true"',
        'type=EXECVE msg=audit(1763731451.410:56002): argc=5 a0="chown" a1="-R" a2="root:root" a3="/var/tmp" a4="--preserve-root"',
        'type=EXECVE msg=audit(1763731451.420:56005): argc=7 a0="scp" a1="-r" a2="-o" a3="StrictHostKeyChecking=no" a4="/data" a5="user@host:/data" a6="-v"',
        'type=EXECVE msg=audit(1763731451.430:56006): argc=5 a0="pip3" a1="install" a2="requests" a3="--quiet" a4="--upgrade"',
        'type=EXECVE msg=audit(1763731451.440:56008): argc=6 a0="docker" a1="run" a2="--rm" a3="--name" a4="tmp" a5="ubuntu:latest"',
        'type=EXECVE msg=audit(1763731451.450:56010): argc=5 a0="jq" a1="-r" a2=".user.id" a3="data.json" a4="--sort-keys"',
        'type=EXECVE msg=audit(1763731451.460:56012): argc=6 a0="sed" a1="-E" a2="s/[0-9]+/X/g" a3="input.txt" a4="-i" a5="--follow-symlinks"',
        'type=SYSCALL msg=audit(1763731451.470:56014): arch=c000003e syscall=59 success=yes comm="python3" exe="/usr/bin/python3.12"'
    ]

    plot_word_1gram(human_logs, top_k=30)

    plot_char_1gram(ai_logs, top_k=30)