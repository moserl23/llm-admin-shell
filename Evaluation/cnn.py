# --------------------------------------------------------
# 0. IMPORTS
# --------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("Reached after imports!")

# --------------------------------------------------------
# 1. LOG DATA
# --------------------------------------------------------
human_logs = [
    'type=SYSCALL msg=audit(1759405349.092:4902): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897330 a1=64b2698972a8 a2=64b2698972c8 a3=64b269897840 items=2 ppid=1900 pid=1925 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="head" exe="/usr/bin/head" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=EXECVE msg=audit(1759405349.092:4902): argc=3 a0="head" a1="-n" a2="10"',
    'type=PATH msg=audit(1759405349.092:4902): item=0 name="/usr/bin/head" inode=1442464 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PATH msg=audit(1759405349.092:4902): item=1 name="/lib64/ld-linux-x86-64.so.2" inode=1483960 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PROCTITLE msg=audit(1759405349.092:4902): proctitle=68656164002D6E003130',
    'type=SYSCALL msg=audit(1759405349.094:4903): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897350 a1=64b2698972c8 a2=64b2698972e8 a3=64b269897 items=2 ppid=1900 pid=1926 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="tr" exe="/usr/bin/tr" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=EXECVE msg=audit(1759405349.094:4903): argc=3 a0="tr" a1="-d" a2="\000-\011\013\014\016-\037"',
    'type=PATH msg=audit(1759405349.094:4903): item=0 name="/usr/bin/tr" inode=1442926 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PATH msg=audit(1759405349.094:4903): item=1 name="/lib64/ld-linux-x86-64.so.2" inode=1483960 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PROCTITLE msg=audit(1759405349.094:4903): proctitle=7472002D64005C3030302D5C3031315C3031335C3031345C3031362D5C303337',
    'type=SYSCALL msg=audit(1759405349.094:4904): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897330 a1=64b2698972a8 a2=64b2698972c8 a3=7 items=2 ppid=1900 pid=1927 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="cut" exe="/usr/bin/cut" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=EXECVE msg=audit(1759405349.094:4904): argc=3 a0="cut" a1="-c" a2="-80"',
    'type=PATH msg=audit(1759405349.094:4904): item=0 name="/usr/bin/cut" inode=1442343 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PATH msg=audit(1759405349.094:4904): item=1 name="/lib64/ld-linux-x86-64.so.2" inode=1483960 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PROCTITLE msg=audit(1759405349.094:4904): proctitle=637574002D63002D3830',
    'type=SYSCALL msg=audit(1759405349.103:4905): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897378 a1=64b2698972f0 a2=64b269897310 a3=7 items=2 ppid=1900 pid=1931 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="cut" exe="/usr/bin/cut" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SYSCALL msg=audit(1759405349.103:4906): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897368 a1=64b2698972e8 a2=64b269897300 a3=75e39e7a7ce0 items=2 ppid=1900 pid=1928 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="cat" exe="/usr/bin/cat" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=EXECVE msg=audit(1759405349.103:4906): argc=2 a0="cat" a1="/tmp/tmp.zF0miBsjhK"',
    'type=PATH msg=audit(1759405349.103:4906): item=0 name="/usr/bin/cat" inode=1442301 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PATH msg=audit(1759405349.103:4906): item=1 name="/lib64/ld-linux-x86-64.so.2" inode=1483960 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',

]

ai_logs = [
    'type=SYSCALL msg=audit(1759405347.496:4896): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=7ffef2765700 a2=10 a3=0 items=0 ppid=1900 pid=1920 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="wget" exe="/usr/bin/wget" key="T1043_Commonly_Used_Port"ARCH=x86_64 SYSCALL=connect AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SOCKADDR msg=audit(1759405347.496:4896): saddr=00000000000000000000000000000000SADDR=unknown-family(0)',
    'type=PROCTITLE msg=audit(1759405347.496:4896): proctitle=77676574002D2D74696D656F7574003630002D5500776765742F312E32312E342D317562756E7475342E31205562756E74752F32342E30342E322F4C545320474E552F4C696E75782F362E382E302D38342D67656E657269632F7838365F363420496E74656C2852292F436F726528544D292F69372D3130363130552F435055',
    'type=SYSCALL msg=audit(1759405347.496:4897): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=5f052640ddc0 a2=1c a3=0 items=0 ppid=1900 pid=1920 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="wget" exe="/usr/bin/wget" key="T1043_Commonly_Used_Port"ARCH=x86_64 SYSCALL=connect AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SOCKADDR msg=audit(1759405347.496:4897): saddr=0A000000000000002A05D018091C3200C8872F22290F0A7C00000000SADDR={ saddr_fam=inet6 laddr=2a05:d018:91c:3200:c887:2f22:290f:a7c lport=0 }',
    'type=PROCTITLE msg=audit(1759405347.496:4897): proctitle=77676574002D2D74696D656F7574003630002D5500776765742F312E32312E342D317562756E7475342E31205562756E74752F32342E30342E322F4C545320474E552F4C696E75782F362E382E302D38342D67656E657269632F7838365F363420496E74656C2852292F436F726528544D292F69372D3130363130552F435055',
    'type=SYSCALL msg=audit(1759405347.496:4898): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=7ffef2765700 a2=10 a3=0 items=0 ppid=1900 pid=1920 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="wget" exe="/usr/bin/wget" key="T1043_Commonly_Used_Port"ARCH=x86_64 SYSCALL=connect AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SOCKADDR msg=audit(1759405347.496:4898): saddr=00000000000000000000000000000000SADDR=unknown-family(0)',
    'type=PROCTITLE msg=audit(1759405347.496:4898): proctitle=77676574002D2D74696D656F7574003630002D5500776765742F312E32312E342D317562756E7475342E31205562756E74752F32342E30342E322F4C545320474E552F4C696E75782F362E382E302D38342D67656E657269632F7838365F363420496E74656C2852292F436F726528544D292F69372D3130363130552F435055',
    'type=SYSCALL msg=audit(1759405347.496:4899): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=5f052640abb0 a2=1c a3=0 items=0 ppid=1900 pid=1920 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="wget" exe="/usr/bin/wget" key="T1043_Commonly_Used_Port"ARCH=x86_64 SYSCALL=connect AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SOCKADDR msg=audit(1759405347.496:4899): saddr=0A000000000000002A05D018091C32005E0D21A926CA90B500000000SADDR={ saddr_fam=inet6 laddr=2a05:d018:91c:3200:5e0d:21a9:26ca:90b5 lport=0 }',
    'type=PROCTITLE msg=audit(1759405347.496:4899): proctitle=77676574002D2D74696D656F7574003630002D5500776765742F312E32312E342D317562756E7475342E31205562756E74752F32342E30342E322F4C545320474E552F4C696E75782F362E382E302D38342D67656E657269632F7838365F363420496E74656C2852292F436F726528544D292F69372D3130363130552F435055',
    'type=SYSCALL msg=audit(1759405347.496:4900): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=7ffef2766160 a2=10 a3=8 items=0 ppid=1900 pid=1920 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="wget" exe="/usr/bin/wget" key="T1043_Commonly_Used_Port"ARCH=x86_64 SYSCALL=connect AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=SOCKADDR msg=audit(1759405347.496:4900): saddr=020001BB36ABE6370000000000000000SADDR={ saddr_fam=inet laddr=54.171.230.55 lport=443 }',
    'type=PROCTITLE msg=audit(1759405347.496:4900): proctitle=77676574002D2D74696D656F7574003630002D5500776765742F312E32312E342D317562756E7475342E31205562756E74752F32342E30342E322F4C545320474E552F4C696E75782F362E382E302D38342D67656E657269632F7838365F363420496E74656C2852292F436F726528544D292F69372D3130363130552F435055',
    'type=SYSCALL msg=audit(1759405349.090:4901): arch=c000003e syscall=59 success=yes exit=0 a0=64b269897320 a1=64b2698972a0 a2=64b2698972b8 a3=75e39e7a7b40 items=2 ppid=1900 pid=1924 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 sgid=0 fsgid=0 tty=(none) ses=4294967295 comm="cat" exe="/usr/bin/cat" key="T1078_Valid_Accounts"ARCH=x86_64 SYSCALL=execve AUID="unset" UID="root" GID="root" EUID="root" SUID="root" FSUID="root" EGID="root" SGID="root" FSGID="root"',
    'type=EXECVE msg=audit(1759405349.090:4901): argc=2 a0="cat" a1="/tmp/tmp.zF0miBsjhK"',
    'type=PATH msg=audit(1759405349.090:4901): item=0 name="/usr/bin/cat" inode=1442301 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'type=PATH msg=audit(1759405349.090:4901): item=1 name="/lib64/ld-linux-x86-64.so.2" inode=1483960 dev=08:02 mode=0100755 ouid=0 ogid=0 rdev=00:00 nametype=NORMAL cap_fp=0 cap_fi=0 cap_fe=0 cap_fver=0 cap_frootid=0OUID="root" OGID="root"',
    'vtype=PROCTITLE msg=audit(1759405349.090:4901): proctitle=636174002F746D702F746D702E7A46306D6942736A684B',

]

# --------------------------------------------------------
# 2. Combine logs + labels
# --------------------------------------------------------
logs = human_logs + ai_logs
labels = np.array([0]*len(human_logs) + [1]*len(ai_logs), dtype=np.float32)

# --------------------------------------------------------
# 3. Character vocabulary
# --------------------------------------------------------
def build_char_vocab(texts):
    chars = sorted({ch for t in texts for ch in t})
    return {ch: i+1 for i, ch in enumerate(chars)}  # 0 = padding

char2idx = build_char_vocab(logs)
vocab_size = len(char2idx) + 1

# --------------------------------------------------------
# 4. Encode + pad
# --------------------------------------------------------
def encode(text, mapping):
    return [mapping.get(ch, 0) for ch in text]

encoded = [encode(t, char2idx) for t in logs]

max_len = int(min(np.percentile([len(x) for x in encoded], 95), 512))

def pad(seq, max_len):
    return seq[:max_len] + [0]*(max_len - len(seq))

X = np.array([pad(s, max_len) for s in encoded], dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, stratify=labels
)

# --------------------------------------------------------
# 5. Dataset Class
# --------------------------------------------------------
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(LogDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader  = DataLoader(LogDataset(X_test,  y_test),  batch_size=8)

# --------------------------------------------------------
# 6. Multi-Kernel CharCNN
# --------------------------------------------------------
class MultiKernelCharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Multi-scale convolution kernels
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5)
        self.conv7 = nn.Conv1d(embed_dim, num_filters, kernel_size=7)

        self.pool = nn.AdaptiveMaxPool1d(1)

        # Final classifier
        self.fc1 = nn.Linear(num_filters * 3, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)        # (B, L, E)
        x = x.permute(0, 2, 1)       # (B, E, L)

        x3 = torch.relu(self.conv3(x))
        x5 = torch.relu(self.conv5(x))
        x7 = torch.relu(self.conv7(x))

        x3 = self.pool(x3).squeeze(-1)
        x5 = self.pool(x5).squeeze(-1)
        x7 = self.pool(x7).squeeze(-1)

        x = torch.cat([x3, x5, x7], dim=1)

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x  # logits


# --------------------------------------------------------
# 7. Training Setup
# --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiKernelCharCNN(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# --------------------------------------------------------
# 8. Training Loop
# --------------------------------------------------------
for epoch in range(20):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# --------------------------------------------------------
# 9. Evaluation
# --------------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += len(yb)

print("\nTest Accuracy:", correct / total)

# --------------------------------------------------------
# 10. Predict New Log
# --------------------------------------------------------
new_log = 'type=SYSCALL msg=audit(1763731451.291:55974): arch=c000003e syscall=59 success=no a0=70cdaa4d1d80 a1=70cda468a438 comm="php" exe="/usr/bin/php8.3"'

enc = encode(new_log, char2idx)
pad_enc = pad(enc, max_len)

with torch.no_grad():
    tensor = torch.tensor([pad_enc], dtype=torch.long).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

print("\nProbability AI:", prob)
print("Predicted Class:", "AI" if prob >= 0.5 else "Human")
