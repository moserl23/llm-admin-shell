import subprocess
import json
from pathlib import Path

def evaluate_combo_detector(train_log_sample: list[str], test_log_sample: list[str]) -> dict:

    LOG_PATH = "/home/lorenz/Documents/DetectMate/Logs_IDS_audit.log"
    # clear this file and then write the lines train_log_sample there and concatenate test_log_sample
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for line in train_log_sample:
            f.write(line.rstrip("\n") + "\n")

        for line in test_log_sample:
            f.write(line.rstrip("\n") + "\n")

    #set train/test correctly
    train_size = len(train_log_sample)
    test_size = len(test_log_sample)

    DETECTMATE_DIR = Path("/home/lorenz/Documents/DetectMate/DetectMateLibrary")

    cmd = [
        "uv",
        "run",
        "python",
        "MA_lorenz.py",
        "--train-size", str(train_size),
        "--test-size", str(test_size),
    ]

    proc = subprocess.run(
        cmd,
        cwd=DETECTMATE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )


    result = json.loads(proc.stdout.strip())
    print("Result:", result)
    return result


