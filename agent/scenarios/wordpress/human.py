"""Manual WordPress scenario runner.

This helper opens a root shell, snapshots log offsets, waits for a human to
perform the scenario, and then collects only the newly generated logs.
"""

from ...utils import ShellSession, init_env_and_log_offsets, read_new_logs




# ---- Manual run ----
if __name__ == "__main__":


    # Prepare the shell state and log offsets before the manual intervention.
    session = ShellSession()
    try:
        session.connect_root_setSentinel()
        init_env_and_log_offsets(session)
        print(">> Ready. Perform your manual steps now. Press Enter when done (Ctrl+C to abort).")

        # Pause until the human has completed the scenario steps.
        input("Human has finished?")

        # Persist only the log tail generated during the manual session.
        read_new_logs(session)
        print(">> Logs extracted to LOGS/.")
    finally:
        session.close()
