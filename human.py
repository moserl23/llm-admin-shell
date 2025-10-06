from utils import ShellSession, init_env_and_log_offsets, read_new_logs




################################################ main ################################################
if __name__ == "__main__":


    # connect server, login-shell as root and configure sentinel, configure log-offset and environment-variable
    session = ShellSession()
    try:
        session.connect_root_setSentinel()
        init_env_and_log_offsets(session)
        print(">> Ready. Perform your manual steps now. Press Enter when done (Ctrl+C to abort).")

        # wait for the human to finish
        input("human is finished?")

        # extract new logs and write to file
        read_new_logs(session)
        print(">> Logs extracted to LOGS/.")
    finally:
        session.close()

