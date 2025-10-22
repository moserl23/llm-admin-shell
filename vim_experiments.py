from utils import *



################################################ main ################################################
if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
    init_env_and_log_offsets(session)

    session.run_cmd("cd /home/vboxuser")

    session.start_vim("myFile.txt")

    content_file = session.print_file_vim()
    print("Previous-Version:", content_file)

    session.edit_file_vim([":%s/^option_c = .*/option_c = AI-Wrote-Here/\r"])

    content_file = session.print_file_vim()
    print("Updated-Version:", content_file)

    session.end_vim()

    read_new_logs(session)
    session.close()