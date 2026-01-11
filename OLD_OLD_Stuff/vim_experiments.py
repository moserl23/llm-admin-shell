from utils import *


################################################ main ################################################
if __name__ == "__main__":
    session = ShellSession()
    session.connect_root_setSentinel()
   
    # Set enrionment variables and start logging point
    init_env_and_log_offsets(session)

    # Start vim session for a selected file
    session.start_vim("/root/my_config.toml")

    #content = session.print_file_vim()
    #print(content)

    pattern = r'\v^\s*request_body_mb\s*=\s*[0-9]+\s*$'

    grep_result = session.grep_vim(pattern=pattern)
    print("grep_result:")
    print(grep_result)

    # close the vim session
    session.end_vim()

    # Extract logs
    read_new_logs(session)
    session.close()