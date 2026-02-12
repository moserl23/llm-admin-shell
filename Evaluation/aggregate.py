import all_file_paths
from pathlib import Path
import numpy as np

### Global ###
print_header_flag = False


def aggregate_over_experiments():

    # Output base directory
    out_base = Path(
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated"
    )

    # Iterate over persons
    for person in all_file_paths.names:
        person_out_dir = out_base / person
        person_out_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over log types (audit, syslog, nextcloud, auth)
        for log_type in all_file_paths.log_types.keys():
            out_file = person_out_dir / f"{log_type}.log"

            with out_file.open("wb") as out:  # binary = safest for logs
                for exp in all_file_paths.exp_range:
                    src = all_file_paths.files["singular"][person][exp][log_type]

                    if not src.exists():
                        print(f"[WARN] Missing: {src}")
                        continue

                    # Optional separator (comment out if you want pure concat)
                    if print_header_flag:
                        header = (
                            f"\n\n===== {person} | {log_type}.log | exp{exp} =====\n\n"
                        )
                        out.write(header.encode("utf-8"))

                    with src.open("rb") as f:
                        last_byte = None
                        while True:
                            chunk = f.read(1024*1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            last_byte = chunk[-1]
                        if last_byte is not None and last_byte != ord(b"\n"):
                            out.write(b"\n")



            print(f"[OK] Aggregated {out_file}")



def aggregate_over_person():
    out_base = Path(
        "/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/PersonAggregated"
    )

    for log_type in all_file_paths.log_types.keys():
        for exp in all_file_paths.exp_range:
            # Build output paths correctly
            out_file_human = out_base / f"Exp{exp}" / "Human" / f"{log_type}.log"
            out_file_ai    = out_base / f"Exp{exp}" / "AI"    / f"{log_type}.log"

            # Ensure directories exist
            out_file_human.parent.mkdir(parents=True, exist_ok=True)
            out_file_ai.parent.mkdir(parents=True, exist_ok=True)

            with out_file_human.open("wb") as out_human, out_file_ai.open("wb") as out_ai:
                for person in all_file_paths.names:
                    src = all_file_paths.files["singular"][person][exp][log_type]

                    if not src.exists():
                        print(f"[WARN] Missing: {src}")
                        continue

                    # Optional separator (comment out if you want pure concat)
                    if print_header_flag:
                        header = f"\n\n===== {person} | exp{exp} | {log_type}.log =====\n\n".encode("utf-8")

                    if "GPT" in person:
                        if print_header_flag:
                            out_ai.write(header)

                        with src.open("rb") as f:
                            last_byte = None
                            while True:
                                chunk = f.read(1024*1024)
                                if not chunk:
                                    break
                                out_ai.write(chunk)
                                last_byte = chunk[-1]
                            if last_byte is not None and last_byte != ord(b"\n"):
                                out_ai.write(b"\n")

                    else:
                        if print_header_flag:
                            out_human.write(header)

                        with src.open("rb") as f:
                            last_byte = None
                            while True:
                                chunk = f.read(1024*1024)
                                if not chunk:
                                    break
                                out_human.write(chunk)
                                last_byte = chunk[-1]
                            if last_byte is not None and last_byte != ord(b"\n"):
                                out_human.write(b"\n")

            print(f"[OK] Wrote exp{exp} {log_type}: Human + AI")                 




def aggregate_all():
    src_base = Path("/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated")
    out_base = Path("/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/AllAggregated")

    human_dir = out_base / "Human"
    ai_dir = out_base / "AI"
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    for log_type in all_file_paths.log_types.keys():
        out_human_file = human_dir / f"{log_type}.log"
        out_ai_file = ai_dir / f"{log_type}.log"

        with out_human_file.open("wb") as out_human, out_ai_file.open("wb") as out_ai:
            for person in all_file_paths.names:
                src = src_base / person / f"{log_type}.log"

                if not src.exists():
                    print(f"[WARN] Missing: {src}")
                    continue

                # Optional separator
                if print_header_flag:
                    header = f"\n\n===== {person} | {log_type}.log =====\n\n".encode("utf-8")

                if "GPT" in person:
                    if print_header_flag:
                        out_ai.write(header)

                    with src.open("rb") as f:
                        last_byte = None
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            out_ai.write(chunk)
                            last_byte = chunk[-1]
                        if last_byte is not None and last_byte != ord(b"\n"):
                            out_ai.write(b"\n")

                else:
                    if print_header_flag:
                        out_human.write(header)

                    with src.open("rb") as f:
                        last_byte = None
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            out_human.write(chunk)
                            last_byte = chunk[-1]
                        if last_byte is not None and last_byte != ord(b"\n"):
                            out_human.write(b"\n")


        print(f"[OK] Wrote AllAggregated Human/AI for {log_type}")
                    


if __name__ == "__main__":
    aggregate_over_experiments()

    aggregate_over_person()

    aggregate_all()
