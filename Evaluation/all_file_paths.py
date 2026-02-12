from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

names = [
    "Armin", "Benni", "Hotti", "Marvin", "Nico", "Torina",
    "GPT4.1", "GPT4.1_V2", "GPT4o", "GPT5"
]

human_names = {"Armin", "Benni", "Hotti", "Marvin", "Nico", "Torina"}
ai_names = set(names) - human_names

log_types = {
    "audit": "audit.log",
    "syslog": "syslog.log",
    "nextcloud": "nextcloud.log",
    "auth": "auth.log",
}


def calc_indices():
    """
    Returns:
      result[log_type][name] = (start, end)
    where indices refer to positions in:
      X_text = concatenated logs in the order of all_file_paths.names
    and end is EXCLUSIVE.
    """

    result: dict[str, dict[str, tuple[int, int]]] = {}

    BASE = Path("/home/lorenz/Documents/llm-admin-shell/ExperimentResult/combine/ExperimentAggregated")

    def count_log_lines(name: str, log_type: str) -> int:
        path = BASE / name / f"{log_type}.log"
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)

    for log_type in log_types.keys():
        result[log_type] = {}
        cumulative = 0

        for name in names:
            file_length = count_log_lines(name=name, log_type=log_type)

            start = cumulative
            end   = cumulative + file_length

            result[log_type][name] = (start, end)

            cumulative = end

    return result




base_path = Path("/home/lorenz/Documents/llm-admin-shell/ExperimentResult")

EXP_FIRST = 1
EXP_LAST_INCLUSIVE = 7
exp_range = range(EXP_FIRST, EXP_LAST_INCLUSIVE + 1)

# -----------------------------
# Build files registry
# -----------------------------
files = {

    # ---------------------------------
    # 1) Singular (per person, per exp)
    # ---------------------------------
    "singular": {
        name: {
            exp: {
                log_type: (
                    base_path
                    / f"LOGS_Result_{name}"
                    / f"exp{exp}"
                    / filename
                )
                for log_type, filename in log_types.items()
            }
            for exp in exp_range
        }
        for name in names
    },

    # ---------------------------------
    # 2) Person-aggregated (per exp, Human vs AI)
    # ---------------------------------
    "personAgg": {
        exp: {
            "Human": {
                log_type: (
                    base_path
                    / "combine"
                    / "PersonAggregated"
                    / f"Exp{exp}"
                    / "Human"
                    / filename
                )
                for log_type, filename in log_types.items()
            },
            "AI": {
                log_type: (
                    base_path
                    / "combine"
                    / "PersonAggregated"
                    / f"Exp{exp}"
                    / "AI"
                    / filename
                )
                for log_type, filename in log_types.items()
            },
        }
        for exp in exp_range
    },

    # ---------------------------------
    # 3) Experiment-aggregated (per person, across all exps)
    # ---------------------------------
    "experimentAgg": {
        name: {
            log_type: (
                base_path
                / "combine"
                / "ExperimentAggregated"
                / name
                / filename
            )
            for log_type, filename in log_types.items()
        }
        for name in names
    },
    # ---------------------------------
    # 4) Total aggregated (all people + all exps)
    # ---------------------------------
    "totalAgg": {
        group: {
            log_type: (
                base_path
                / "combine"
                / "AllAggregated"
                / group
                / filename
            )
            for log_type, filename in log_types.items()
        }
        for group in ("Human", "AI")
    },
}


if __name__ == "__main__":
    

    print(files["totalAgg"]["AI"]["syslog"])

