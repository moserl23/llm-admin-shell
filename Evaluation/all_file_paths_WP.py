from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

names = [
    "Armin", "Hotti", "Marvin", "Nico",
    "GPT4.1", "GPT4.1_V2", "GPT4.1_V3", "GPT5"
]

human_names = {"Armin", "Hotti", "Marvin", "Nico"}
ai_names = set(names) - human_names

log_types = {
    "audit": "audit.log",
    "syslog": "syslog.log",
    "nextcloud": "nextcloud.log",
    "auth": "auth.log",
}




base_path = Path("/home/lorenz/Documents/llm-admin-shell/ExperimentResult_WP")

EXP_FIRST = 1
EXP_LAST_INCLUSIVE = 5
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

