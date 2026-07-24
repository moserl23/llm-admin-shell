from __future__ import annotations

"""Runner for one-gram distance experiments in single and sweep mode.

This script builds experiment configurations, executes the one-gram statistic,
and forwards results to the shared evaluation pipeline under true or null-style
label assignments.
"""

import argparse

from src.stats_tools import one_gram
from src.core.stats.common_runner import evaluate_single_run


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments for one-gram experiments.

    The validation logic enforces mode-specific requirements and prevents
    dataset/log-type combinations that are not supported. Returns a validated
    argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run one_gram in single or sweep mode."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "sweep"],
        required=True,
        help="Whether to run a single configuration or a sweep.",
    )
    parser.add_argument(
        "--dataset",
        choices=["Nextcloud", "WordPress"],
        required=True,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--assignment_mode",
        choices=["true", "random_stratified", "indexed_stratified"],
        required=True,
        help="Assignment mode for group labels.",
    )
    parser.add_argument(
        "--assignment_idx",
        type=int,
        default=None,
        help="Assignment index, required when assignment_mode=indexed_stratified.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help=(
            "Optional output CSV path. "
            "If omitted, a dataset-specific default path is used."
        ),
    )

    parser.add_argument(
        "--log_type",
        choices=["syslog", "nextcloud", "audit"],
        help="Log type for single mode.",
    )

    # ---- Single-run configuration ----
    parser.add_argument(
        "--ngram_mode",
        choices=["char", "word"],
        help="One-gram mode for single mode.",
    )
    parser.add_argument(
        "--metric",
        choices=["js", "l1"],
        help="Distance metric for single mode.",
    )

    args = parser.parse_args()

    # Indexed stratification represents a specific precomputed label assignment,
    # so the index is required there and meaningless for other modes.
    if args.assignment_mode == "indexed_stratified" and args.assignment_idx is None:
        parser.error(
            "--assignment_idx is required when --assignment_mode indexed_stratified"
        )

    if (
        args.assignment_mode != "indexed_stratified"
        and args.assignment_idx is not None
    ):
        parser.error(
            "--assignment_idx may only be used when --assignment_mode indexed_stratified"
        )

    # Single mode executes one concrete configuration and therefore requires
    # all fields needed to build that configuration explicitly.
    if args.mode == "single":
        missing = []
        if args.log_type is None:
            missing.append("--log_type")
        if args.ngram_mode is None:
            missing.append("--ngram_mode")
        if args.metric is None:
            missing.append("--metric")

        if args.dataset == "WordPress" and args.log_type == "nextcloud":
            parser.error(
                "--log_type nextcloud is not valid for dataset WordPress"
            )

        if missing:
            parser.error(
                f"In single mode the following arguments are required: {', '.join(missing)}"
            )

    if args.mode == "sweep":
        if args.dataset == "WordPress" and args.log_type == "nextcloud":
            parser.error(
                "--log_type nextcloud is not valid for dataset WordPress"
            )

    return args


def run_one_gram(
    mode: str,
    dataset: str,
    assignment_mode: str,
    assignment_idx: int | None = None,
    log_type: str | None = None,
    ngram_mode: str | None = None,
    metric: str | None = None,
    out_csv: str | None = None,
) -> None:
    """Run one-gram analysis for one configuration or a predefined sweep.

    In single mode, the function evaluates one explicit setup and generates
    plots. In sweep mode, it iterates over a fixed grid of configurations and
    appends comparable evaluation results to the output CSV.
    """
    # Keep dataset-specific outputs separate so repeated sweeps append to the
    # appropriate results table by default.
    output_path = out_csv or (
        f"results/statistic_one_gram"
        f"{'_wordpress' if dataset == 'WordPress' else '_nextcloud'}.csv"
    )

    if mode == "single":
        # ---- Single configuration ----
        config = {
            "dataset": dataset,
            "log_type": log_type,
            "mode": ngram_mode,
            "metric": metric,
            "min_count": 1,
        }

        result = one_gram.run(config)

        evaluate_single_run(
            tool_name="one_gram",
            labels=result["labels"],
            pairwise_results=result["pairwise_results"],
            distance_name=config["metric"],
            distance_extractor=lambda item: item[config["metric"]],
            hyperparameters=config,
            assignment_mode=assignment_mode,
            assignment_idx=assignment_idx,
            plot=True,
            output_path=output_path,
        )

    elif mode == "sweep":
        # ---- Parameter sweep ----
        if log_type is not None:
            log_types = [log_type]

        modes = ["word", "char"]
        metrics = ["l1", "js"]

        configs = one_gram.build_sweep_configs(
            dataset=dataset,
            log_types=log_types,
            modes=modes,
            metrics=metrics,
            min_count=1,
        )

        print(
            f"\nRunning one_gram sweep with: "
            f"dataset={dataset}, "
            f"assignment_mode={assignment_mode}, "
            f"assignment_idx={assignment_idx}, "
            f"output_path={output_path}"
        )

        # Each sweep entry is evaluated independently so the shared runner can
        # aggregate comparable results across metrics and representation modes.
        for i, config in enumerate(configs, 1):
            print(
                f"\n[{i}/{len(configs)}] "
                f"log_type={config['log_type']} "
                f"mode={config['mode']} "
                f"metric={config['metric']}"
            )

            result = one_gram.run(config)

            evaluate_single_run(
                tool_name="one_gram",
                labels=result["labels"],
                pairwise_results=result["pairwise_results"],
                distance_name=config["metric"],
                # Bind the metric name at definition time so each sweep result
                # is evaluated against the intended distance column.
                distance_extractor=lambda item, metric_name=config["metric"]: item[metric_name],
                hyperparameters={
                    "log_type": config["log_type"],
                    "mode": config["mode"],
                    "metric": config["metric"],
                    "min_count": config["min_count"],
                },
                assignment_mode=assignment_mode,
                assignment_idx=assignment_idx,
                output_path=output_path,
                plot=False,
            )

    else:
        raise ValueError("mode must be 'single' or 'sweep'")


if __name__ == "__main__":
    args = parse_args()

    run_one_gram(
        mode=args.mode,
        dataset=args.dataset,
        assignment_mode=args.assignment_mode,
        assignment_idx=args.assignment_idx,
        log_type=args.log_type,
        ngram_mode=args.ngram_mode,
        metric=args.metric,
        out_csv=args.out_csv,
    )
