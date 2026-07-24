"""Run complexity-based sequence metrics under true or null-style assignments.

This runner supports a single configuration or a sweep over window/stride settings
and forwards each result to the shared evaluation pipeline.
"""

from __future__ import annotations

import argparse

from src.stats_tools import complexity_metrics
from src.core.stats.common_runner import evaluate_single_run


CLI_TO_METRIC_KEY = {
    "gini": "gini_seq",
    "kurtosis": "kurtosis_seq",
    "mad": "mad_seq",
    "entropy": "entropy_seq",
}


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments for single-run and sweep execution.

    Enforces mode-specific requirements and dataset/log-type compatibility.
    Returns the validated namespace used to launch the runner.
    """
    parser = argparse.ArgumentParser(
        description="Run complexity metrics in single or sweep mode."
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
        help="Log type for single mode, or optional restriction for sweep mode.",
    )

    # single mode arguments
    parser.add_argument(
        "--window_size",
        type=int,
        help="Window size for single mode.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Stride for single mode.",
    )
    parser.add_argument(
        "--metric",
        choices=["gini", "kurtosis", "mad", "entropy"],
        help="Metric for single mode.",
    )

    args = parser.parse_args()

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

    if args.mode == "single":
        missing = []
        if args.log_type is None:
            missing.append("--log_type")
        if args.window_size is None:
            missing.append("--window_size")
        if args.stride is None:
            missing.append("--stride")
        if args.metric is None:
            missing.append("--metric")

        if missing:
            parser.error(
                f"In single mode the following arguments are required: {', '.join(missing)}"
            )

    if args.dataset == "WordPress" and args.log_type == "nextcloud":
        parser.error("--log_type nextcloud is not valid for dataset WordPress")

    return args


def run_complexity_metrics(
    mode: str,
    dataset: str,
    assignment_mode: str,
    assignment_idx: int | None = None,
    log_type: str | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    metric_name: str | None = None,
    out_csv: str | None = None,
) -> None:
    """Execute one complexity-metric run or a sweep of predefined configurations.

    Single mode evaluates one selected metric, while sweep mode iterates over a
    fixed set of temporal resolutions and all available metric variants.
    """
    # Keep dataset-specific outputs separate so sweeps append to the expected table.
    output_path = out_csv or (
        f"results/statistic_complexity_metrics"
        f"{'_wordpress' if dataset == 'WordPress' else '_nextcloud'}.csv"
    )

    if mode == "single":
        # ---- Single configuration ----
        config = {
            "dataset": dataset,
            "log_type": log_type,
            "window_size": window_size,
            "stride": stride,
            "preprocess_mode": "soft",
            "drain_ini_path": None,
        }

        result = complexity_metrics.run(config)
        pairwise_results = result["pairwise_results"]
        labels = result["labels"]

        # Non-finite distances make the downstream test invalid for the affected metric.
        invalid_labels = complexity_metrics.get_invalid_labels_for_metric(
            pairwise_results,
            metric_name,
        )
        if invalid_labels:
            print(
                f"Cannot evaluate metric={metric_name} in single mode "
                f"(non-finite distances; affected labels={invalid_labels})"
            )
            return

        evaluate_single_run(
            tool_name="complexity_metrics",
            labels=labels,
            pairwise_results=pairwise_results,
            distance_name=metric_name,
            distance_extractor=lambda item: item["distances"][metric_name],
            hyperparameters={
                "log_type": config["log_type"],
                "window_size": config["window_size"],
                "stride": config["stride"],
                "preprocess_mode": config["preprocess_mode"],
            },
            assignment_mode=assignment_mode,
            assignment_idx=assignment_idx,
            output_path=output_path,
            plot=True,
        )

    elif mode == "sweep":
        # ---- Sweep over temporal resolutions ----
        if log_type is not None:
            log_types = [log_type]

        # These settings probe local to longer-range sequence structure without
        # exploding the grid size.
        window_stride_pairs = [
            (1, 1),   # unigram-like
            (3, 1),   # short local patterns
            (5, 2),   # slightly smoother local structure
            (10, 2),  # medium-scale behavior
            (25, 5),  # long-range structure
        ]

        configs = [
            {
                "dataset": dataset,
                "log_type": log_type,
                "window_size": window_size,
                "stride": stride,
                "preprocess_mode": "soft",
                "drain_ini_path": None,
            }
            for log_type in log_types
            for window_size, stride in window_stride_pairs
        ]

        print(
            f"\nRunning complexity_metrics sweep with: "
            f"dataset={dataset}, "
            f"assignment_mode={assignment_mode}, "
            f"assignment_idx={assignment_idx}, "
            f"output_path={output_path}"
        )

        for i, config in enumerate(configs, 1):
            print(
                f"\n[CONFIG {i}/{len(configs)}] "
                f"log_type={config['log_type']} "
                f"window_size={config['window_size']} "
                f"stride={config['stride']}"
            )

            result = complexity_metrics.run(config)
            pairwise_results = result["pairwise_results"]
            labels = result["labels"]

            # Reuse the same pairwise distances across all metric-specific evaluations.
            for metric_name in complexity_metrics.METRIC_KEYS:
                print(f"\n--- Evaluating metric={metric_name} ---")

                invalid_labels = complexity_metrics.get_invalid_labels_for_metric(
                    pairwise_results,
                    metric_name,
                )
                if invalid_labels:
                    print(
                        f"skipped metric={metric_name} for "
                        f"log_type={config['log_type']}, "
                        f"window_size={config['window_size']}, "
                        f"stride={config['stride']} "
                        f"(non-finite distances; affected labels={invalid_labels})"
                    )
                    continue

                evaluate_single_run(
                    tool_name="complexity_metrics",
                    labels=labels,
                    pairwise_results=pairwise_results,
                    distance_name=metric_name,
                    distance_extractor=lambda item, metric_name=metric_name: item["distances"][metric_name],
                    hyperparameters={
                        "log_type": config["log_type"],
                        "window_size": config["window_size"],
                        "stride": config["stride"],
                        "preprocess_mode": config["preprocess_mode"],
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

    metric_name = None
    if args.metric is not None:
        metric_name = CLI_TO_METRIC_KEY[args.metric]

    run_complexity_metrics(
        mode=args.mode,
        dataset=args.dataset,
        assignment_mode=args.assignment_mode,
        assignment_idx=args.assignment_idx,
        log_type=args.log_type,
        window_size=args.window_size,
        stride=args.stride,
        metric_name=metric_name,
        out_csv=args.out_csv,
    )
