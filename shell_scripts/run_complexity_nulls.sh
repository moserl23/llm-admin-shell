#!/usr/bin/env bash
set -euo pipefail

DATASET="Nextcloud"
LOG_TYPE="audit"

OUT_DIR="results"
mkdir -p "$OUT_DIR"

if [[ "$DATASET" == "WordPress" ]]; then
    START_IDX=0
    END_IDX=68
elif [[ "$DATASET" == "Nextcloud" ]]; then
    START_IDX=0
    END_IDX=208
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

for IDX in $(seq "$START_IDX" "$END_IDX"); do
    echo "Running complexity_metrics: dataset=$DATASET, log_type=$LOG_TYPE, null_idx=$IDX"

    python -m src.runners.stats.complexity_metrics_runner \
        --mode sweep \
        --dataset "$DATASET" \
        --log_type "$LOG_TYPE" \
        --assignment_mode indexed_stratified \
        --assignment_idx "$IDX" \
        --out_csv "$OUT_DIR/complexity_metrics_${DATASET}_${LOG_TYPE}_null_idx_${IDX}.csv"
done

echo "Finished all complexity_metrics null runs."