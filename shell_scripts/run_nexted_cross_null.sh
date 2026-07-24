#!/usr/bin/env bash
set -euo pipefail

# =========================
# User settings
# =========================
DATASET="Nextcloud"   # "Nextcloud" or "WordPress"
LOG_TYPE="nextcloud"      # For Nextcloud: audit | syslog | nextcloud
                       # For WordPress: audit | syslog
N_JOBS=6

# =========================
# Fixed settings
# =========================
MODEL="svm"
LIMIT_OUTER=50
PYTHON_MODULE="src.runners.ml.tfidf_360_nested"
OUT_DIR="results"
mkdir -p "$OUT_DIR"

# =========================
# Assignment index range
# =========================
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

# =========================
# Run all null assignments
# =========================
for IDX in $(seq "$START_IDX" "$END_IDX"); do
    echo "=================================================================="
    echo "Running TF-IDF nested null experiment"
    echo "Dataset       : $DATASET"
    echo "Log type      : $LOG_TYPE"
    echo "Model         : $MODEL"
    echo "Assignment idx: $IDX"
    echo "n_jobs        : $N_JOBS"
    echo "=================================================================="

    OUT_CSV="${OUT_DIR}/tfidf_360_nested_${DATASET}_${LOG_TYPE}_${MODEL}_null_idx_${IDX}.csv"

    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python -m "$PYTHON_MODULE" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --log_type "$LOG_TYPE" \
        --limit_outer "$LIMIT_OUTER" \
        --n_jobs "$N_JOBS" \
        --randomize_actor_labels \
        --assignment_idx "$IDX" \
        --out_csv "$OUT_CSV"
done

echo "Finished all TF-IDF null runs."