#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH_DEFAULT="/Users/folgueras/cernbox/L1T/2025_09_GNN_L1Nano/CMSSW_15_1_0_pre6/src/emeleTrigger/test/datasets/l1nano_graphs_withGenPropagated.pt"

DATASET_PATH="${1:-$DATASET_PATH_DEFAULT}"
NUM_EXAMPLES="${2:-6}"
OUTPUT_PREFIX="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Inspecting L1Nano dataset"
echo "  Dataset path  : $DATASET_PATH"
echo "  Num examples  : $NUM_EXAMPLES"
if [ -n "$OUTPUT_PREFIX" ]; then
  echo "  Output prefix : $OUTPUT_PREFIX"
fi

MPLBACKEND=Agg conda run -n cmsl1t python "$SCRIPT_DIR/inspect_dataset.py" \
  --dataset_path "$DATASET_PATH" \
  --num_examples "$NUM_EXAMPLES" \
  ${OUTPUT_PREFIX:+--output_prefix "$OUTPUT_PREFIX"} \
  --no_show

echo "Done!"
