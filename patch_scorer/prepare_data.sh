#!/bin/bash
# Data preparation script for Temporal Patch Scorer
# Downloads GUIAct data and processes with OmniParser

set -e  # Exit on error

echo "============================================="
echo "Temporal Patch Scorer - Data Preparation"
echo "============================================="

# Configuration
SPLIT="${1:-train}"  # Default to train split
OUTPUT_DIR="${2:-./data/processed}"
DEVICE="${3:-cuda}"

echo ""
echo "Configuration:"
echo "  Split: $SPLIT"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""

# Run data preparation
python prepare_data.py \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "============================================="
echo "Data preparation complete!"
echo "============================================="
echo ""
echo "Processed data saved to: $OUTPUT_DIR/$SPLIT"
echo ""
echo "Next steps:"
echo "  1. Review the processed data and statistics"
echo "  2. Run training: bash run_training.sh"
echo ""
