#!/bin/bash
# Training script for Temporal Patch Scorer
# Uses preprocessed data from prepare_data.py

set -e  # Exit on error

echo "============================================="
echo "Temporal Patch Scorer - Training"
echo "============================================="

# Configuration
DATA_DIR="${1:-./data/processed/train}"
OUTPUT_DIR="${2:-./checkpoints/temporal_patch_scorer}"
BASE_MODEL="${3:-Qwen/Qwen2.5-VL-3B-Instruct}"

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Base model: $BASE_MODEL"
echo ""

# Check if preprocessed data exists
if [ ! -f "$DATA_DIR/temporal_pairs_processed.json" ]; then
    echo "Error: Preprocessed data not found at $DATA_DIR"
    echo ""
    echo "Please run data preparation first:"
    echo "  bash prepare_data.sh train ./data/processed"
    echo ""
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

# Run training
python -m patch_scorer.train_temporal_scorer \
  --data_dir "$DATA_DIR" \
  --base_model_path "$BASE_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --save_steps 1000 \
  --logging_steps 10 \
  --bf16 True \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine

echo ""
echo "============================================="
echo "Training complete!"
echo "============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
