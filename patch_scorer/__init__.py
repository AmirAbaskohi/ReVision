"""
Temporal Patch Scorer - Standalone Module

A self-contained implementation for identifying salient changes between consecutive GUI steps.
This module works independently without requiring the main FocusUI codebase.

Directory Structure:
-------------------
patch_scorer/
├── __init__.py                    # Package initialization
├── requirements.txt               # Python dependencies
├── run_training.sh               # Quick start training script
├── README.md                     # Documentation
├── temporal_patch_scorer.py      # Core model architecture
├── temporal_dataset.py           # Dataset loader for GUIAct
├── temporal_preprocess.py        # Ground truth generation
├── train_temporal_scorer.py      # Training script
└── base_models/                  # Self-contained base models
    ├── qwen2_5_vl/              # Qwen 2.5 VL model
    └── qwen3_vl/                # Qwen 3 VL model

Usage:
------
1. Install dependencies:
   pip install -r requirements.txt

2. Run training:
   bash run_training.sh

3. Or customize:
   python train_temporal_scorer.py --split train --output_dir ./checkpoints
"""

from .temporal_patch_scorer import TemporalPatchScorerModel, TemporalPatchScorerConfig
from .temporal_dataset import TemporalPatchScorerDataset
from .temporal_preprocess import preprocess_temporal_patch_data

__version__ = "1.0.0"

__all__ = [
    "TemporalPatchScorerModel",
    "TemporalPatchScorerConfig",
    "TemporalPatchScorerDataset",
    "preprocess_temporal_patch_data",
]
