# Temporal Patch Scorer

A temporal extension of patch scoring that identifies salient changes between consecutive GUI steps using OmniParser-detected UI elements.

## Overview

The Temporal Patch Scorer computes **prev-image-to-curr-image** similarity to identify:

- **High saliency patches**: New or changed regions (low similarity to previous frame)
- **Low saliency patches**: Redundant/unchanged regions (high similarity to previous frame)

## Pipeline

### 1. Data Preparation (`prepare_data.py`)

Downloads GUIAct dataset and processes screenshots with OmniParser:

```bash
# Prepare training data
bash prepare_data.sh train ./data/processed

# Prepare test data
bash prepare_data.sh test ./data/processed
```

**What it does:**
- Downloads `web-multi_{split}_data.json` and `web-multi_{split}_images.parquet` from HuggingFace
- Groups QA pairs by `image_id`
- Uses **OmniParser** to detect ALL UI elements (not just action targets)
- Creates temporal pairs with full UI element annotations
- Saves to `temporal_pairs_processed.json`

### 2. Model Training (`train_temporal_scorer.py`)

Trains the temporal patch scorer on preprocessed data:

```bash
# Train model
bash run_training.sh ./data/processed/train ./checkpoints/temporal_scorer

# Or run complete pipeline (prepare + train)
bash run_pipeline.sh
```

## How It Works

### Input
- **Previous frame** image with OmniParser-detected UI elements
- **Current frame** image with OmniParser-detected UI elements
- Preprocessed temporal pairs from `prepare_data.py`

### Ground Truth Generation

The model learns from component-based changes:

**1. OmniParser detects ALL UI elements:**
```python
elements = parse_ui_elements_with_omniparser(image, processor, model)
# Returns: [{'id': 0, 'bbox': [x1, y1, x2, y2], 'label': 1, 'score': 0.95}, ...]
```

**2. Find new/changed components:**
```python
new_components = find_new_components(prev_components, curr_components)
patch_scores_label = build_patch_score_from_component_changes(curr_image, new_components)
```

Components are matched using IoU (Intersection over Union):
- IoU > threshold → Same component (unchanged)
- IoU < threshold → New/changed component (high saliency)

**3. Patch-level saliency:**
- Patches overlapping new/changed components → **High scores**
- Patches in unchanged regions → **Low scores**

### Model Architecture

```
Previous Image → Vision Encoder → Normalized Embeds ──┐
                                                       │
                                                   Similarity Matrix
                                                       │
Current Image → Vision Encoder → Normalized Embeds ───┘
                                                       │
                                                       ↓
                                             Saliency = 1 - Similarity
```

**Inversion logic:**
- High similarity to previous frame → Low saliency (unchanged/redundant)
- Low similarity to previous frame → High saliency (new/changed)

### 4. Loss

KL divergence between predicted and ground truth saliency distributions.

## Installation

This module is completely self-contained and does not require the main FocusUI codebase.

### Install Dependencies

```bash
cd patch_scorer
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare Data

```bash
# Prepare training data
bash prepare_data.sh train ./data/processed

# Prepare test data (optional)
bash prepare_data.sh test ./data/processed
```

This will:
1. Download GUIAct dataset from HuggingFace
2. Use OmniParser to detect all UI elements
3. Save preprocessed temporal pairs

### Step 2: Train Model

```bash
# Train on preprocessed data
bash run_training.sh ./data/processed/train ./checkpoints/temporal_scorer
```

### Or Run Complete Pipeline

```bash
# Prepare data + train in one command
bash run_pipeline.sh
```

## Advanced Usage

### Custom Data Preparation

```bash
python prepare_data.py \
  --split train \
  --output_dir ./data/processed \
  --device cuda
```

### Custom Training

```bash
python train_temporal_scorer.py \
  --data_dir ./data/processed/train \
  --base_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir ./checkpoints/temporal_scorer \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --save_steps 1000
```

## Data Format

### Preprocessed Data Structure

After running `prepare_data.py`, the data is saved as `temporal_pairs_processed.json`:

```json
[
  {
    "pair_id": "pair_000000",
    "image_id": "dffe7794-aa20-48fa-98e9-c4342caa93de",
    "prev_qa_num": "01",
    "curr_qa_num": "02",
    "image_size": {"width": 1920, "height": 1200},
    "base64": "...",
    "elements": [
      {
        "id": 0,
        "bbox": [1399, 30, 1545, 74],
        "label": 1,
        "score": 0.95
      }
    ],
    "prev_action": [...],
    "curr_action": [...],
    "prev_question": "Sign up for an account",
    "curr_question": "Click login button"
  }
]
```

**Key Fields:**
- `elements`: ALL UI elements detected by OmniParser (not just action targets)
- `bbox`: [x1, y1, x2, y2] format
- `prev_action`/`curr_action`: Original GUIAct action annotations
    "width": 155,
    "x": 145,
    "y": 108
  },
  "text": "Thursday,",
  "ui_type": "TEXT"
}
```

The dataset loader:
1. Groups samples by episode ID
2. Sorts by step number
3. Creates consecutive pairs (step_n, step_n+1)
4. Converts elements to bboxes for change detection

## Applications

1. **Temporal visual token reduction**: Drop unchanged patches across video frames
2. **Change detection**: Identify UI updates in dynamic interfaces
3. **Attention guidance**: Focus model on regions that changed after user actions
4. **Video understanding**: Efficient processing of GUI interaction sequences

## Differences from Original PatchScorer

| Aspect | Original PatchScorer | Temporal PatchScorer |
|--------|---------------------|---------------------|
| Input | Text instruction + Image | Previous Image + Current Image |
| Similarity | Text ↔ Image patches | Prev patches ↔ Curr patches |
| Goal | Find task-relevant regions | Find changed regions |
| Ground Truth | BBox + UI graph | Component changes |
| Application | Single-step grounding | Multi-step trajectories |

## Future Enhancements

- [ ] Add visual change detection fallback (pixel-level diff)
- [ ] Support for longer temporal windows (t-2, t-1, t)
- [ ] Integration with action prediction models
- [ ] Optical flow-based saliency augmentation
