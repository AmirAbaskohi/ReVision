# AgentNet to Qwen Training Data Converter

Automated pipeline to download and prepare AgentNet data for training Qwen vision-language models.

## Overview

This pipeline processes **ALL samples from BOTH datasets** (Ubuntu + Windows/Mac) totaling **~23K training samples**.

### Pipeline Steps

1. **Download** (`1_download_agentnet.sh`) - Downloads JSONL files and split zip archives from HuggingFace
2. **Extract** (`2_extract_images.sh`) - Extracts images from split zip files with progress tracking  
3. **Generate** (`3_generate_training_data.py`) - Converts JSONL data to individual Qwen-format JSON files
4. **Windowing** (`4_windowing_samples.py`) - Creates windowed samples with configurable screenshot history (default: 3 most recent images per sample)
5. **Export to LLaMA-Factory** (`5_add_to_llama_factory.py`) - Aggregates windowed samples into sharded JSONL files compatible with LLaMA-Factory training pipeline

All scripts process both datasets automatically - no configuration needed.

### Source Data Format (AgentNet)

AgentNet contains GUI interaction trajectories with:
- **task_id**: Unique identifier for each task
- **instruction**: Natural language task description
- **traj**: List of interaction steps, each containing:
  - `index`: Step number
  - `image`: Screenshot filename
  - `value`: Contains `observation`, `thought`, `action`, `code`, `reflection`

### Target Data Format (Qwen)

Converts to a multi-turn conversation format where the assistant's reasoning is
followed by a structured `<tool_call>` block:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "## Task: ...\n\nWe are now on this page. What should we do next?\n<image>"
    },
    {
      "role": "assistant",
      "content": "The last action opened GIMP, I will now open the file. To load the image I'll press Ctrl+O.\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"type\": \"key\", \"keys\": [\"ctrl\", \"o\"]}}\n</tool_call>"
    }
  ],
  "images": [
    "path/to/image1.png",
    "path/to/image2.png"
  ]
}
```

The assistant's reasoning text is the step's `reflection` (reflecting on the prior
step's outcome) followed by its `thought`, concatenated as plain prose. The
`code` (raw pyautogui/`computer.*` call) is parsed and converted into a
`{"name": "computer_use", "arguments": {...}}` tool call. Normalized `x`/`y`
coordinates (`[0, 1]`) are converted to pixel `coordinate: [x, y]` using the
actual resolution of the step's screenshot (read via Pillow). Steps whose code
can't be mapped to a supported action are dropped from that trajectory.

## Requirements

- **Python 3.8+** with `tqdm`, `Pillow`
- **HuggingFace CLI** (auto-installed by scripts)
- **7z** for extracting split archives (auto-installed on macOS via brew)
- **~181GB disk space** for full dataset

## Usage

### Quick Start - Run Complete Pipeline

Run all three steps automatically:
```bash
chmod +x run_all.sh
./run_all.sh
Process all ~23K samples from both datasets:
```bash
chmod +x *.sh
./run_all.sh
```

This will:
- Download Ubuntu dataset (5K samples, ~73GB) 
- Download Windows/Mac dataset (18K samples, ~108GB)
- Extract all images from split zip archives
- Generate ~23K individual JSON training files
- Create windowed samples with configurable screenshot history
- Export to sharded JSONL format for LLaMA-Factory training

### Step-by-Step Execution

Run each step individually:

```bash
# Step 1: Download both datasets from HuggingFace (~181GB)
./1_download_agentnet.sh

# Step 2: Extract images from split zip archives
./2_extract_images.sh

# Step 3: Generate training JSON files (all samples)
python3 3_generate_training_data.py

# Step 4: Create windowed samples (3 most recent images default)
python3 4_windowing_samples.py

# Step 5: Export to LLaMA-Factory format (sharded JSONL)
python3 5_add_to_llama_factory.py
```

### Configuration Options

#### Window Size
Control how many most recent screenshots are kept per sample (default: 3):
```bash
# Use 5 most recent screenshots
./run_all.sh --window-size=5

# Use default of 3 screenshots
./run_all.sh --window-size=3
```

#### Skip Steps

If you already have downloaded or extracted data:

```bash
# Skip download (use existing data)
./run_all.sh --skip-download

# Skip download and extraction (just regenerate JSONs and windowing)
./run_all.sh --skip-download --skip-extract
```

#### Combine Options

```bash
# Custom window size with skipped steps
./run_all.sh --window-size=8 --skip-download --skip-extract
```

## Step 5: Export to LLaMA-Factory

### Overview

Step 5 aggregates all windowed training samples into a format optimized for LLaMA-Factory training. It creates **sharded JSONL files** where each shard contains at most 10,000 samples for efficient loading and memory management.

### What Step 5 Does

1. **Reads windowed samples** from: `agentnet_data/windowed_training_data/*.json`
2. **Rewrites image paths** to be relative to the LLaMA-Factory data directory:
   - Input: `agentnet_data/images/file.png`
   - Output: `../../agentnet_data/images/file.png`
3. **Creates sharded JSONL files** with up to 10k samples each:
   - `agentnet_windowed.shard00001.jsonl`
   - `agentnet_windowed.shard00002.jsonl`
   - etc.
4. **Saves output** to: `../llama-factory-with-removal/data/agentnet_windowed_shards/`

### Output Format

Each JSONL file contains one JSON object per line (JSONL format):
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "images": ["../../agentnet_data/..."]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "images": ["../../agentnet_data/..."]}
```

### Why Sharding?

- **Memory efficiency**: 10k samples per file prevents OOM during loading
- **Parallel processing**: LLaMA-Factory can load multiple shards independently
- **Training flexibility**: Easy to use subsets (train on shards 1-3, validate on shard 4, etc.)
- **Fault tolerance**: Corrupt shard doesn't affect other shards

### Usage

```bash
# Generate sharded JSONL files (called automatically by run_all.sh)
python3 5_add_to_llama_factory.py
```

### Configuration for LLaMA-Factory

In your LLaMA-Factory dataset config, point to the shards directory:

```yaml
# example_dataset.yaml
data_path: data/agentnet_windowed_shards/agentnet_windowed.shard*.jsonl
template: qwen_vl  # or your desired template
output_dir: ./models/agentnet_finetuned
```

### Output Example

For ~23K total samples with 10k per shard:
```
agentnet_windowed_shards/
├── agentnet_windowed.shard00001.jsonl (10,000 samples)
├── agentnet_windowed.shard00002.jsonl (10,000 samples)
└── agentnet_windowed.shard00003.jsonl (3,000 samples)
```

## Output Structure
|--------|-------------|---------|
| `--output-dir DIR` | Output directory for all data | `./agentnet_data` |
| `--dataset-type TYPE` | Dataset to process: `ubuntu`, `win_mac`, or `both` | `ubuntu` |
| `--skip-download` | Skip downloading (use existing data) | `false` |
| `--include-all-fields` | Include observation, thought, reflection | `false` |
| `--max-samples N` | Limit number of samples to process | None (all) |

### Python Script Arguments

All shell script options are available, plus:
- Can be imported as a module for custom processing
- `convert_trajectory_to_qwen_format()` function for custom conversions

## Dataset Information

### Ubuntu Dataset
- **File**: `agentnet_ubuntu_5k.jsonl` (~5K samples)
- **Images**: Split across 13 zip parts + final zip (~73 GB total)
- **Domain**: Linux GUI interactions

### Windows/Mac Dataset
- **File**: `agentnet_win_mac_18k.jsonl` (~18K samples)
- **Images**: Split across 23 zip parts + final zip (~108 GB total)
- run_all.sh --max-samples 10 --data-dir ./test_data
```

### Example 2: High-quality Ubuntu dataset
```bash
./run_all.sh \
    --dataset-type ubuntu \
    --min-score 7 \
    --include-all-fields \
    --data-dir /path/to/large/storage
```

### Example 3: Process existing downloaded data
```bash
./run_all.sh \
    --skip-download \
    --skip-extract \
    --dataset-type both
```

### Example 4: Step-by-step with different datasets
```bash
# Download both datasets
python3 1_download_agentnet.py --dataset-type both

# Extract only Ubuntu
python3 2_extract_images.py --dataset-type ubuntu

# Generate with custom filtering
python3 3_generate_training_data.py \
    --dataset-type ubuntu \
    --min-score 6 \
    --max-samples 1000et with all fields
```bash
./prepare_agentnet.sh \
    --dataset-type ubuntu \
    --include-all-fields \
    --output-dir /path/to/large/storage
```

### Example 3: Process existing downloaded data
```bash
./prepare_agentnet.sh \
    --skip-download \
    --dataset-type both \
    --output-dir ./agentnet_data
```

## Troubleshooting

### 7z command not found
Install p7zip:
- **macOS**: `brew install p7zip`
- **Ubuntu/Debian**: `sudo apt-get install p7zip-full`
- **Other**: Download from [7-zip.org](https://www.7-zip.org/)

### Out of disk space
The full datasets are very large (~180 GB for both). Consider:
- Processing one dataset at a time
- Using `--max-samples` to limit data
- Ensuring adequate storage before downloading

### HuggingFace download issues
- Check internet connection
- Verify you can access `xlangai/AgentNet` on HuggingFace
- May need to login: `huggingface-cli login`
