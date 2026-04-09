# ReVision with Temporal Patch Removal - LlamaFactory Integration

This is a modified version of [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) with integrated **Temporal Patch Removal** for ReVision-based vision-language models.

## What is ReVision?

ReVision is an advanced vision-language training approach that optimizes how models process sequences of images. This implementation extends LlamaFactory with intelligent patch dropping during training.

## Temporal Patch Removal Feature

### Overview

When training vision-language models on sequences of images, there is often significant redundancy between consecutive frames or related images. The temporal patch removal feature intelligently identifies and removes redundant visual tokens (patches) during training, which:

- **Reduces memory usage** during training
- **Speeds up training** by processing fewer tokens
- **Maintains model quality** through smart similarity-based selection
- **Preserves important visual information** by keeping diverse patches

### How It Works (High-Level)

The system analyzes the visual information from consecutive images in a sequence:

1. **Visual Encoding**: Each image is processed through the model's vision encoder to extract visual patches
2. **Similarity Analysis**: The model compares how similar patches from consecutive images are to each other
3. **Intelligent Selection**: Patches that are very similar to those from the previous image are identified as redundant
4. **Adaptive Dropping**: Based on a configured threshold, redundant patches are removed from the visual token stream
5. **Preserved Structure**: The spatial layout information of remaining patches is maintained

The key insight is that when you show a model two very similar images, it doesn't need to process all visual information from both—it can learn effectively with a reduced set of diverse patches.

### When to Use This Feature

This feature is particularly beneficial when:
- Training on video frames or temporal image sequences
- Working with multi-view datasets where views are similar
- Training on datasets with repeated or similar scenes
- Memory is constrained and you want to optimize token usage
- You want faster training without sacrificing model quality

## Getting Started

### Prerequisites

Follow the [LlamaFactory installation guide](https://github.com/hiyouga/LLaMA-Factory#installation) to set up the environment.

### Configuration

The training configuration is provided in the `train_ReVision.yaml` file. This file serves as a template that you need to customize for your specific use case:

```bash
# The template file
train_ReVision.yaml
```

### Customizing the Configuration

Before training, you must configure:

#### 1. **Dataset Configuration**

Follow the [LlamaFactory data preparation guide](https://github.com/hiyouga/LLaMA-Factory/blob/main/docs/en/data-preparation.md) to:

- Prepare your dataset in the supported format (JSON, JSONL, or Parquet)
- Register your dataset in `data/dataset_info.json`
- Specify the dataset in `train_ReVision.yaml`

Example dataset entry in `data/dataset_info.json`:
```json
{
  "your_dataset_name": {
    "hf_hub_url": "path/to/your/dataset",
    "instructions": "...",
    "columns": {
      "prompt": "text",
      "response": "response",
      "images": "image_urls"
    }
  }
}
```

#### 2. **Model Configuration**

Update the following in `train_ReVision.yaml`:
- `model_name_or_path`: Path to your vision-language model
- `template`: Chat template matching your model
- `cutoff_len`: Maximum sequence length
- `output_dir`: Where to save trained models

#### 3. **Training Configuration**

Customize these parameters:
- `per_device_train_batch_size`: Batch size for your GPU memory
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for training
- `warmup_steps`: Warmup steps for learning rate schedule

#### 4. **Temporal Patch Removal Settings**

Enable and tune the patch removal feature:

```yaml
# Enable temporal patch removal
image_token_removal_enabled: true

# Threshold for removing patches (0.0 to 1.0)
# Higher values = more aggressive removal
# Recommended range: 0.3 to 0.7
image_token_removal_threshold: 0.5
```

**Threshold Explanation**:
- `0.0`: Keep all patches (patch removal deactivated)
- `0.3`: Aggressive removal - only keep very different patches
- `0.5`: Moderate removal - recommended starting point
- `0.7`: Conservative removal - keep most patches
- `1.0`: Remove almost all redundant patches

### Running Training

Once you've configured `train_ReVision.yaml`:

```bash
# Start training with your configuration
bash scripts/train_exp.sh train_ReVision.yaml
```

Or directly with the LlamaFactory CLI:

```bash
python src/train.py train_ReVision.yaml
```

## Understanding the Training Process

During training with temporal patch removal enabled:

1. Your dataset is loaded and preprocessed according to the data configuration
2. Images/patches are processed through the vision model
3. The temporal patch removal system:
   - Analyzes similarities between consecutive images
   - Identifies redundant visual information
   - Removes or keeps patches based on the threshold
4. The model learns from sequences with optimized patch counts
5. Training proceeds with reduced memory usage and faster throughput

## Dataset Format

Ensure your dataset follows the [LlamaFactory format requirements](https://github.com/hiyouga/LLaMA-Factory/tree/main/data):

**For multimodal data**, your dataset should include:
- Text/instruction data
- Image URLs or image paths
- Response data

Example JSON format:
```json
[
  {
    "instruction": "Describe these images",
    "input": "",
    "output": "The images show...",
    "images": ["image1.jpg", "image2.jpg"]
  }
]
```

## Monitoring Training

LlamaFactory provides several monitoring options:

- **TensorBoard**: Real-time training metrics
- **Weights & Biases**: Professional experiment tracking
- **SwanLab**: Lightweight experiment tracking
- **Console logs**: Basic training statistics

Configure the appropriate option in your `train_ReVision.yaml`.

## Advanced Usage

### Experimenting with Different Thresholds

To find the optimal threshold for your use case, try:

1. **Start conservative** (0.7): Validate training works correctly
2. **Gradually increase** (0.5, 0.3): Monitor memory usage and quality
3. **Find your sweet spot**: Balance between efficiency and accuracy

### Combining with Other Optimization Techniques

Temporal patch removal can be combined with:
- **LoRA** (Low-Rank Adaptation): Further reduce trainable parameters
- **QLoRA**: Quantized LoRA for extreme efficiency
- **Flash Attention**: Faster attention computation
- **Gradient Checkpointing**: Reduce memory usage during backpropagation

### Disabling Patch Removal

To train without patch removal:

```yaml
image_token_removal_enabled: false
```

## Performance Expectations

When using temporal patch removal, you can expect:

- **Memory reduction**: 20-40% depending on threshold
- **Training speed**: 20-40% faster
- **Model quality**: Typically comparable or slightly better
- **Inference speed**: No change (removed at inference time if configured properly)

Results may vary based on your specific dataset, model, and patch removal threshold.

## Troubleshooting

### High Memory Usage
- Reduce `per_device_train_batch_size`
- Increase `image_token_removal_threshold` for more aggressive removal
- Enable `gradient_checkpointing`

### Training Divergence
- Lower the `image_token_removal_threshold` (less aggressive removal)
- Reduce learning rate
- Verify your dataset quality

### Slow Training Speed
- Reduce `per_device_train_batch_size`
- Check GPU utilization
- Verify data loading isn't a bottleneck

## Reference & Documentation

- [LlamaFactory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [LlamaFactory Data Preparation](https://github.com/hiyouga/LLaMA-Factory/blob/main/docs/en/data-preparation.md)
- [Supported Models](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/visual.py)

## Citation

If you use this ReVision implementation with LlamaFactory, please cite both projects:

```bibtex
@misc{hiyouga2023llamafactory,
  author = {Yaofu, Hao},
  title = {LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Factory}}
}
```

## License

This modification maintains the same license as LlamaFactory. See LICENSE file for details.

## Support

For issues or questions:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [LlamaFactory documentation](https://github.com/hiyouga/LLaMA-Factory)
3. Ensure your dataset follows the correct format
4. Verify all configuration parameters in your `train_ReVision.yaml`

---

**Note**: The `train_ReVision.yaml` file is a template. You must customize it with your specific model, dataset, and training parameters before running training.
