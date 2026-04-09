"""
Example: Training with Temporal Patch Removal

This example demonstrates how to enable and use temporal patch removal
for efficient training on multi-image sequences.
"""

# Example 1: Using YAML Configuration
# Save this as `train_with_patch_removal.yaml`

YAML_CONFIG = """
### Model Configuration
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
image_max_pixels: 2100000
trust_remote_code: true

### Training Setup
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false

### Dataset
dataset: your_dataset_name
template: qwen2_vl
cutoff_len: 16384
dataset_dir: ./data

### Important: Temporal Patch Removal Settings
image_token_removal_enabled: true              # Enable patch removal
image_token_removal_threshold: 0.5             # Drop patches with saliency < 0.5

### Training Parameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
bf16: true

### Output
output_dir: ./output/qwen_with_patch_removal
logging_steps: 10
save_steps: 200
"""


# Example 2: Programmatic Usage
# Run this script to train with patch removal

def example_training_with_patch_removal():
    """Demonstrates training setup with patch removal enabled."""
    
    from llamafactory.hparams import DataArguments, ModelArguments, TrainingArguments, FinetuningArguments
    from llamafactory.train import run_exp
    
    # Create data arguments with patch removal enabled
    data_args = DataArguments(
        dataset="agentnet_windowed_shard00000",
        template="qwen2_vl",
        cutoff_len=16384,
        dataset_dir="./data",
        # Patch removal settings
        image_token_removal_enabled=True,          # Enable temporal patch removal
        image_token_removal_threshold=0.5,         # Saliency threshold
    )
    
    # Model arguments
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        image_max_pixels=2100000,
        trust_remote_code=True,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/qwen_patch_removal",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1.0e-5,
        num_train_epochs=1.0,
        bf16=True,
        logging_steps=10,
        save_steps=200,
    )
    
    # Fine-tuning arguments
    finetuning_args = FinetuningArguments(
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
    )
    
    # Run training
    run_exp(data_args, model_args, training_args, finetuning_args)


# Example 3: Understanding Temporal Patch Removal

def example_understand_patch_removal():
    """
    Show how patch removal works with consecutive frames.
    """
    import torch
    from llamafactory.model.model_utils.patch_removal import (
        compute_patch_saliency,
        drop_patches_by_saliency,
    )
    
    # Simulate two consecutive image frames
    # Each frame has 100 patches with 768-dim embeddings
    
    # Frame 1 (previous)
    prev_features = torch.randn(100, 768, dtype=torch.float32)
    
    # Frame 2 (current) - mostly similar to frame 1, with some changes
    curr_features = prev_features.clone()
    # Add some variations (10 out of 100 patches are different)
    curr_features[10:15] = torch.randn(5, 768, dtype=torch.float32)
    
    # Compute saliency scores
    saliency_scores = compute_patch_saliency(
        prev_features,
        curr_features,
        aggregation="mean"
    )
    
    print(f"Saliency scores shape: {saliency_scores.shape}")
    print(f"Min score: {saliency_scores.min():.4f}")
    print(f"Max score: {saliency_scores.max():.4f}")
    print(f"Mean score: {saliency_scores.mean():.4f}")
    
    # Drop patches with low saliency
    threshold = 0.5
    visual_tokens = torch.randn(100, 2048)  # LLM hidden dim
    patch_positions = torch.tensor([[0, 8, 8]] * 100)  # [T, H, W] for each patch
    
    dropped_tokens, dropped_positions, keep_mask = drop_patches_by_saliency(
        visual_tokens,
        patch_positions,
        saliency_scores,
        threshold
    )
    
    num_dropped = (~keep_mask).sum().item()
    num_kept = keep_mask.sum().item()
    
    print(f"\nPatches kept: {num_kept}/100")
    print(f"Patches dropped: {num_dropped}/100")
    print(f"Reduction: {100 * num_dropped / 100:.1f}%")
    
    # The kept tokens maintain position information
    if dropped_positions is not None:
        print(f"\nKept position info shape: {dropped_positions.shape}")
        print(f"Position IDs preserved: {dropped_positions[0]}")  # T, H, W


# Example 4: Hyperparameter Tuning

def example_hyperparameter_tuning():
    """Show the effect of different thresholds."""
    
    import torch
    from llamafactory.model.model_utils.patch_removal import compute_patch_saliency
    
    print("Effect of Threshold on Patch Retention:\n")
    print("Threshold | Patches Kept | Patches Dropped | Reduction")
    print("-" * 55)
    
    # Simulate batch of image pairs
    prev_features = torch.randn(100, 768)
    curr_features = prev_features.clone()
    curr_features[::5] += torch.randn(20, 768) * 0.5  # 20% variation
    
    saliency = compute_patch_saliency(prev_features, curr_features)
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        kept = (saliency >= threshold).sum().item()
        dropped = 100 - kept
        reduction = 100 * dropped / 100
        print(f"{threshold:.1f}      | {kept:3d}/100      | {dropped:3d}          | {reduction:5.1f}%")
    
    print("\nRecommendations:")
    print("- GUI with static background: threshold=0.6-0.7 (more aggressive)")
    print("- Mixed dynamic/static: threshold=0.5 (balanced)")
    print("- High-variation content: threshold=0.3-0.4 (conservative)")


if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Patch Removal Examples")
    print("=" * 60)
    
    # Example 3
    print("\n[Example 3] Understanding Patch Removal")
    print("-" * 60)
    example_understand_patch_removal()
    
    # Example 4
    print("\n\n[Example 4] Hyperparameter Effects")
    print("-" * 60)
    example_hyperparameter_tuning()
    
    print("\n" + "=" * 60)
    print("To train with patch removal:")
    print("1. Use YAML: llamafactory-cli train train_with_patch_removal.yaml")
    print("2. Or modify your config with:")
    print("   - image_token_removal_enabled: true")
    print("   - image_token_removal_threshold: 0.5")
    print("=" * 60)
