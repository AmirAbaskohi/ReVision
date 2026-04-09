"""
Training script for Temporal Patch Scorer.

Trains a model to identify salient changes between consecutive GUI steps
by comparing image patches across timesteps.

This is a standalone module and does not require the main FocusUI codebase.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoProcessor, TrainingArguments, Trainer

# Add patch_scorer to path for standalone execution
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from patch_scorer.temporal_patch_scorer import TemporalPatchScorerModel, TemporalPatchScorerConfig
from patch_scorer.temporal_dataset import TemporalPatchScorerDataset, collate_temporal_batch
from patch_scorer.base_models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from patch_scorer.base_models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


# Model configuration: embedding dimensions for each model
MODEL_EMBEDDING_DIMS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": 3584,
    "Qwen/Qwen3-VL-8B-Instruct": 4096,
}


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    base_model_path: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path to base VLM for extracting image embeddings. Supported: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct"}
    )
    projection_dim: int = field(
        default=None,
        metadata={"help": "Projection dimension for temporal scorer. If None, automatically set based on model. For Qwen2.5-VL-7B: 3584, for Qwen3-VL-8B: 4096"}
    )
    similarity_aggregation: str = field(
        default="mean",
        metadata={"help": "How to aggregate similarity scores: mean, max, or min"}
    )

    def __post_init__(self):
        """Auto-detect embedding dimension if not specified."""
        if self.projection_dim is None:
            if self.base_model_path in MODEL_EMBEDDING_DIMS:
                self.projection_dim = MODEL_EMBEDDING_DIMS[self.base_model_path]
            else:
                # Default fallback
                self.projection_dim = 3584
                print(f"Warning: Unknown model path {self.base_model_path}. Using default embedding dim 3584")


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    data_dir: str = field(
        metadata={"help": "Directory containing preprocessed data (e.g., ./data/processed/train)"}
    )
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=5720064)
    iou_threshold: float = field(
        default=0.5,
        metadata={"help": "IoU threshold for component matching"}
    )


@dataclass
class TemporalTrainingArguments(TrainingArguments):
    """Training arguments specific to temporal scorer."""
    learning_rate: float = field(default=1e-4)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    save_steps: int = field(default=1000)
    logging_steps: int = field(default=10)


class TemporalPatchScorerTrainer(Trainer):
    """
    Custom trainer for temporal patch scorer.
    Extracts image embeddings from frozen base VLM and trains temporal scorer.
    """
    
    def __init__(self, base_vlm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_vlm = base_vlm
        self.base_vlm.eval()
        
        # Freeze base VLM
        for param in self.base_vlm.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute temporal patch scorer loss.
        """
        # Extract image embeddings from frozen VLM
        with torch.no_grad():
            # Previous frame
            prev_vision_outputs = self.base_vlm.visual(
                inputs['prev_pixel_values'],
                grid_thw=inputs['prev_image_grid_thw']
            )
            prev_image_embeds = prev_vision_outputs[0]  # [B, V_prev, D]
            
            # Current frame
            curr_vision_outputs = self.base_vlm.visual(
                inputs['curr_pixel_values'],
                grid_thw=inputs['curr_image_grid_thw']
            )
            curr_image_embeds = curr_vision_outputs[0]  # [B, V_curr, D]
        
        # Forward through temporal scorer
        outputs = model(
            prev_image_embeds=prev_image_embeds,
            curr_image_embeds=curr_image_embeds,
            image_grid_thw=inputs['curr_image_grid_thw'],
            patch_scores_label=inputs['patch_scores_label'],
            return_dict=True,
        )
        
        loss = outputs['loss']
        
        if return_outputs:
            return loss, outputs
        return loss


def main():
    """Main training function."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TemporalTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load base VLM (frozen, for extracting embeddings)
    print(f"Loading base VLM: {model_args.base_model_path}")
    
    # Select appropriate model class based on model path
    if "Qwen2.5" in model_args.base_model_path or "qwen2" in model_args.base_model_path.lower():
        base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif "Qwen3" in model_args.base_model_path or "qwen3" in model_args.base_model_path.lower():
        base_vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.base_model_path}. Supported models: Qwen2.5-VL, Qwen3-VL")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_args.base_model_path)
    
    # Initialize temporal patch scorer with auto-detected embedding dimension
    print(f"Initializing temporal patch scorer with projection_dim={model_args.projection_dim}")
    config = TemporalPatchScorerConfig(
        projection_dim=model_args.projection_dim,
        similarity_aggregation=model_args.similarity_aggregation,
    )
    model = TemporalPatchScorerModel(config)
    
    # Load dataset
    print("Loading dataset")
    train_dataset = TemporalPatchScorerDataset(
        processor=processor,
        data_dir=data_args.data_dir,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
        iou_threshold=data_args.iou_threshold,
    )
    
    # Initialize trainer
    trainer = TemporalPatchScorerTrainer(
        base_vlm=base_vlm,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_temporal_batch,
    )
    
    # Train
    print("Starting training")
    trainer.train()
    
    # Save model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
