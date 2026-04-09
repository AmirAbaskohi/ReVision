"""
Dataset for temporal patch scorer training.
Loads preprocessed temporal pairs with OmniParser-detected UI elements.
"""

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .temporal_preprocess import preprocess_temporal_patch_data


def decode_base64_image(base64_str: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


class TemporalPatchScorerDataset(Dataset):
    """
    Dataset for training temporal patch scorer.
    
    Loads preprocessed temporal pairs with OmniParser-detected UI elements.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        data_dir: str,
        min_pixels: int = 3136,
        max_pixels: int = 5720064,
        iou_threshold: float = 0.5,
    ):
        """
        Args:
            processor: Qwen processor for image encoding
            data_dir: Directory containing preprocessed data (temporal_pairs_processed.json)
            min_pixels: Minimum pixels for image resizing
            max_pixels: Maximum pixels for image resizing
            iou_threshold: IoU threshold for component matching
        """
        super().__init__()
        
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.iou_threshold = iou_threshold
        
        # Load preprocessed data
        data_path = Path(data_dir) / "temporal_pairs_processed.json"
        
        print(f"Loading preprocessed data from: {data_path}")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {data_path}\n"
                f"Please run: python prepare_data.py --output_dir {Path(data_dir).parent}"
            )
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.temporal_pairs = json.load(f)
        
        print(f"Loaded {len(self.temporal_pairs)} temporal pairs with OmniParser-detected elements")

    def __len__(self):
        return len(self.temporal_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample consisting of preprocessed temporal pair.
        
        Returns:
            Dictionary containing:
            - prev_pixel_values: Previous frame image tensor
            - curr_pixel_values: Current frame image tensor
            - prev_image_grid_thw: Previous frame grid dimensions
            - curr_image_grid_thw: Current frame grid dimensions
            - patch_scores_label: Ground truth saliency scores
        """
        try:
            return self._get_item(idx)
        except Exception as e:
            print(f"Failed to fetch sample {idx}. Exception: {e}")
            # Fallback to random sample
            import random
            new_idx = random.randint(0, len(self.temporal_pairs) - 1)
            return self.__getitem__(new_idx)

    def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        pair_data = self.temporal_pairs[idx]
        
        # Decode base64 image (same image for both prev and curr in this dataset)
        image = decode_base64_image(pair_data['base64'])
        
        # For temporal pairs, we use the same image twice
        # since the dataset groups QA pairs by the same screenshot
        prev_image = image
        curr_image = image
        
        # Process images with Qwen processor
        # Previous frame
        prev_messages = [{"role": "user", "content": [{"type": "image", "image": prev_image}]}]
        prev_text = self.processor.apply_chat_template(prev_messages, tokenize=False, add_generation_prompt=True)
        prev_inputs = self.processor(
            text=[prev_text],
            images=[prev_image],
            return_tensors="pt",
            padding=True,
        )
        
        # Current frame
        curr_messages = [{"role": "user", "content": [{"type": "image", "image": curr_image}]}]
        curr_text = self.processor.apply_chat_template(curr_messages, tokenize=False, add_generation_prompt=True)
        curr_inputs = self.processor(
            text=[curr_text],
            images=[curr_image],
            return_tensors="pt",
            padding=True,
        )
        
        # Get OmniParser-detected elements (all UI elements, not just actions)
        elements = pair_data.get('elements', [])
        
        # For prev/curr components, we use the same elements since they share the image
        # The difference is in which elements are relevant to each action
        prev_components = elements
        curr_components = elements
        
        # Generate ground truth saliency scores
        temporal_data = preprocess_temporal_patch_data(
            prev_image=prev_image,
            curr_image=curr_image,
            prev_components=prev_components,
            curr_components=curr_components,
            iou_threshold=self.iou_threshold,
            patch_size=self.processor.image_processor.patch_size,
            merge_size=self.processor.image_processor.merge_size,
        )
        
        return {
            'prev_pixel_values': prev_inputs['pixel_values'][0],
            'curr_pixel_values': curr_inputs['pixel_values'][0],
            'prev_image_grid_thw': prev_inputs['image_grid_thw'][0],
            'curr_image_grid_thw': curr_inputs['image_grid_thw'][0],
            'patch_scores_label': temporal_data['patch_scores_label'],
            'num_new_components': temporal_data['num_new_components'],
            'image_id': pair_data['image_id'],
            'prev_qa_num': pair_data['prev_qa_num'],
            'curr_qa_num': pair_data['curr_qa_num'],
        }


def collate_temporal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for temporal patch scorer batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    # Stack tensors
    collated = {
        'prev_pixel_values': torch.stack([item['prev_pixel_values'] for item in batch]),
        'curr_pixel_values': torch.stack([item['curr_pixel_values'] for item in batch]),
        'prev_image_grid_thw': torch.stack([item['prev_image_grid_thw'] for item in batch]),
        'curr_image_grid_thw': torch.stack([item['curr_image_grid_thw'] for item in batch]),
        'patch_scores_label': torch.stack([item['patch_scores_label'] for item in batch]),
    }
    
    return collated
