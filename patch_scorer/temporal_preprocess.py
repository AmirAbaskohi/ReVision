"""
Preprocessing utilities for temporal patch scoring.
Computes ground truth saliency based on component changes between consecutive steps.
"""

import math
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from PIL import Image

# Default patch configuration (matching Qwen2.5-VL)
PATCH_SIZE = 14
MERGE_SIZE = 2
IMAGE_FACTOR = PATCH_SIZE * MERGE_SIZE  # = 28


def extract_component_bboxes(components: List[Dict]) -> List[Tuple[float, float, float, float]]:
    """
    Extract bounding boxes from GuiAct component annotations.
    
    Args:
        components: List of component dicts with 'bbox' field
        
    Returns:
        List of bounding boxes as [x1, y1, x2, y2]
    """
    bboxes = []
    for comp in components:
        if 'bbox' in comp:
            bbox = comp['bbox']
            # Handle different bbox formats
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bboxes.append(tuple(bbox))
    return bboxes


def compute_bbox_overlap(bbox1: Tuple[float, float, float, float], 
                         bbox2: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes as [x1, y1, x2, y2]
        
    Returns:
        IoU score in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def find_new_components(prev_components: List[Dict], 
                       curr_components: List[Dict],
                       iou_threshold: float = 0.85) -> List[Tuple[float, float, float, float]]:
    """
    Identify new or significantly changed components between consecutive steps.
    
    Args:
        prev_components: Component annotations from previous step
        curr_components: Component annotations from current step
        iou_threshold: IoU threshold for matching components
        
    Returns:
        List of bounding boxes for new/changed components
    """
    prev_bboxes = extract_component_bboxes(prev_components)
    curr_bboxes = extract_component_bboxes(curr_components)
    
    new_bboxes = []
    
    for curr_bbox in curr_bboxes:
        # Check if this component existed in previous frame
        is_new = True
        for prev_bbox in prev_bboxes:
            iou = compute_bbox_overlap(curr_bbox, prev_bbox)
            if iou > iou_threshold:
                # Component existed in previous frame
                is_new = False
                break
        
        if is_new:
            new_bboxes.append(curr_bbox)
    
    return new_bboxes


def build_patch_score_from_component_coverage(
    image: Image.Image,
    all_component_bboxes: List[Tuple[float, float, float, float]],
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> np.ndarray:
    """
    Compute patch scores based on whether they contain any UI elements.
    
    Patches with elements receive high scores (informative).
    Patches without elements receive low scores (redundant, e.g., white background).
    
    Args:
        image: PIL image
        all_component_bboxes: List of all component bboxes in the frame
        patch_size: Size of each patch in pixels
        merge_size: Spatial merge factor
        
    Returns:
        1D array of coverage scores with shape (num_patches,)
    """
    width, height = image.size
    
    # Initialize element coverage mask (0 = no elements, 1 = has elements)
    coverage_mask = np.zeros((height, width), dtype=np.float32)
    
    # Mark regions with UI elements
    for bbox in all_component_bboxes:
        x1, y1, x2, y2 = bbox
        
        # Handle different bbox formats
        if x1 <= 1 and x2 <= 1 and y1 <= 1 and y2 <= 1:
            # Normalized coordinates [0, 1] -> convert to pixels
            x1, x2 = int(x1 * width), int(x2 * width)
            y1, y2 = int(y1 * height), int(y2 * height)
        else:
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
        
        # Clip to image boundaries
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        
        # Mark element region
        coverage_mask[y1:y2, x1:x2] = 1.0
    
    # Compute patch grid dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size
    
    # Vectorized patch mean computation
    patches_reshaped = coverage_mask.reshape(grid_h, patch_size, grid_w, patch_size)
    patches_reshaped = patches_reshaped.transpose(0, 2, 1, 3)
    patch_means = np.mean(patches_reshaped, axis=(2, 3))
    
    # Reorder to match Qwen's patch token ordering
    patch_means_reordered = patch_means.reshape(
        grid_h // merge_size, merge_size,
        grid_w // merge_size, merge_size
    )
    patch_means_reordered = patch_means_reordered.transpose(0, 2, 1, 3)
    
    # Flatten to 1D
    patch_scores = patch_means_reordered.flatten()
    
    return patch_scores.astype(np.float32)


def build_patch_score_from_component_changes(
    image: Image.Image,
    new_component_bboxes: List[Tuple[float, float, float, float]],
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> np.ndarray:
    """
    Compute patch saliency scores based on new/changed components.
    
    Patches overlapping with new components receive high scores (salient).
    Patches in unchanged regions receive low scores (redundant).
    
    Args:
        image: Current frame PIL image
        new_component_bboxes: List of bboxes for new/changed components
        patch_size: Size of each patch in pixels
        merge_size: Spatial merge factor
        
    Returns:
        1D array of saliency scores with shape (num_patches,)
    """
    width, height = image.size
    
    # Initialize saliency mask (0 = unchanged, 1 = changed)
    saliency_mask = np.zeros((height, width), dtype=np.float32)
    
    # Mark regions with new/changed components
    for bbox in new_component_bboxes:
        x1, y1, x2, y2 = bbox
        
        # Handle different bbox formats
        if x1 <= 1 and x2 <= 1 and y1 <= 1 and y2 <= 1:
            # Normalized coordinates [0, 1] -> convert to pixels
            x1, x2 = int(x1 * width), int(x2 * width)
            y1, y2 = int(y1 * height), int(y2 * height)
        else:
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
        
        # Clip to image boundaries
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        
        # Mark changed region
        saliency_mask[y1:y2, x1:x2] = 1.0
    
    # Compute patch grid dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size
    
    # Vectorized patch mean computation
    patches_reshaped = saliency_mask.reshape(grid_h, patch_size, grid_w, patch_size)
    patches_reshaped = patches_reshaped.transpose(0, 2, 1, 3)
    patch_means = np.mean(patches_reshaped, axis=(2, 3))
    
    # Reorder to match Qwen's patch token ordering
    patch_means_reordered = patch_means.reshape(
        grid_h // merge_size, merge_size,
        grid_w // merge_size, merge_size
    )
    patch_means_reordered = patch_means_reordered.transpose(0, 2, 1, 3)
    
    # Flatten to 1D
    patch_scores = patch_means_reordered.flatten()
    
    return patch_scores.astype(np.float32)


def preprocess_temporal_patch_data(
    prev_image: Image.Image,
    curr_image: Image.Image,
    prev_components: Optional[List[Dict]] = None,
    curr_components: Optional[List[Dict]] = None,
    iou_threshold: float = 0.5,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> Dict[str, torch.Tensor]:
    """
    Generate temporal patch scoring ground truth from consecutive GUI steps.
    
    Combines two signals:
    1. Component changes: New/changed components have high saliency
    2. Element coverage: Patches without any elements are redundant (low saliency)
    
    Args:
        prev_image: PIL image from previous step
        curr_image: PIL image from current step
        prev_components: Component annotations from previous step (GuiAct format)
        curr_components: Component annotations from current step (GuiAct format)
        iou_threshold: IoU threshold for component matching
        patch_size: Size of each patch in pixels
        merge_size: Spatial merge factor
        
    Returns:
        Dictionary containing:
        - patch_scores_label: Ground truth saliency scores [num_patches]
            High scores = new/changed regions with elements (salient)
            Low scores = unchanged regions or patches without elements (redundant)
    """
    # Identify new/changed components
    if prev_components is not None and curr_components is not None:
        new_bboxes = find_new_components(
            prev_components, 
            curr_components, 
            iou_threshold=iou_threshold
        )
        all_curr_bboxes = extract_component_bboxes(curr_components)
    else:
        # Fallback: if no components, assume everything is new
        new_bboxes = []
        all_curr_bboxes = []
    
    # 1. Compute saliency from new/changed components
    patch_scores_changes = build_patch_score_from_component_changes(
        curr_image,
        new_bboxes,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    
    # 2. Compute element coverage scores
    # Patches without any elements should be redundant (low score)
    patch_scores_coverage = build_patch_score_from_component_coverage(
        curr_image,
        all_curr_bboxes,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    
    # 3. Combine both signals
    # Final score = max(change_score, coverage_score)
    # This ensures:
    # - Patches with new elements get high scores (from changes)
    # - Patches with existing elements get medium scores (from coverage)
    # - Patches without any elements get low scores (redundant, e.g., white background)
    patch_scores_label = np.maximum(patch_scores_changes, patch_scores_coverage)
    
    # Patches with no elements at all should be marked as redundant
    # Set minimum score only for patches that have some element coverage
    base_saliency = 0.1
    patch_scores_label = np.where(
        patch_scores_coverage > 0.01,  # Has at least some element coverage
        np.maximum(patch_scores_label, base_saliency),
        patch_scores_label  # No coverage = keep low score (redundant)
    )
    
    # Normalize to [-1, 1] range for consistency with original PatchScorer
    patch_scores_label = patch_scores_label * 2 - 1
    
    return {
        "patch_scores_label": torch.from_numpy(patch_scores_label),
        "num_new_components": len(new_bboxes),
        "num_total_components": len(all_curr_bboxes),
    }
