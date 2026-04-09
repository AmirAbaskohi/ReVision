"""
Utilities for temporal patch removal and saliency-based token dropping.

This module handles the logic of dropping (not masking) image patches based on
temporal similarity between consecutive frames, implementing the ReVision approach.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_patch_saliency(
    prev_features: torch.Tensor,
    curr_features: torch.Tensor,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Compute patch saliency scores by comparing consecutive frame features.
    
    Args:
        prev_features: Features from previous frame [num_patches, feat_dim]
        curr_features: Features from current frame [num_patches, feat_dim]
        aggregation: How to aggregate similarity scores ("mean", "max", "min")
    
    Returns:
        Saliency scores [num_patches]. Higher = more salient (changed/new).
    """
    if prev_features is None:
        # First frame: all patches are salient
        return torch.ones(curr_features.shape[0], device=curr_features.device)
    
    # Normalize embeddings
    prev_norm = F.normalize(prev_features, p=2, dim=-1)  # [V_prev, D]
    curr_norm = F.normalize(curr_features, p=2, dim=-1)  # [V_curr, D]
    
    # Compute similarity matrix
    similarity = torch.mm(curr_norm, prev_norm.t())  # [V_curr, V_prev]
    
    # Aggregate across previous patches
    if aggregation == "mean":
        similarity_to_prev = similarity.mean(dim=-1)
    elif aggregation == "max":
        similarity_to_prev = similarity.max(dim=-1).values
    elif aggregation == "min":
        similarity_to_prev = similarity.min(dim=-1).values
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    # Invert to get saliency (dissimilarity)
    # High similarity = low saliency (redundant)
    # Low similarity = high saliency (new/changed)
    saliency = 1.0 - similarity_to_prev
    return saliency


def drop_patches_by_saliency(
    visual_tokens: torch.Tensor,
    patch_positions: Optional[torch.Tensor],
    saliency_scores: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Drop patches from visual tokens and corresponding position information based on saliency.
    
    Args:
        visual_tokens: Visual token embeddings [num_patches, hidden_dim]
        patch_positions: Position information (T, H, W) or None [num_patches, 3]
        saliency_scores: Saliency scores for each patch [num_patches]
        threshold: Drop patches with saliency < threshold
    
    Returns:
        Tuple of:
        - dropped_tokens: Tokens for patches above threshold [num_kept, hidden_dim]
        - dropped_positions: Position info for kept patches or None
        - keep_mask: Boolean mask indicating which patches were kept [num_patches]
    """
    # Create keep mask (keep patches with high saliency)
    keep_mask = saliency_scores >= threshold
    
    if keep_mask.sum() == 0:
        # If all patches would be dropped, keep the most salient one
        keep_mask = saliency_scores == saliency_scores.max()
    
    # Apply mask to tokens
    dropped_tokens = visual_tokens[keep_mask]
    
    # Apply mask to positions if provided
    dropped_positions = None
    if patch_positions is not None:
        dropped_positions = patch_positions[keep_mask]
    
    return dropped_tokens, dropped_positions, keep_mask


def process_multiimage_patch_removal(
    image_embeds_list: list,
    image_positions_list: list,
    patch_removal_config: dict,
) -> Tuple[list, list]:
    """
    Process patch removal for multiple consecutive images while preserving first image.
    
    Implements the algorithm where:
    - First image: keep all patches
    - Subsequent images: drop patches by comparing to previous image
    
    Args:
        image_embeds_list: List of image embeddings [num_images, num_patches, feat_dim]
        image_positions_list: List of position info [num_images, num_patches, 3]
        patch_removal_config: Config dict with keys:
            - "threshold": Saliency threshold for dropping patches
            - "aggregation": How to aggregate similarity scores
            - "enabled": Whether removal is enabled
    
    Returns:
        Tuple of:
        - processed_embeds: Embeddings after patch removal
        - processed_positions: Position info after patch removal
    """
    if not patch_removal_config.get("enabled", False) or len(image_embeds_list) <= 1:
        return image_embeds_list, image_positions_list
    
    processed_embeds = []
    processed_positions = []
    
    threshold = patch_removal_config.get("threshold", 0.5)
    aggregation = patch_removal_config.get("aggregation", "mean")
    
    for idx, (curr_embeds, curr_positions) in enumerate(
        zip(image_embeds_list, image_positions_list)
    ):
        if idx == 0:
            # Keep all patches from first image
            processed_embeds.append(curr_embeds)
            processed_positions.append(curr_positions)
        else:
            # Compare to previous image
            prev_embeds = image_embeds_list[idx - 1]
            prev_positions = image_positions_list[idx - 1]
            
            # Compute saliency
            saliency = compute_patch_saliency(
                prev_embeds,
                curr_embeds,
                aggregation=aggregation,
            )
            
            # Drop patches based on saliency
            kept_embeds, kept_positions, _ = drop_patches_by_saliency(
                curr_embeds,
                curr_positions,
                saliency,
                threshold=threshold,
            )
            
            processed_embeds.append(kept_embeds)
            processed_positions.append(kept_positions)
    
    return processed_embeds, processed_positions


def apply_patch_removal_to_mm_input(
    mm_inputs: dict,
    batch_imglens: list[int],
    patch_removal_config: dict,
) -> dict:
    """
    Apply patch removal to multimodal input dictionary.
    
    IMPORTANT: This function is called AFTER vision encoding and projection in the model forward pass.
    It receives actual embeddings (not images) and performs temporal patch dropping.
    
    Args:
        mm_inputs: Dict containing embeddings after vision encoding/projection:
            - pixel_values: Vision token embeddings from projection [total_patches, hidden_dim]
            - image_grid_thw: Tensor with shape (num_images, 3) where values are (T, H, W)
        batch_imglens: List of image counts per batch item
        patch_removal_config: Configuration with keys:
            - "enabled": Whether removal is enabled  
            - "threshold": Saliency threshold (0.0-1.0)
            - "aggregation": How to combine similarity scores
    
    Returns:
        Modified mm_inputs with patches dropped from subsequent images
    """
    if not patch_removal_config.get("enabled", False):
        return mm_inputs
    
    # Check if we have the required tensors
    if "pixel_values" not in mm_inputs or "image_grid_thw" not in mm_inputs:
        return mm_inputs
    
    pixel_values = mm_inputs["pixel_values"]  # Embeddings [total_patches, hidden_dim]
    image_grid_thw = mm_inputs.get("image_grid_thw")  # Positions [num_images, 3]
    
    if pixel_values is None or image_grid_thw is None:
        return mm_inputs
    
    # Split pixel_values by image index
    image_starts = []
    current_pos = 0
    
    for img_idx in range(len(image_grid_thw)):
        image_starts.append(current_pos)
        # Number of patches in this image
        num_patches = int(image_grid_thw[img_idx].prod().item())
        current_pos += num_patches
    
    image_starts.append(current_pos)  # End marker
    
    # Process images
    threshold = patch_removal_config.get("threshold", 0.5)
    aggregation = patch_removal_config.get("aggregation", "mean")
    
    kept_embeddings = []
    kept_positions = []
    updated_image_grid = []
    
    for img_idx in range(len(image_grid_thw)):
        start_idx = image_starts[img_idx]
        end_idx = image_starts[img_idx + 1]
        
        curr_embeds = pixel_values[start_idx:end_idx]  # [num_patches_i, hidden_dim]
        curr_positions = image_grid_thw[img_idx:img_idx+1]  # [1, 3]
        
        if img_idx == 0:
            # Keep all patches from first image
            kept_embeddings.append(curr_embeds)
            kept_positions.append(curr_positions)
            updated_image_grid.append(curr_positions[0].clone())
        else:
            # Get previous image  
            prev_start = image_starts[img_idx - 1]
            prev_end = image_starts[img_idx]
            prev_embeds = pixel_values[prev_start:prev_end]
            
            # Compute saliency
            saliency = compute_patch_saliency(prev_embeds, curr_embeds, aggregation)
            
            # Drop patches
            kept_embeds, kept_pos_thw, keep_mask = drop_patches_by_saliency(
                curr_embeds,
                curr_positions,
                saliency,
                threshold
            )
            
            kept_embeddings.append(kept_embeds)
            kept_positions.append(kept_pos_thw)
            
            # Update image grid to reflect dropped patches
            updated_image_grid.append(kept_pos_thw[0].clone() if kept_pos_thw is not None else curr_positions[0].clone())
    
    # Reconstruct mm_inputs with dropped patches
    if kept_embeddings:
        mm_inputs["pixel_values"] = torch.cat(kept_embeddings, dim=0)
    
    if kept_positions and kept_positions[0] is not None:
        # Create new image_grid_thw with updated dimensions
        mm_inputs["image_grid_thw"] = torch.stack(updated_image_grid, dim=0)
    
    return mm_inputs
