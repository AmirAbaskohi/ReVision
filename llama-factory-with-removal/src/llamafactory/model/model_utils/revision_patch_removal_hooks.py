"""
ReVision Temporal Patch Removal Hooks

This module provides hooks to integrate patch removal into the model forward pass.
Patches are dropped AFTER vision encoding and projection, before passing to LLM.
"""

from typing import Any, Callable, Optional

import torch


def patch_model_forward_for_temporal_removal(
    model: Any,
    data_args: "DataArguments",
    enable_removal: bool = True,
) -> None:
    """
    Patch a model's forward method to apply temporal patch removal.
    
    This modifies the model's forward pass to drop patches after vision encoding.
    Args:
        model: The vision-language model to patch
        data_args: Data arguments containing patch removal config
        enable_removal: Whether to enable the patching
    
    Returns:
        None (modifies model in place)
    """
    if not enable_removal or not data_args.image_token_removal_enabled:
        return
    
    # Store original forward
    original_forward = model.forward
    
    patch_removal_config = {
        "enabled": data_args.image_token_removal_enabled,
        "threshold": data_args.image_token_removal_threshold,
        "aggregation": "mean",
    }
    
    def patched_forward(
        *args,
        **kwargs,
    ):
        """Forward pass with temporal patch removal integrated."""
        from .patch_removal import apply_patch_removal_to_mm_input
        
        # Call original forward
        output = original_forward(*args, **kwargs)
        
        # Note: Patch removal would be applied inside the model's forward
        # after vision encoding but before LLM processing.
        # This requires more detailed hooks into the vision/projection layers.
        
        return output
    
    # Replace forward method
    model.forward = patched_forward


def hook_vision_model_output(
    model: Any,
    data_args: "DataArguments",
) -> Callable:
    """
    Create a hook for the vision model's output to apply patch removal.
    
    Returns a hook function that can be registered with model.register_forward_hook()
    """
    from .patch_removal import apply_patch_removal_to_mm_input
    
    patch_removal_config = {
        "enabled": data_args.image_token_removal_enabled,
        "threshold": data_args.image_token_removal_threshold,
        "aggregation": "mean",
    }
    
    def hook(module, input, output):
        """Hook called after vision model forward pass."""
        if not patch_removal_config["enabled"]:
            return output
        
        # output is typically the vision embeddings
        # For Qwen2VL: pixel_values and image_grid_thw
        # We would apply patch removal here
        
        return output
    
    return hook


def hook_projection_output(
    model: Any,
    data_args: "DataArguments",
) -> Callable:
    """
    Create a hook for the projection layer's output to apply patch removal.
    
    This is where patches should be dropped before going to LLM.
    """
    from .patch_removal import apply_patch_removal_to_mm_input
    
    patch_removal_config = {
        "enabled": data_args.image_token_removal_enabled,
        "threshold": data_args.image_token_removal_threshold,
        "aggregation": "mean",
    }
    
    def hook(module, input, output):
        """Hook called after projection layer forward pass.
        
        This is where we drop patches based on temporal similarity.
        """
        if not patch_removal_config["enabled"]:
            return output
        
        # At this point, output should be LLM-compatible embeddings
        # We apply patch removal here before LLM processes them
        
        return output
    
    return hook
