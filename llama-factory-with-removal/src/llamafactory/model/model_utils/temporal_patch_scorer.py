"""
Temporal Patch Scorer: Identifies salient changes between consecutive GUI steps.

Instead of instruction-to-image similarity, this computes image-to-image patch similarity
across consecutive timesteps to identify which patches have changed (high saliency)
and which are redundant (low saliency).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import logging


# Constants
SPATIAL_MERGE_SIZE = 2
DEFAULT_PROJECTION_DIM = 2048
PROJECTION_DROPOUT = 0.1


class MHATokenFeatureEnhancer(nn.Module):
    """Lightweight transformer-style enhancer for token embeddings.

    Keeps embedding dimension unchanged; strengthens local/global interactions
    Expects inputs of shape [B, L, D].
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, D]
        residual = x
        attn_out, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        x_norm = self.layer_norm(residual + self.dropout(attn_out))
        x = self.mlp(x_norm)
        return x


class TemporalPatchScorerConfig(PretrainedConfig):
    """Configuration class for Temporal Patch Scorer model."""
    
    model_type = "temporal_patch_scorer"

    def __init__(
        self,
        projection_dim: int = DEFAULT_PROJECTION_DIM,
        projection_dropout: float = PROJECTION_DROPOUT,
        similarity_aggregation: str = "mean",  # "mean", "max", "min"
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.projection_dim = projection_dim
        self.projection_dropout = projection_dropout
        self.similarity_aggregation = similarity_aggregation


class TemporalPatchScorerModel(PreTrainedModel):
    """
    Temporal Patch Scorer for identifying salient changes between consecutive frames.
    
    Unlike the original PatchScorer which compares text-to-image,
    this model compares prev_image-to-curr_image to identify:
    - High saliency patches: New or changed regions
    - Low saliency patches: Unchanged/redundant regions
    """

    model_type = "temporal_patch_scorer"
    config_class = TemporalPatchScorerConfig

    def __init__(self, config: TemporalPatchScorerConfig):
        super().__init__(config)

        self.projection_dim = config.projection_dim
        self.projection_dropout = config.projection_dropout
        self.vision_embed_dim = self.projection_dim

        # Initialize vision enhancers for both timesteps
        self.prev_vision_enhancer = MHATokenFeatureEnhancer(
            embed_dim=self.vision_embed_dim,
            dropout=self.projection_dropout,
        ).to(torch.bfloat16)

        self.curr_vision_enhancer = MHATokenFeatureEnhancer(
            embed_dim=self.vision_embed_dim,
            dropout=self.projection_dropout,
        ).to(torch.bfloat16)

        self.similarity_aggregation = config.similarity_aggregation

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply tanh constraint and L2 normalization to embeddings."""
        embeddings = torch.tanh(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)

    def _compute_merged_patches_info(self, image_grid_thw: torch.LongTensor) -> torch.Tensor:
        """Compute cumulative sequence lengths for merged image patches."""
        t, h, w = image_grid_thw.unbind(dim=1)
        merged_patches_per_image = (
            (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE) * t
        )
        return F.pad(merged_patches_per_image.cumsum(0), (1, 0), value=0)

    def forward(
        self,
        prev_image_embeds: Optional[torch.FloatTensor] = None,
        curr_image_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        patch_scores_label: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for temporal patch scoring.
        
        Args:
            prev_image_embeds: Image embeddings from previous timestep [B, V, D]
            curr_image_embeds: Image embeddings from current timestep [B, V, D]
            image_grid_thw: Image grid dimensions
            patch_scores_label: Ground truth patch scores [B, V] (optional)
            
        Returns:
            Dict containing:
            - patch_scores: Saliency scores for current image patches [B, V]
                Higher scores = changed/new patches
                Lower scores = redundant/unchanged patches
            - loss: KL divergence loss if labels provided
        """
        
        if prev_image_embeds is None or curr_image_embeds is None:
            raise ValueError("Both prev_image_embeds and curr_image_embeds must be provided")

        # Enhance embeddings
        prev_embeds = self.prev_vision_enhancer(prev_image_embeds.unsqueeze(0) if prev_image_embeds.dim() == 2 else prev_image_embeds)
        curr_embeds = self.curr_vision_enhancer(curr_image_embeds.unsqueeze(0) if curr_image_embeds.dim() == 2 else curr_image_embeds)

        # Normalize embeddings
        prev_embeds = self._normalize_embeddings(prev_embeds)  # [B, V_prev, D]
        curr_embeds = self._normalize_embeddings(curr_embeds)  # [B, V_curr, D]

        # Compute similarity matrix between previous and current patches
        # High similarity = patch is similar to previous frame = low saliency (redundant)
        # Low similarity = patch is new/changed = high saliency (important)
        similarity_matrix = torch.bmm(
            curr_embeds,                    # [B, V_curr, D]
            prev_embeds.transpose(-1, -2)   # [B, V_prev, D] -> [B, D, V_prev]
        )  # [B, V_curr, V_prev]

        # Aggregate similarity scores across previous patches
        if self.similarity_aggregation == "mean":
            similarity_to_prev = similarity_matrix.mean(dim=-1)  # [B, V_curr]
        elif self.similarity_aggregation == "max":
            similarity_to_prev = similarity_matrix.max(dim=-1).values  # [B, V_curr]
        elif self.similarity_aggregation == "min":
            similarity_to_prev = similarity_matrix.min(dim=-1).values  # [B, V_curr]
        else:
            raise ValueError(f"Unsupported similarity_aggregation: {self.similarity_aggregation}")

        # Invert similarity to get saliency (dissimilarity)
        # High similarity to previous -> Low saliency (redundant)
        # Low similarity to previous -> High saliency (new/changed)
        patch_scores = 1.0 - similarity_to_prev  # [B, V_curr]

        # Compute loss if labels provided
        if patch_scores_label is not None:
            loss = self.compute_loss(patch_scores, patch_scores_label)
        else:
            loss = None

        if not return_dict:
            return (
                prev_embeds,
                curr_embeds,
                similarity_matrix,
                patch_scores,
                loss,
            )

        return {
            "prev_embeds": prev_embeds,
            "curr_embeds": curr_embeds,
            "similarity_matrix": similarity_matrix,
            "patch_scores": patch_scores,
            "loss": loss,
        }
    
    def compute_loss(self, patch_scores: torch.Tensor, patch_scores_label: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between predicted and ground truth patch saliency.
        
        Args:
            patch_scores: Predicted saliency scores [B, V]
            patch_scores_label: Ground truth saliency scores [B, V]
            
        Returns:
            KL divergence loss
        """
        ps_log_probs = F.log_softmax(patch_scores, dim=-1)
        ps_target_dist = F.softmax(patch_scores_label, dim=-1).clamp_min(1e-12)
        loss = F.kl_div(ps_log_probs, ps_target_dist, reduction="batchmean")
        return loss
