"""AI AINAK Version 2 — Multimodal Fusion Transformer

Unifies:
1. Vision V2: YOLO-World v2 Small (256-D: 251 category max-conf + 5-D spatial geometry)
2. Audio: PANNs CNN14 (2,048-D dense acoustic embedding)
3. Gesture V2: GestureTemporalNet (128-D kinematic embedding from 24-frame pose sequences)

Multi-Task Output Heads:
- Hazard Scenario Classification (8 classes: ENVIRONMENT_INFO, VEHICLE_APPROACHING, etc.)
- Continuous Hazard Priority Score ([0.0, 1.0])
- Discrete Alert Tier Classification (4 classes: none, informational, caution, urgent)
- Dynamic Cross-Modal Attention Contributions ([B, 3])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import nn

MODALITIES_V2 = ("vision", "audio", "gesture")
DEFAULT_FEATURE_DIMS_V2 = {"vision": 256, "audio": 2048, "gesture": 128}

HAZARD_SCENARIOS = (
    "ENVIRONMENT_INFO",
    "VEHICLE_APPROACHING",
    "VEHICLE_CROSSING_PATH",
    "PEDESTRIAN_IN_PATH",
    "PATH_OBSTRUCTION",
    "TERRAIN_DROP_AHEAD",
    "FIRE_HAZARD",
    "LOW_CONFIDENCE",
)

ALERT_TIERS = ("none", "informational", "caution", "urgent")


@dataclass
class V2FusionOutput:
    scenario_logits: torch.Tensor        # [B, 8]
    priority: torch.Tensor               # [B] continuous in [0.0, 1.0]
    alert_logits: torch.Tensor           # [B, 4]
    modality_contributions: torch.Tensor # [B, 3] normalized sum to 1.0
    context: torch.Tensor                # [B, dim] pooled latent state


class V2MultimodalTransformer(nn.Module):
    """Deep temporal cross-modal Transformer for AINAK Version 2."""

    def __init__(
        self,
        feature_dims: dict[str, int] | None = None,
        dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_steps: int = 32,
        modality_dropout: float = 0.1,
        num_scenarios: int = len(HAZARD_SCENARIOS),
        num_alert_tiers: int = len(ALERT_TIERS),
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        dims = feature_dims or DEFAULT_FEATURE_DIMS_V2
        self.dim = dim
        self.max_steps = max_steps
        self.modality_dropout = modality_dropout
        self.feature_dims = dims

        # 1. Modality-specific projections into common latent space
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(dims[m], dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for m in MODALITIES_V2
        })

        # 2. Modality & Temporal Embeddings
        self.modality_embedding = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.time_projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.Tanh(),
        )
        self.position = nn.Parameter(torch.randn(max_steps, dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )

        # 4. Multi-Task Output Heads
        self.scenario_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_scenarios),
        )

        self.priority_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        self.alert_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_alert_tiers),
        )

        # Contribution scoring layer
        self.contribution_scorer = nn.Linear(dim, 1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: torch.Tensor,
        relative_times: torch.Tensor | None = None,
    ) -> V2FusionOutput:
        """Forward pass.

        Args:
            features: Dict mapping modality name to tensor of shape [B, T, D_m]
            mask: Binary/boolean tensor [B, T, 3] where True means observed
            relative_times: Optional tensor [B, T] of relative seconds from sequence start
        """
        if mask.ndim != 3 or mask.shape[-1] != 3:
            raise ValueError(f"mask must have shape [batch, time, 3], got {mask.shape}")
        b, t, _ = mask.shape
        if t > self.max_steps:
            raise ValueError(f"sequence length {t} exceeds max_steps={self.max_steps}")

        device = mask.device
        tokens = []

        # Project each modality to common dimension D
        for idx, modality in enumerate(MODALITIES_V2):
            expected_dim = self.projections[modality][0].in_features
            if modality not in features or features[modality] is None:
                raw = torch.zeros(b, t, expected_dim, device=device)
            else:
                raw = features[modality]
                if raw.ndim != 3 or raw.shape[:2] != (b, t) or raw.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Modality '{modality}' tensor must have shape [{b}, {t}, {expected_dim}], got {raw.shape}"
                    )
            proj = self.projections[modality](raw)
            # Add learned modality embedding
            tokens.append(proj + self.modality_embedding[idx])

        # Interleaved token tensor: [B, T, 3, D]
        x = torch.stack(tokens, dim=2)

        # Add temporal and positional encodings
        if relative_times is None:
            relative_times = torch.zeros(b, t, device=device)
        time_sig = self.time_projection(relative_times.unsqueeze(-1)).unsqueeze(2)  # [B, T, 1, D]
        pos_sig = self.position[:t].view(1, t, 1, self.dim)                         # [1, T, 1, D]
        x = x + time_sig + pos_sig

        # Apply modality dropout during training
        valid = mask.bool()
        if self.training and self.modality_dropout > 0:
            keep = torch.rand_like(valid, dtype=torch.float32) >= self.modality_dropout
            # Ensure we do not drop all modalities simultaneously for an observation
            keep = keep | (valid.sum(dim=-1, keepdim=True) == 1)
            valid = valid & keep

        # Flatten sequence: [B, T * 3, D]
        flat = x.reshape(b, t * 3, self.dim)
        key_padding = ~valid.reshape(b, t * 3)

        # Handle edge case where entire batch sequence is masked
        empty = ~valid.reshape(b, -1).any(dim=1)
        if empty.any():
            flat = flat.clone()
            key_padding = key_padding.clone()
            flat[empty, 0] = 0.0
            key_padding[empty, 0] = False

        # Transformer Encoder forward pass
        encoded = self.encoder(flat, src_key_padding_mask=key_padding)
        encoded = encoded.reshape(b, t, 3, self.dim)  # [B, T, 3, D]

        # Compute dynamic cross-modal attention contributions
        valid_weights = valid.to(encoded.dtype)
        # Score each token's relevance
        token_scores = torch.exp(self.contribution_scorer(encoded).squeeze(-1)) * valid_weights
        token_scores = token_scores.sum(dim=1)  # Sum across time -> [B, 3]
        contrib_denom = token_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        modality_contributions = token_scores / contrib_denom

        # Pool contextual representation across valid tokens
        weights_sum = valid_weights.sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
        context = (encoded * valid_weights.unsqueeze(-1)).sum(dim=(1, 2)) / weights_sum
        context = context.masked_fill(empty.unsqueeze(-1), 0.0)

        # Multi-task predictions
        scenario_logits = self.scenario_head(context).masked_fill(empty.unsqueeze(-1), 0.0)
        priority = self.priority_head(context).squeeze(-1).masked_fill(empty, 0.0)
        alert_logits = self.alert_head(context).masked_fill(empty.unsqueeze(-1), 0.0)

        return V2FusionOutput(
            scenario_logits=scenario_logits,
            priority=priority,
            alert_logits=alert_logits,
            modality_contributions=modality_contributions,
            context=context,
        )
