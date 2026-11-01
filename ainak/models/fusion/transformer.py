from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from ...data.schemas import ALERT_TYPES

MODALITIES = ("vision", "audio", "gesture")
DEFAULT_FEATURE_DIMS = {"vision": 128, "audio": 2048, "gesture": 128}


@dataclass
class FusionOutput:
    context: torch.Tensor
    priority: torch.Tensor
    confidence: torch.Tensor
    alert_logits: torch.Tensor
    modality_contributions: torch.Tensor


class MultimodalTransformer(nn.Module):
    """2-layer, 4-head temporal encoder; a time step has one token per modality."""
    def __init__(self, feature_dims: dict[str, int] | None = None, dim: int = 128, heads: int = 4,
                 layers: int = 2, ff_dim: int = 256, dropout: float = .1, max_steps: int = 128, modality_dropout: float = .1):
        super().__init__()
        if dim % heads: raise ValueError("embedding dimension must be divisible by attention heads")
        dims = feature_dims or DEFAULT_FEATURE_DIMS
        self.projections = nn.ModuleDict({m: nn.Sequential(nn.Linear(dims[m], dim), nn.LayerNorm(dim)) for m in MODALITIES})
        self.modality_embedding = nn.Parameter(torch.randn(3, dim) * .02)
        self.time_projection = nn.Sequential(nn.Linear(1, dim), nn.Tanh())
        self.position = nn.Parameter(torch.randn(max_steps, dim) * .02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.priority_head = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim//2, 1), nn.Sigmoid())
        self.confidence_head = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim//2, 1), nn.Sigmoid())
        self.alert_head = nn.Linear(dim, len(ALERT_TYPES))
        if not 0 <= modality_dropout <= 1: raise ValueError("modality_dropout must be in [0,1]")
        self.dim, self.max_steps, self.modality_dropout = dim, max_steps, modality_dropout

    def forward(self, features: dict[str, torch.Tensor], mask: torch.Tensor, relative_times: torch.Tensor | None = None) -> FusionOutput:
        """features: modality [B,T,D]; mask true means observed, shape [B,T,3]."""
        if mask.ndim != 3 or mask.shape[-1] != 3: raise ValueError("mask must have shape [batch,time,3]")
        b, t, _ = mask.shape
        if t > self.max_steps: raise ValueError(f"sequence length {t} exceeds max_steps={self.max_steps}")
        tokens = []
        for idx, modality in enumerate(MODALITIES):
            if modality not in features:
                raw = torch.zeros(b, t, self.projections[modality][0].in_features, device=mask.device)
            else:
                raw = features[modality]
                expected=self.projections[modality][0].in_features
                if raw.ndim != 3 or raw.shape[:2] != (b,t) or raw.shape[-1] != expected:
                    raise ValueError(f"{modality} must have shape [batch,time,{expected}]")
            token = self.projections[modality](raw)
            tokens.append(token + self.modality_embedding[idx])
        x = torch.stack(tokens, dim=2)  # B,T,M,D
        if relative_times is None: relative_times = torch.zeros(b,t,device=mask.device)
        if relative_times.shape != (b,t): raise ValueError("relative_times must have shape [batch,time]")
        time_signal = self.time_projection(relative_times.unsqueeze(-1)).unsqueeze(2)
        x = x + time_signal + self.position[:t].view(1,t,1,self.dim)
        valid = mask.bool()
        if self.training and self.modality_dropout:
            keep = torch.rand_like(valid, dtype=torch.float32) >= self.modality_dropout
            valid = valid & keep
        flat = x.reshape(b, t*3, self.dim); key_padding = ~valid.reshape(b, t*3)
        empty = ~valid.reshape(b,-1).any(dim=1)
        # Transformer attention cannot consume a row with all keys masked; use a temporary zero key then zero its output.
        if empty.any():
            flat = flat.clone(); key_padding = key_padding.clone()
            flat[empty, 0] = 0.; key_padding[empty, 0] = False
        encoded = self.encoder(flat, src_key_padding_mask=key_padding)
        encoded = encoded.reshape(b,t,3,self.dim)
        weights = valid.to(encoded.dtype); denom = weights.sum(dim=(1,2), keepdim=False).clamp_min(1).unsqueeze(-1)
        context = (encoded * weights.unsqueeze(-1)).sum(dim=(1,2)) / denom
        context = context.masked_fill(empty.unsqueeze(-1), 0.)
        contribution = weights.sum(dim=1) / weights.sum(dim=(1,2)).clamp_min(1).unsqueeze(-1)
        priority=self.priority_head(context).squeeze(-1).masked_fill(empty,0.)
        confidence=self.confidence_head(context).squeeze(-1).masked_fill(empty,0.)
        alert_logits=self.alert_head(context).masked_fill(empty.unsqueeze(-1),0.)
        return FusionOutput(context=context, priority=priority,
            confidence=confidence, alert_logits=alert_logits,
            modality_contributions=contribution)
