"""Multi-task FractalCNN with uncertainty heads.

Differences from :class:`FractalCNN`:

* **Three heads** sharing the backbone: family classification, Julia c
  regression, and Multibrot exponent regression.
* **Mean-Variance Estimation (MVE) heads** for both regression tasks: each
  predicts a Gaussian ``(μ, log σ²)`` per output dimension. Uncertainty is
  intrinsic to inverse fractal inference — multiple parameter values can
  produce visually similar images, and a model that can say "I don't know
  which c this is" is more honest than one that always commits to a point
  estimate. The training loss is Gaussian NLL; at inference we surface
  ``σ`` as a calibration signal.
* **Backbone unchanged.** GroupNorm + small ResNet stack — same as v1, so
  v1 hyperparameters transfer cleanly.

The Tanh × 1.5 trick from v1 is dropped from the c-head because the MVE
formulation already covers numerical stability via the ``log σ²`` clamping
in the loss; clamping the mean as well over-constrains the regression on
hard_overlap scenarios where targets sit near the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cnn import _ResidualBlock, _norm


@dataclass
class ModelOutputV2:
    family_logits: torch.Tensor       # (B, num_families)
    c_mean: torch.Tensor              # (B, 2) — predicted (cRe, cIm) mean
    c_log_var: torch.Tensor           # (B, 2) — log σ² per output
    exp_mean: torch.Tensor            # (B,) — predicted multibrot exponent mean
    exp_log_var: torch.Tensor         # (B,) — log σ² for exponent
    viewport_mean: torch.Tensor       # (B, 4) — μ for (xMin, xMax, yMin, yMax)
    viewport_log_var: torch.Tensor    # (B, 4) — log σ² per viewport bound
    embedding: torch.Tensor           # (B, embedding_dim) L2-normalised; unit sphere


class FractalCNNv2(nn.Module):
    def __init__(
        self,
        num_families: int = 4,
        base_channels: int = 32,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=False),
            _norm(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = _ResidualBlock(c1, c2, stride=2)
        self.layer2 = _ResidualBlock(c2, c3, stride=2)
        self.layer3 = _ResidualBlock(c3, c4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.family_head = nn.Linear(c4, num_families)

        # Julia c head — 4 outputs: (μ_re, μ_im, log σ²_re, log σ²_im).
        self.c_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

        # Multibrot exponent head — 2 outputs: (μ, log σ²). Discrete target
        # but the regression formulation lets us reuse the same MVE machinery
        # and emit calibration-aware predictions instead of a hard argmax.
        self.exp_head = nn.Sequential(
            nn.Linear(c4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        # Viewport head — 8 outputs: 4 means (xMin, xMax, yMin, yMax) +
        # 4 log-variances. Always-on: every example carries a viewport, so
        # this head is unmasked and gradient is computed for every batch.
        # The motivation is reconstruction: Mandelbrot and Burning Ship have
        # no recipe-internal continuous parameter to refine, so the only
        # leverage the inverse pipeline has is over the viewport bounds.
        self.viewport_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )

        # Embedding head — projects backbone features onto the unit sphere
        # via the standard SimCLR/SupCon "projection MLP" pattern (Linear →
        # ReLU → Linear → L2 normalise). The downstream supervised
        # contrastive loss expects unit-norm inputs; doing the normalise
        # inside the model keeps every consumer (training, retrieval bank
        # build, retrieval query) on the same numerics without each
        # remembering to call ``F.normalize``.
        self.embedding_head = nn.Sequential(
            nn.Linear(c4, c4),
            nn.ReLU(inplace=True),
            nn.Linear(c4, embedding_dim),
        )

        self._num_families = num_families
        self._base_channels = base_channels
        self._embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> ModelOutputV2:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        feat = self.flatten(x)

        family_logits = self.family_head(feat)
        c_out = self.c_head(feat)
        exp_out = self.exp_head(feat)
        vp_out = self.viewport_head(feat)
        emb = self.embedding_head(feat)
        emb = nn.functional.normalize(emb, p=2.0, dim=1)

        return ModelOutputV2(
            family_logits=family_logits,
            c_mean=c_out[:, :2],
            c_log_var=c_out[:, 2:],
            exp_mean=exp_out[:, 0],
            exp_log_var=exp_out[:, 1],
            viewport_mean=vp_out[:, :4],
            viewport_log_var=vp_out[:, 4:],
            embedding=emb,
        )

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward + return only the embedding. Used by the
        retrieval-bank builder which doesn't care about the other heads."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(x).embedding
        finally:
            if was_training:
                self.train()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def base_channels(self) -> int:
        return self._base_channels

    @property
    def num_families(self) -> int:
        return self._num_families

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
