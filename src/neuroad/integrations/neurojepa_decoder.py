"""
neurojepa_decoder — L2 Option B: a NeuroJEPA-conditioned 3D U-Net segmentation
decoder (the diagram's "U-Net Decoder / few-shot 3D biomarker segmentation").

Design, mapped 1:1 to the pipeline diagram:

  * RAW T1w MRI --> a 3D CNN encoder that produces multi-scale **anatomical skip
    connections** (the diagram's arrow). Fine anatomy the JEPA bottleneck discards
    is recovered here.
  * L1 NeuroJEPA 768-d latent --> **FiLM-conditions** the decoder bottleneck
    (per-channel affine modulation). This is the "NeuroJEPA-conditioned" part, and
    it reuses the embedding we ALREADY extract (no need to re-tap ViT internals) —
    so the decoder is trainable from (raw NIfTI, 768-d embedding, FastSurfer label)
    triples.
  * DECODER --> up-samples with skip connections to a voxel-wise segmentation, from
    which exact hippocampal / ventricular / cortical **volumes** are read (mm^3),
    the same VOLUME_KEYS the contract/FastSurfer path already uses.

Torch is imported at module load, so this module is intentionally NOT exported from
``integrations/__init__`` — the offline test suite never imports it. It is used by
``scripts/train_neurojepa_decoder.py`` (GPU) and validated with a forward-pass shape
check. Nothing here fabricates volumes: at inference a class is a voxel count times
the voxel volume, both real.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Segmentation label set — compact, matches VOLUME_KEYS aggregation. FastSurfer
# aseg labels are remapped into these at data-prep time.
LABELS = (
    "background",          # 0
    "left_hippocampus",    # 1
    "right_hippocampus",   # 2
    "ventricle",           # 3  (lateral+inf-lat+3rd+4th merged)
    "cortex",              # 4  (cortical gray matter)
    "other_brain",         # 5  (remaining brain tissue)
)
N_CLASSES = len(LABELS)
JEPA_DIM = 768


def _conv_block(cin, cout):
    """Two 3x3x3 convs + GroupNorm + GELU — the U-Net double-conv."""
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout), nn.GELU(),
    )


class RawMRIEncoder(nn.Module):
    """3D CNN over the raw T1w -> bottleneck + 4 skip feature maps (fine->coarse)."""

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc0 = _conv_block(1, c[0])
        self.enc1 = _conv_block(c[0], c[1])
        self.enc2 = _conv_block(c[1], c[2])
        self.enc3 = _conv_block(c[2], c[3])
        self.bottleneck = _conv_block(c[3], c[4])
        self.pool = nn.MaxPool3d(2)
        self.skip_channels = c[:4]
        self.bottleneck_channels = c[4]

    def forward(self, x):
        s0 = self.enc0(x)                 # full res
        s1 = self.enc1(self.pool(s0))     # /2
        s2 = self.enc2(self.pool(s1))     # /4
        s3 = self.enc3(self.pool(s2))     # /8
        b = self.bottleneck(self.pool(s3))  # /16
        return b, [s0, s1, s2, s3]


class FiLM(nn.Module):
    """FiLM: 768-d NeuroJEPA embedding -> per-channel (gamma, beta) for a feature map."""

    def __init__(self, jepa_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(jepa_dim, channels), nn.GELU(),
            nn.Linear(channels, channels * 2))

    def forward(self, feat, jepa):
        gb = self.net(jepa)                       # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)          # [B, C] each
        shape = (feat.size(0), feat.size(1), 1, 1, 1)
        return feat * (1 + gamma.view(shape)) + beta.view(shape)


class UpBlock(nn.Module):
    """Trilinear up + concat skip + double conv."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = nn.Conv3d(cin, cout, 1)
        self.conv = _conv_block(cout + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                          align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class NeuroJEPADecoder(nn.Module):
    """Raw-MRI 3D U-Net whose bottleneck is FiLM-conditioned by the L1 768-d latent.

    forward(mri [B,1,D,H,W], jepa [B,768]) -> logits [B, N_CLASSES, D,H,W].
    """

    def __init__(self, base=16, jepa_dim=JEPA_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = RawMRIEncoder(base)
        skip = self.encoder.skip_channels           # [c0,c1,c2,c3]
        bott = self.encoder.bottleneck_channels     # c4
        self.film = FiLM(jepa_dim, bott)
        self.up3 = UpBlock(bott, skip[3], skip[3])
        self.up2 = UpBlock(skip[3], skip[2], skip[2])
        self.up1 = UpBlock(skip[2], skip[1], skip[1])
        self.up0 = UpBlock(skip[1], skip[0], skip[0])
        self.head = nn.Conv3d(skip[0], n_classes, 1)

    def forward(self, mri, jepa):
        b, skips = self.encoder(mri)
        b = self.film(b, jepa)                      # NeuroJEPA conditioning
        x = self.up3(b, skips[3])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[1])
        x = self.up0(x, skips[0])
        return self.head(x)


# ---------------------------------------------------------------------------
# Losses + volume read-out
# ---------------------------------------------------------------------------


def dice_ce_loss(logits, target, *, ce_w=0.5, eps=1e-6):
    """Combined Dice + cross-entropy over the segmentation logits.

    logits [B,C,D,H,W]; target [B,D,H,W] int64 class indices. Standard 3D-seg loss.
    """
    ce = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=1)
    tgt1h = F.one_hot(target, probs.size(1)).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (probs * tgt1h).sum(dims)
    denom = probs.sum(dims) + tgt1h.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return ce_w * ce + (1 - ce_w) * (1 - dice.mean())


@dataclass
class DecoderVolumes:
    hippocampal_volume: float
    ventricle_volume: float
    cortex_volume: float
    source: str = "neurojepa_unet"

    def to_dict(self):
        return {"hippocampal_volume": self.hippocampal_volume,
                "ventricle_volume": self.ventricle_volume,
                "cortex_volume": self.cortex_volume, "source": self.source}


@torch.no_grad()
def volumes_from_logits(logits, voxel_mm3: float) -> DecoderVolumes:
    """Argmax the seg and convert per-class voxel counts to mm^3 volumes.

    A volume is a real voxel count times the real voxel volume — never fabricated.
    Expects a single-example logits [1,C,D,H,W]. Bilateral hippocampus = L+R.
    """
    seg = logits.argmax(dim=1)[0]                    # [D,H,W]
    def vox(cls):  # noqa: E306
        return int((seg == cls).sum().item())
    hip = (vox(1) + vox(2)) * voxel_mm3
    ven = vox(3) * voxel_mm3
    cor = vox(4) * voxel_mm3
    return DecoderVolumes(round(hip, 2), round(ven, 2), round(cor, 2))


__all__ = ["LABELS", "N_CLASSES", "JEPA_DIM", "NeuroJEPADecoder",
           "RawMRIEncoder", "FiLM", "dice_ce_loss", "DecoderVolumes",
           "volumes_from_logits"]
