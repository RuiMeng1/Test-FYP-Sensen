"""
reid_model.py — Person Re-Identification feature encoder.

Model hierarchy (best to fallback):

  1. PersonViT-ReID ViT-Small (MSMT17)   [DEFAULT — already downloaded]
        A ViT-Small fine-tuned specifically for person Re-ID on MSMT17.
        1041 training identities, BN-Neck, normalized 384-dim embeddings.
        Much better than generic models: same-person sim ≥ 0.85.
        Location: models/personvit_reid/
        Source:   https://huggingface.co/maennyn/personvit-reid-msmt17-vit-s

  2. DINOv2 ViT-Small   [fallback if models/ is missing]
        Self-supervised contrastive learning. Not ReID-trained, but still
        significantly better than plain ImageNet ViT for similarity tasks.
        Auto-downloads via timm/HuggingFace on first use (~85 MB).

The model runs on CPU (thread-safe — see REID_DEVICE in config.py).
"""

from __future__ import annotations


import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


# ── Transforms ────────────────────────────────────────────────────────────────

# PersonViT was trained at 256×128 (H×W) — exactly the standard ReID crop size
_REID_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),          # PersonViT native size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DINOv2 fallback uses 224×224
_DINO_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TransReIDEncoder:
    """
    Person Re-ID feature extractor.  Thread-safe.

    Uses PersonViT (proper ReID model trained on MSMT17) by default.
    Falls back to DINOv2-ViT-Small if the model files are missing.
    """

    PERSONVIT_DIR = Path(__file__).parent.parent / "models" / "personvit_reid"

    def __init__(self, weights_path: Optional[Path] = None, device: str = "cpu"):
        self.device  = device
        self._lock   = threading.Lock()
        self._transform = _REID_TRANSFORM

        if self.PERSONVIT_DIR.exists() and (self.PERSONVIT_DIR / "model.safetensors").exists():
            self._model, self.FEAT_DIM = self._load_personvit()
        else:
            print("[ReID] PersonViT model not found — falling back to DINOv2 ViT-Small")
            self._model, self.FEAT_DIM = self._load_dinov2()
            self._transform = _DINO_TRANSFORM

        self._model.eval()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_personvit(self):
        """Load PersonViT-ReID ViT-Small trained on MSMT17 (proper ReID model)."""
        print("[ReID] Loading PersonViT-ReID ViT-Small (MSMT17, 384-dim) …")
        from transformers import AutoModel, AutoConfig

        cfg   = AutoConfig.from_pretrained(str(self.PERSONVIT_DIR), trust_remote_code=True)
        model = AutoModel.from_pretrained(
            str(self.PERSONVIT_DIR),
            trust_remote_code=True,
        )
        model = model.to(self.device)
        feat_dim = cfg.embedding_dim   # 384
        print(f"[ReID] PersonViT-ReID ready  (feat_dim={feat_dim}, MSMT17-trained ✓)")
        return model, feat_dim

    def _load_dinov2(self):
        """Fallback: DINOv2-ViT-Small (self-supervised, auto-downloads ~85 MB)."""
        import timm
        print("[ReID] Loading DINOv2 ViT-Small fallback …")
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        model = model.to(self.device)
        feat_dim = model.embed_dim   # 384
        print(f"[ReID] DINOv2 ViT-Small ready  (feat_dim={feat_dim})")
        return model, feat_dim

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward(self, tensor: torch.Tensor) -> np.ndarray:
        """Run the model and return unit-norm embeddings as numpy array."""
        tensor = tensor.to(self.device)
        with self._lock:
            out = self._model(tensor)

        # PersonViT returns a tuple: (embeddings [B,384], hidden_states [B,N,384])
        # DINOv2/timm returns a raw tensor [B, D]
        if isinstance(out, tuple):
            feats = out[0]          # PersonViT: [B, 384] — already L2-normalised
        elif isinstance(out, dict):
            feats = out.get("embeddings", out.get("last_hidden_state"))
            if feats is not None and feats.dim() == 3:
                feats = feats[:, 0]     # CLS token
        else:
            feats = out             # timm raw tensor

        # L2-normalise (PersonViT already is, but safe to re-normalise)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a unit-norm feature vector from a single BGR person crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.FEAT_DIM, dtype=np.float32)
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        tensor   = self._transform(crop_rgb).unsqueeze(0)
        return self._forward(tensor)[0].astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Extract unit-norm feature vectors for a list of BGR person crops."""
        if not crops_bgr:
            return np.empty((0, self.FEAT_DIM), dtype=np.float32)
        tensors = []
        for crop in crops_bgr:
            if crop is None or crop.size == 0:
                tensors.append(torch.zeros(3, *self._transform.transforms[1].size))
            else:
                crop_rgb = crop[:, :, ::-1].copy()
                tensors.append(self._transform(crop_rgb))
        batch = torch.stack(tensors)
        return self._forward(batch).astype(np.float32)
