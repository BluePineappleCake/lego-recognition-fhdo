"""Shape classifier: small CNN trained from scratch (no pretrained weights).

Optional size input: the brick size is fed in as z = standardised log(px).
[z, z^2] is concatenated to the CNN features; a linear layer on (z, z^2) per
class is a Gaussian log-likelihood of the brick's size for that class.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from legorec.shape.augment import normalize
from legorec.shape.preprocess import preprocess_crop
from legorec.types import BrickCrop


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks + global average pooling, ~1.2 M parameters at width 32."""

    def __init__(self, n_classes, width=32, dropout=0.3, size_feature=False):
        super().__init__()
        w = width
        self.size_feature = size_feature
        self.features = nn.Sequential(
            _block(1, w),
            _block(w, 2 * w),
            _block(2 * w, 4 * w),
            _block(4 * w, 8 * w),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(8 * w + (2 if size_feature else 0), n_classes)

    def forward(self, x, z=None):
        f = self.features(x)
        if self.size_feature:
            z = z.to(f.dtype)[:, None]
            f = torch.cat([f, z, z * z], 1)
        return self.fc(f)


@dataclass
class SizeNorm:
    """log(px) standardised with the statistics of the training set."""

    mu: float
    sd: float

    @classmethod
    def fit(cls, px_train):
        lp = np.log(np.asarray(px_train, np.float64))
        return cls(float(lp.mean()), float(lp.std()))

    def __call__(self, px):
        return ((np.log(np.asarray(px, np.float32)) - self.mu) / self.sd).astype(
            np.float32
        )


class ShapeClassifier:
    """Trained model + everything needed to use it on new crops.

    real_scale converts sizes in photo pixels to render pixels:
    real_scale = (render px per mm) / (photo px per mm). Only used with size_feature.
    """

    def __init__(self, checkpoint, device="cpu", real_scale=None):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        self.parts, self.size = ck["parts"], cfg["size"]
        self.size_norm = SizeNorm(**ck["size_norm"])
        self.model = SmallCNN(
            len(self.parts), cfg["width"], cfg["dropout"], cfg["size_feature"]
        )
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device
        if cfg["size_feature"] and real_scale is None:
            raise ValueError(
                "this model uses the brick size: pass real_scale (see docstring)"
            )
        self.real_scale = real_scale or 1.0

    @torch.no_grad()
    def predict(self, crops: list[BrickCrop]) -> list[tuple[str, float]]:
        """-> [(part_id, confidence)] per crop, e.g. [("3001", 0.93)]."""
        res = [preprocess_crop(c, self.size) for c in crops]
        x = normalize(torch.from_numpy(np.stack([r[0] for r in res])).unsqueeze(1))
        z = torch.from_numpy(self.size_norm([r[1] * self.real_scale for r in res]))
        p = self.model(x.to(self.device), z.to(self.device)).softmax(1).cpu()
        conf, idx = p.max(1)
        return [(self.parts[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]
