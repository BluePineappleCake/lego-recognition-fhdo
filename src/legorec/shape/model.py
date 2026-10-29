"""Shape classifier: small CNN trained from scratch (no pretrained weights).

Optional size input (size_feature=True)
---------------------------------------
Resizing every crop to 96 x 96 removes the absolute size, so a 1x3 and a 1x4
brick look alike. We therefore also give the network the measured size:

    z = (log(size in px) - mean) / std        (standardised on the training set)

and append [z, z^2] to the CNN features before the last layer. Why z^2? The
last layer computes for every class c a score  a_c * z + b_c * z^2 + ...
A quadratic in z can describe "class c has a typical size and gets less likely
the further z is away from it" (a bell curve in log space). With z alone the
score could only grow or shrink with size, never peak at the right size.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from legorec.shape.augment import normalize
from legorec.shape.preprocess import preprocess_crop, preprocess_image
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
        if self.size_feature:  # append [z, z^2], see module docstring
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
    """Trained model + everything needed to use it on new images.

    real_scale converts sizes in photo pixels to render pixels:
    real_scale = (render px per mm) / (photo px per mm). Only needed for models
    trained with size_feature. See scripts/predict_shape.py for a rough calibration.
    """

    def __init__(self, checkpoint, device="cpu", real_scale=None):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        self.parts, self.size = ck["parts"], cfg["size"]
        self.names = ck.get("names", {})
        self.size_feature = bool(cfg.get("size_feature", False))
        norm = ck.get("size_norm", {"mu": 0.0, "sd": 1.0})  # older checkpoints
        self.size_norm = SizeNorm(**norm)
        self.model = SmallCNN(
            len(self.parts), cfg["width"], cfg["dropout"], self.size_feature
        )
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.device = device
        if self.size_feature and real_scale is None:
            raise ValueError(
                "this model uses the brick size: pass real_scale (see docstring)"
            )
        self.real_scale = real_scale or 1.0

    @torch.no_grad()
    def _topk(self, arrays, pxs, k):
        x = normalize(torch.from_numpy(np.stack(arrays)).unsqueeze(1))
        z = torch.from_numpy(self.size_norm([p * self.real_scale for p in pxs]))
        p = self.model(x.to(self.device), z.to(self.device)).softmax(1).cpu()
        conf, idx = p.topk(min(k, p.shape[1]), 1)
        return [
            [(self.parts[i], float(c)) for i, c in zip(ii, cc)]
            for ii, cc in zip(idx.tolist(), conf.tolist())
        ]

    def predict(self, crops: list[BrickCrop]) -> list[tuple[str, float]]:
        """Segmentation crops -> [(part_id, confidence)], e.g. [("3001", 0.93)]."""
        res = [preprocess_crop(c, self.size) for c in crops]
        top = self._topk([r[0] for r in res], [r[1] for r in res], 1)
        return [t[0] for t in top]

    def predict_images(self, images, k=3):
        """Photos with ONE brick on a plain background.

        -> (top-k [(part_id, confidence)] per image, model inputs, sizes in px)
        """
        res = [preprocess_image(img, self.size) for img in images]
        arrays, pxs = [r[0] for r in res], [r[1] for r in res]
        return self._topk(arrays, pxs, k), arrays, pxs
