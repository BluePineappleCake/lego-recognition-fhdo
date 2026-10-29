"""Shape classifier: a small CNN, trained from scratch (no pretrained weights).

Input:  grey image of one brick, 1 x 96 x 96
Output: one score per part (50 parts) -> softmax -> probability per part

    block 1:  32 channels, 96 x 96 -> 48 x 48
    block 2:  64 channels, 48 x 48 -> 24 x 24
    block 3: 128 channels, 24 x 24 -> 12 x 12
    block 4: 256 channels, 12 x 12 ->  6 x  6
    global average pooling -> 256 numbers ("which features are in the image")
    dropout -> linear layer -> 50 scores

About 1.2 million parameters (ResNet-18 for comparison: about 11 million).

Optional size input (size_feature=True)
---------------------------------------
Resizing every brick to 96 x 96 removes its real size, so a 1x3 and a 1x4
brick look alike. We therefore also give the network the measured size:

    z = (log(size in px) - mean) / std        (mean/std from the training set)

and add [z, z^2] to the 256 numbers before the last layer. Why also z^2?
The last layer computes for every part a score  a * z + b * z^2 + ...
With z^2 the score can be highest at the typical size of that part and get
lower the further the brick is away from it (a bell curve). With z alone the
score could only grow or shrink with the size, never peak at the right one.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from legorec.shape.augment import normalize
from legorec.shape.preprocess import preprocess_crop, preprocess_image
from legorec.types import BrickCrop


def _conv_block(in_channels, out_channels):
    """Two 3x3 convolutions (each followed by BatchNorm and ReLU), then halve
    the image size with max pooling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """See the module docstring. `width` = channels of the first block."""

    def __init__(self, n_classes, width=32, dropout=0.3, size_feature=False):
        super().__init__()
        self.size_feature = size_feature

        # NOTE: the order of these layers must not change, otherwise saved
        # checkpoints can no longer be loaded (their weights are stored by
        # position: features.0, features.1, ...).
        self.features = nn.Sequential(
            _conv_block(1, width),
            _conv_block(width, 2 * width),
            _conv_block(2 * width, 4 * width),
            _conv_block(4 * width, 8 * width),
            nn.AdaptiveAvgPool2d(1),  # average over the remaining 6 x 6 positions
            nn.Flatten(),  # -> one vector of 8 * width numbers per image
            nn.Dropout(dropout),
        )

        n_features = 8 * width
        if size_feature:
            n_features += 2  # z and z^2
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x, z=None):
        """x: images (B, 1, 96, 96); z: size feature (B,), only used with size_feature."""
        features = self.features(x)
        if self.size_feature:
            z = z.to(features.dtype)[:, None]  # (B,) -> (B, 1)
            features = torch.cat([features, z, z * z], dim=1)
        return self.fc(features)


@dataclass
class SizeNorm:
    """Turns a brick size in pixels into the size feature z.

    z = (log(size) - mu) / sd, with mu and sd from the training set, so that
    z is around 0 for a typical brick and around +-1 for big/small ones.
    """

    mu: float
    sd: float

    @classmethod
    def fit(cls, px_train):
        log_sizes = np.log(np.asarray(px_train, dtype=np.float64))
        return cls(float(log_sizes.mean()), float(log_sizes.std()))

    def __call__(self, px):
        log_sizes = np.log(np.asarray(px, dtype=np.float32))
        return ((log_sizes - self.mu) / self.sd).astype(np.float32)


class ShapeClassifier:
    """A trained model plus everything needed to use it on new images.

    Loads a checkpoint written by scripts/train_shape.py and applies the same
    preprocessing settings that were used in training (crop size, contrast).

    real_scale: converts brick sizes in photo pixels to render pixels,
        real_scale = (render pixels per mm) / (photo pixels per mm).
    Only needed for models trained with size_feature. Rough calibration: see
    scripts/predict_shape.py.
    """

    def __init__(self, checkpoint, device="cpu", real_scale=None):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        config = saved["cfg"]

        self.parts = saved["parts"]  # part numbers in the order of the outputs
        self.names = saved.get("names", {})  # part number -> readable name
        self.size = config["size"]
        self.size_feature = bool(config.get("size_feature", False))
        self.contrast = config.get("contrast", "none")  # same as in training
        self.size_norm = SizeNorm(**saved.get("size_norm", {"mu": 0.0, "sd": 1.0}))

        self.model = SmallCNN(
            len(self.parts), config["width"], config["dropout"], self.size_feature
        )
        self.model.load_state_dict(saved["model"])
        self.model.to(device).eval()
        self.device = device

        if self.size_feature and real_scale is None:
            raise ValueError(
                "this model uses the brick size: pass real_scale (see docstring)"
            )
        self.real_scale = real_scale or 1.0

    @torch.no_grad()
    def _top_k(self, arrays, sizes_px, k):
        """Preprocessed arrays + sizes -> for every image the k most likely parts."""
        images = normalize(torch.from_numpy(np.stack(arrays)).unsqueeze(1))
        sizes_in_render_px = [size * self.real_scale for size in sizes_px]
        z = torch.from_numpy(self.size_norm(sizes_in_render_px))

        scores = self.model(images.to(self.device), z.to(self.device))
        probabilities = scores.softmax(dim=1).cpu()
        top_probs, top_indices = probabilities.topk(
            min(k, probabilities.shape[1]), dim=1
        )

        results = []
        for indices, probs in zip(top_indices.tolist(), top_probs.tolist()):
            results.append([(self.parts[i], float(p)) for i, p in zip(indices, probs)])
        return results

    def predict(self, crops: list[BrickCrop]) -> list[tuple[str, float]]:
        """Bricks cut out by the segmentation -> [(part number, probability)].

        Example: [("3001", 0.93), ("3022", 0.71)]
        """
        preprocessed = [
            preprocess_crop(crop, self.size, contrast=self.contrast) for crop in crops
        ]
        arrays = [array for array, _ in preprocessed]
        sizes = [size for _, size in preprocessed]
        top_1 = self._top_k(arrays, sizes, k=1)
        return [best[0] for best in top_1]

    def predict_images(self, images, k=3):
        """Photos with ONE brick each on a plain background.

        Returns (top-k [(part number, probability)] per image,
                 the preprocessed images the model saw,
                 the measured brick sizes in pixels).
        """
        preprocessed = [
            preprocess_image(image, self.size, contrast=self.contrast)
            for image in images
        ]
        arrays = [array for array, _ in preprocessed]
        sizes = [size for _, size in preprocessed]
        return self._top_k(arrays, sizes, k), arrays, sizes
