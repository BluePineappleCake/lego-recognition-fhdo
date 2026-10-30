"""Train the shape classifier on the Kaggle renders and log everything to W&B.

    python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
    python scripts/train_shape.py --config configs/shape_baseline.yaml
    python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
    python scripts/train_shape.py --config configs/shape_baseline.yaml --data C:\\lego\\data

--data: folder that directly contains the 40,000 PNGs (default: see legorec.paths).
To try a trained model on real photos, use scripts/predict_shape.py.

What happens, in order:
    1. read the config, load (or build) the preprocessed renders
    2. start a W&B run and log a few training images
    3. train for N epochs; after every epoch check the validation set and keep
       the checkpoint with the best validation macro-F1
    4. load that best checkpoint and evaluate it once on the render test set
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch import nn

from legorec.data.renders import build_cache, index_renders, load_or_create_split
from legorec.eval.metrics import classification_metrics, error_breakdown
from legorec.paths import MODELS_DIR, processed_dir, renders_dir
from legorec.shape.augment import augment_batch, denormalize
from legorec.shape.loaders import Batches
from legorec.shape.model import SizeNorm, SmallCNN
from legorec.shape.preprocess import PREPROCESS_VERSION

# Values used when the config file does not set them
DEFAULTS = {
    "size": 96,  # side length of the brick images in pixels
    "width": 32,  # channels of the first conv block (see model.py)
    "dropout": 0.3,
    "size_feature": False,  # also give the network the measured brick size
    "contrast": "none",  # "stretch": stretch the contrast per brick (preprocess.py)
    "size_noise": 0.05,  # about 5 % noise on that size during training
    "epochs": 30,  # how often the whole training set is used
    "batch_size": 128,  # images per training step
    "lr": 3e-3,  # (maximum) learning rate
    "weight_decay": 5e-4,  # keeps the weights small, against overfitting
    "label_smoothing": 0.1,  # target 0.9 instead of 1.0 for the right class
    "hflip": False,  # mirror images during training (only without mirror-twin parts)
    "seed": 0,  # random seed: same seed + same code = same result
    "project": "lego-recognition",  # W&B project
    "tags": ["renders-only"],
}

EVAL_BATCH_SIZE = 512  # no training during evaluation, so bigger batches are fine


# --------------------------------------------------------------------------
# 1. Config and data
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sanity", action="store_true", help="overfit one batch only")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = DEFAULTS | yaml.safe_load(file)  # the config file wins
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    return args, config


def set_seed(seed):
    """Make the run repeatable: same seed and same code -> same result."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(config, data_dir):
    """Renders -> split by pose -> preprocessed arrays (cached on disk).

    Returns
        data:      {"train"/"val"/"test": (images, size feature z, labels)}
        parts:     part numbers; label i means parts[i]
        names:     part number -> readable name
        size_norm: turns brick sizes in pixels into the size feature z
        split_hash: fingerprint of the split
    """
    images = index_renders(data_dir or renders_dir())
    parts = sorted({image.part for image in images})
    label_of_part = {part: label for label, part in enumerate(parts)}
    names = {image.part: image.name for image in images}
    images_per_split, split_hash = load_or_create_split(images)

    # Every setting that changes the preprocessed images must be part of the
    # cache file name, otherwise an old cache would be reused without warning.
    cache_tag = (
        f"{config['size']}_{PREPROCESS_VERSION}_{config['contrast']}_{split_hash}"
    )

    raw = {}
    for split in ("train", "val", "test"):
        cache_file = processed_dir() / f"renders_{split}_{cache_tag}.npz"
        x, sizes_px = build_cache(
            images_per_split[split], config["size"], cache_file, config["contrast"]
        )
        labels = [label_of_part[image.part] for image in images_per_split[split]]
        raw[split] = (x, sizes_px, labels)
        print(f"[data] {split}: {len(labels)} images")

    # The size feature is standardised with the TRAINING set only. Using val or
    # test statistics would leak information about them into the model.
    train_sizes = raw["train"][1]
    size_norm = SizeNorm.fit(train_sizes)

    data = {}
    for split, (x, sizes_px, labels) in raw.items():
        data[split] = (x, size_norm(sizes_px), labels)
    return data, parts, names, size_norm, split_hash


# --------------------------------------------------------------------------
# 2. Logging
# --------------------------------------------------------------------------


def start_wandb_run(config, args, split_hash, n_classes):
    tags = list(config["tags"])
    if config["size_feature"]:
        tags.append("size-feature")

    logged_config = config | {
        "split_hash": split_hash,
        "n_classes": n_classes,
        "preprocess": PREPROCESS_VERSION,
    }
    return wandb.init(
        project=config["project"],
        config=logged_config,
        tags=tags,
        job_type="sanity" if args.sanity else "train",
    )


def log_sample_images(data, device, config, parts, names):
    """Log 16 augmented training images: exactly what the network sees."""
    x, _, labels = next(iter(Batches(*data["train"], device, 16, shuffle=True)))
    augmented = denormalize(augment_batch(x, config["hflip"]))
    as_uint8 = (augmented.cpu().numpy() * 255).astype(np.uint8)

    samples = []
    for image, label in zip(as_uint8, labels):
        samples.append(wandb.Image(image[0], caption=names[parts[int(label)]]))
    wandb.log({"samples/train": samples})


# --------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------


def train_epoch(
    model, batches, optimizer, scheduler, loss_fn, config, size_norm, augment=True
):
    """One pass over the training batches. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, n_correct, n_images = 0.0, 0, 0

    for x, z, labels in batches:
        if augment:
            x = augment_batch(x, config["hflip"])
            # we will never measure the size perfectly on real photos, so the
            # model should not rely on it too much: add a little noise
            z = z + torch.randn_like(z) * config["size_noise"] / size_norm.sd

        scores = model(x, z)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()  # compute how each weight should change
        optimizer.step()  # change the weights
        scheduler.step()  # adjust the learning rate (OneCycle: up, then down)

        total_loss += loss.item() * len(labels)
        n_correct += (scores.argmax(dim=1) == labels).sum().item()
        n_images += len(labels)

    return total_loss / n_images, n_correct / n_images


@torch.no_grad()
def predict(model, batches):
    """Run the model without training. Returns (scores (N, C), labels (N,), loss)."""
    model.eval()
    all_scores, all_labels = [], []
    for x, z, labels in batches:
        all_scores.append(model(x, z).cpu())
        all_labels.append(labels.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    loss = F.cross_entropy(scores, labels).item()
    return scores.numpy(), labels.numpy(), loss


def run_sanity_check(
    model, one_batch, optimizer, scheduler, loss_fn, config, size_norm
):
    """A correct model and training loop must be able to memorise one batch.

    If the loss does not go close to 0 here, something is broken (labels,
    model, optimizer), and a full training run would be wasted time.
    """
    train_epoch(
        model,
        one_batch,
        optimizer,
        scheduler,
        loss_fn,
        config,
        size_norm,
        augment=False,
    )
    x, z, labels = one_batch[0]
    with torch.no_grad():
        loss = loss_fn(model(x, z), labels).item()
    print(f"[sanity] loss on that batch after training {loss:.4f} (should be ~0)")


def save_checkpoint(path, model, parts, names, config, split_hash, size_norm):
    """Everything needed to use the model later (see ShapeClassifier)."""
    torch.save(
        {
            "model": model.state_dict(),
            "parts": parts,
            "names": names,
            "cfg": config,
            "split_hash": split_hash,
            "size_norm": {"mu": size_norm.mu, "sd": size_norm.sd},
        },
        path,
    )


# --------------------------------------------------------------------------
# 4. Final test
# --------------------------------------------------------------------------


def evaluate_on_test(model, test, parts, names):
    """Evaluate the best checkpoint once on the render test set and log it."""
    scores, labels, _ = predict(model, test)
    predictions = scores.argmax(axis=1)
    metrics = classification_metrics(scores, labels)
    metrics |= error_breakdown(labels, predictions, parts)

    wandb.run.summary.update({f"render_test/{key}": v for key, v in metrics.items()})
    confusion = wandb.plot.confusion_matrix(
        y_true=labels.tolist(),
        preds=predictions.tolist(),
        class_names=[names[part] for part in parts],
    )
    wandb.log({"render_test/confusion": confusion})
    print(
        f"[render test] top1 {metrics['top1']:.4f}  top3 {metrics['top3']:.4f}  "
        f"length errors {metrics['errors_length']}  "
        f"other errors {metrics['errors_other']}"
    )
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    args, config = parse_args()
    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. data
    data, parts, names, size_norm, split_hash = load_data(config, args.data)
    train = Batches(*data["train"], device, config["batch_size"], shuffle=True)
    val = Batches(*data["val"], device, EVAL_BATCH_SIZE)
    test = Batches(*data["test"], device, EVAL_BATCH_SIZE)

    # 2. logging
    run = start_wandb_run(config, args, split_hash, len(parts))
    print(f"[run] device {device}, size feature {config['size_feature']}")
    log_sample_images(data, device, config, parts, names)

    # 3. model, optimizer, learning-rate schedule, loss
    model = SmallCNN(
        len(parts), config["width"], config["dropout"], config["size_feature"]
    ).to(device)
    run.summary["params"] = sum(p.numel() for p in model.parameters())

    if args.sanity:
        one_batch = [next(iter(Batches(*data["train"], device, 32, shuffle=True)))]
        train = one_batch * 50  # the same batch 50 times
        config["epochs"] = 1
        config["label_smoothing"] = 0.0  # otherwise the loss cannot reach 0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    total_steps = config["epochs"] * len(train)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config["lr"], total_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    if args.sanity:
        run_sanity_check(model, train, optimizer, scheduler, loss_fn, config, size_norm)
        run.finish()
        return

    # training: keep the checkpoint with the best validation macro-F1
    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint = MODELS_DIR / f"shape_{run.id}.pt"
    best_val_f1 = -1.0

    for epoch in range(config["epochs"]):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train, optimizer, scheduler, loss_fn, config, size_norm
        )
        val_scores, val_labels, val_loss = predict(model, val)
        val_metrics = classification_metrics(val_scores, val_labels)
        seconds = time.time() - start

        wandb.log(
            {"epoch": epoch, "train/loss": train_loss, "train/top1": train_acc}
            | {f"val/{key}": value for key, value in val_metrics.items()}
            | {"val/loss": val_loss, "lr": scheduler.get_last_lr()[0]}
            | {"epoch_time_s": seconds}
        )
        print(
            f"ep {epoch:02d}  train {train_loss:.3f}/{train_acc:.3f}  "
            f"val {val_loss:.3f}/{val_metrics['top1']:.3f}  ({seconds:.1f}s)"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                checkpoint, model, parts, names, config, split_hash, size_norm
            )

    # 4. final test with the best checkpoint (not the last epoch)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    test_metrics = evaluate_on_test(model, test, parts, names)

    artifact = wandb.Artifact("shape-cnn", type="model", metadata=test_metrics)
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)
    run.finish()
    print(f"[done] model saved to {checkpoint}")


if __name__ == "__main__":
    main()
