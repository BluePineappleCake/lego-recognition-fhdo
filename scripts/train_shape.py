"""Train the shape classifier and log everything to W&B.

    python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
    python scripts/train_shape.py --config configs/shape_baseline.yaml
    python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
    python scripts/train_shape.py --config configs/shape_baseline.yaml --data C:\\lego\\data
    python scripts/train_shape.py --config configs/shape_contrast_internet.yaml
    python scripts/train_shape.py --config configs/shape_internet_own.yaml

--data: folder that directly contains the 40,000 Kaggle PNGs (default: see
    legorec.paths).
--internet-data: folder with photos/ and renders/ of the internet dataset
    (default: data/raw/internet). If photos/ is missing and not in train_on,
    it is skipped.
All settings and their defaults: TrainConfig in src/legorec/shape/train.py.
To try a trained model on real photos, use scripts/eval_own.py.

What happens, in order:
    1. read the config, load (or build) the preprocessed images of every
       source in train_on, and the internet photos for evaluation
    2. start a W&B run and log a few training images
    3. train for N epochs; after every epoch check the val split of every
       source (Kaggle renders as val/*, internet photos as internet_val/*,
       others as <source>_val/*) and keep the checkpoint with the best
       macro-F1 on the val split of select_on
    4. load that best checkpoint and evaluate it once on the test splits of
       the render sources in train_on, and on internet/val. The test splits
       of the photos are never touched here (scripts/eval_*.py).
"""

import argparse
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import wandb
from torch import nn

from legorec.eval.metrics import classification_metrics, error_breakdown
from legorec.paths import MODELS_DIR
from legorec.shape.augment import augment_batch, denormalize
from legorec.shape.loaders import Batches
from legorec.shape.model import SmallCNN, load_checkpoint, save_checkpoint
from legorec.shape.preprocess import PREPROCESS_VERSION
from legorec.shape.train import (
    PHOTO_SOURCES,
    TrainConfig,
    TrainData,
    combine_train,
    evaluate,
    load_training_data,
    overfit_one_batch,
    set_seed,
    train_epoch,
)

EVAL_BATCH_SIZE = 512  # no gradients during evaluation, so bigger batches fit


def prefix(source: str, split: str) -> str:
    """W&B name of a source's split; the old names stay for comparison."""
    old = {("kaggle_renders", "val"): "val", ("kaggle_renders", "test"): "render_test"}
    if (source, split) in old:
        return old[source, split]
    if source == "internet_photos":
        return f"internet_{split}"
    return f"{source}_{split}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--internet-data", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sanity", action="store_true", help="overfit one batch only")
    return parser.parse_args()


def start_wandb_run(config: TrainConfig, data: TrainData, sanity: bool):
    tags = list(config.tags)
    if config.size_feature:
        tags.append("size-feature")

    logged_config = asdict(config) | {
        "split_hash": data.split_hash,
        "split_hashes": data.split_hashes,
        "internet_split_hash": data.split_hashes.get("internet_photos"),
        "n_classes": len(data.parts),
        "parts": data.parts,
        "preprocess": PREPROCESS_VERSION,
    }
    return wandb.init(
        project=config.project,
        config=logged_config,
        tags=tags,
        job_type="sanity" if sanity else "train",
    )


def log_sample_images(train_arrays, data: TrainData, device, config: TrainConfig):
    """Log 16 augmented training images: exactly what the network sees."""
    x, _, labels = next(iter(Batches(*train_arrays, device, 16, shuffle=True)))
    augmented = denormalize(augment_batch(x, config.hflip))
    as_uint8 = (augmented.cpu().numpy() * 255).astype(np.uint8)

    samples = [
        wandb.Image(image[0], caption=data.names[data.parts[int(label)]])
        for image, label in zip(as_uint8, labels)
    ]
    wandb.log({"samples/train": samples})


def evaluate_on_test(model, test, data: TrainData, name: str) -> dict:
    """Evaluate the best checkpoint once on a render test set and log it."""
    scores, labels, _ = evaluate(model, test)
    predictions = scores.argmax(axis=1)
    metrics = classification_metrics(scores, labels)
    metrics |= error_breakdown(labels, predictions, data.parts)

    wandb.run.summary.update({f"{name}/{key}": v for key, v in metrics.items()})
    confusion = wandb.plot.confusion_matrix(
        y_true=labels.tolist(),
        preds=predictions.tolist(),
        class_names=[data.names[part] for part in data.parts],
    )
    wandb.log({f"{name}/confusion": confusion})
    print(
        f"[{name}] top1 {metrics['top1']:.4f}  top3 {metrics['top3']:.4f}  "
        f"length errors {metrics['errors_length']}  "
        f"other errors {metrics['errors_other']}"
    )
    return metrics


def evaluate_on_internet_val(model, internet_val, data: TrainData) -> dict:
    """Evaluate the best checkpoint on internet/val and log it.

    Same numbers as scripts/eval_internet.py (top1, top3, macro_f1, conf):
    all classes of the model, all images, one row per photo.
    """
    scores, labels, _ = evaluate(model, internet_val)
    metrics = classification_metrics(scores, labels)
    metrics |= error_breakdown(labels, scores.argmax(axis=1), data.parts)
    wandb.run.summary.update({f"internet_val/{key}": v for key, v in metrics.items()})
    print(
        f"[internet val] top1 {metrics['top1']:.4f}  top3 {metrics['top3']:.4f}  "
        f"macro-F1 {metrics['macro_f1']:.4f}"
    )
    return metrics


def main():
    args = parse_args()
    config = TrainConfig.from_yaml(args.config, seed=args.seed, epochs=args.epochs)
    if args.sanity:
        # one pass over the same batch; without label smoothing the loss can reach 0
        config = replace(config, epochs=1, label_smoothing=0.0)
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. data
    roots = {"kaggle_renders": args.data}
    roots |= dict.fromkeys(["internet_photos", "internet_renders"], args.internet_data)
    data = load_training_data(config, roots)
    train_arrays = combine_train(data, config.train_on, config.real_repeat)
    print(
        f"[data] train on {', '.join(config.train_on)}: {len(train_arrays[2])} "
        f"images (photos {config.real_repeat}x), {len(data.parts)} classes"
    )

    train = Batches(*train_arrays, device, config.batch_size, shuffle=True)
    # every source with images in its val split, checked after every epoch
    vals = {
        source: Batches(*splits["val"], device, EVAL_BATCH_SIZE)
        for source, splits in data.sources.items()
        if len(splits["val"][2])
    }
    if config.select_on not in vals:
        raise ValueError(f"select_on {config.select_on} has no val images")

    # 2. logging
    run = start_wandb_run(config, data, args.sanity)
    print(f"[run] device {device}, size feature {config.size_feature}")
    log_sample_images(train_arrays, data, device, config)

    # 3. model and training
    model = SmallCNN(len(data.parts), config.width, config.dropout, config.size_feature)
    model.to(device)
    run.summary["params"] = sum(p.numel() for p in model.parameters())

    if args.sanity:
        one_batch = next(iter(Batches(*train_arrays, device, 32, shuffle=True)))
        train = [one_batch] * 50

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config.lr, total_steps=config.epochs * len(train)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    if args.sanity:
        loss = overfit_one_batch(
            model, train, optimizer, scheduler, loss_fn, config, data.size_norm
        )
        print(f"[sanity] loss on that batch after training {loss:.4f} (should be ~0)")
        run.finish()
        return

    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint = MODELS_DIR / f"shape_{run.id}.pt"
    best_val_f1 = -1.0

    for epoch in range(config.epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train, optimizer, scheduler, loss_fn, config, data.size_norm
        )
        logged = {"epoch": epoch, "train/loss": train_loss, "train/top1": train_acc}
        val_metrics = {}
        for source, batches in vals.items():
            scores, labels, loss = evaluate(model, batches)
            val_metrics[source] = classification_metrics(scores, labels)
            name = prefix(source, "val")
            logged |= {f"{name}/{k}": v for k, v in val_metrics[source].items()}
            logged[f"{name}/loss"] = loss
        seconds = time.time() - start
        wandb.log(logged | {"lr": scheduler.get_last_lr()[0], "epoch_time_s": seconds})

        shown = "  ".join(f"{s} {m['top1']:.3f}" for s, m in val_metrics.items())
        print(
            f"ep {epoch:02d}  train {train_loss:.3f}/{train_acc:.3f}  "
            f"val top1: {shown}  ({seconds:.1f}s)"
        )

        # keep the best epoch, not the last one
        if val_metrics[config.select_on]["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics[config.select_on]["macro_f1"]
            save_checkpoint(
                checkpoint,
                model,
                data.parts,
                data.names,
                asdict(config),
                data.split_hash,
                data.size_norm,
            )

    # 4. final test with the best checkpoint
    model.load_state_dict(load_checkpoint(checkpoint, device)["model"])
    test_metrics = {}
    for source in config.train_on:
        test = data.sources[source]["test"]
        if source in PHOTO_SOURCES or not len(test[2]):
            continue  # photo test splits belong to scripts/eval_*.py
        name = prefix(source, "test")
        batches = Batches(*test, device, EVAL_BATCH_SIZE)
        metrics = evaluate_on_test(model, batches, data, name)
        test_metrics |= {f"{name}_{k}": v for k, v in metrics.items()}
    if "internet_photos" in vals:
        real_metrics = evaluate_on_internet_val(model, vals["internet_photos"], data)
        test_metrics |= {f"internet_val_{key}": v for key, v in real_metrics.items()}

    artifact = wandb.Artifact("shape-cnn", type="model", metadata=test_metrics)
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)
    run.finish()
    print(f"[done] model saved to {checkpoint}")


if __name__ == "__main__":
    main()
