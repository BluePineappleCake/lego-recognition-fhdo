"""Train the shape classifier on the Kaggle renders and log everything to W&B.

    python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
    python scripts/train_shape.py --config configs/shape_baseline.yaml
    python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
    python scripts/train_shape.py --config configs/shape_baseline.yaml --data C:\\lego\\data

--data: folder that directly contains the 40,000 PNGs (default: see legorec.paths).
All settings and their defaults: TrainConfig in src/legorec/shape/train.py.
To try a trained model on real photos, use scripts/predict_shape.py.

What happens, in order:
    1. read the config, load (or build) the preprocessed renders
    2. start a W&B run and log a few training images
    3. train for N epochs; after every epoch check the validation set and keep
       the checkpoint with the best validation macro-F1
    4. load that best checkpoint and evaluate it once on the render test set
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
from legorec.shape.model import SmallCNN, save_checkpoint
from legorec.shape.preprocess import PREPROCESS_VERSION
from legorec.shape.train import (
    RenderData,
    TrainConfig,
    evaluate,
    load_render_data,
    overfit_one_batch,
    set_seed,
    train_epoch,
)

EVAL_BATCH_SIZE = 512  # no gradients during evaluation, so bigger batches fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sanity", action="store_true", help="overfit one batch only")
    return parser.parse_args()


def start_wandb_run(config: TrainConfig, data: RenderData, sanity: bool):
    tags = list(config.tags)
    if config.size_feature:
        tags.append("size-feature")

    logged_config = asdict(config) | {
        "split_hash": data.split_hash,
        "n_classes": len(data.parts),
        "preprocess": PREPROCESS_VERSION,
    }
    return wandb.init(
        project=config.project,
        config=logged_config,
        tags=tags,
        job_type="sanity" if sanity else "train",
    )


def log_sample_images(data: RenderData, device, config: TrainConfig):
    """Log 16 augmented training images: exactly what the network sees."""
    x, _, labels = next(iter(Batches(*data.splits["train"], device, 16, shuffle=True)))
    augmented = denormalize(augment_batch(x, config.hflip))
    as_uint8 = (augmented.cpu().numpy() * 255).astype(np.uint8)

    samples = [
        wandb.Image(image[0], caption=data.names[data.parts[int(label)]])
        for image, label in zip(as_uint8, labels)
    ]
    wandb.log({"samples/train": samples})


def evaluate_on_test(model, test, data: RenderData) -> dict:
    """Evaluate the best checkpoint once on the render test set and log it."""
    scores, labels, _ = evaluate(model, test)
    predictions = scores.argmax(axis=1)
    metrics = classification_metrics(scores, labels)
    metrics |= error_breakdown(labels, predictions, data.parts)

    wandb.run.summary.update({f"render_test/{key}": v for key, v in metrics.items()})
    confusion = wandb.plot.confusion_matrix(
        y_true=labels.tolist(),
        preds=predictions.tolist(),
        class_names=[data.names[part] for part in data.parts],
    )
    wandb.log({"render_test/confusion": confusion})
    print(
        f"[render test] top1 {metrics['top1']:.4f}  top3 {metrics['top3']:.4f}  "
        f"length errors {metrics['errors_length']}  "
        f"other errors {metrics['errors_other']}"
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
    data = load_render_data(config, args.data)
    train = Batches(*data.splits["train"], device, config.batch_size, shuffle=True)
    val = Batches(*data.splits["val"], device, EVAL_BATCH_SIZE)
    test = Batches(*data.splits["test"], device, EVAL_BATCH_SIZE)

    # 2. logging
    run = start_wandb_run(config, data, args.sanity)
    print(f"[run] device {device}, size feature {config.size_feature}")
    log_sample_images(data, device, config)

    # 3. model and training
    model = SmallCNN(len(data.parts), config.width, config.dropout, config.size_feature)
    model.to(device)
    run.summary["params"] = sum(p.numel() for p in model.parameters())

    if args.sanity:
        one_batch = next(iter(Batches(*data.splits["train"], device, 32, shuffle=True)))
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
        val_scores, val_labels, val_loss = evaluate(model, val)
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

        # keep the best epoch, not the last one
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
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
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    test_metrics = evaluate_on_test(model, test, data)

    artifact = wandb.Artifact("shape-cnn", type="model", metadata=test_metrics)
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)
    run.finish()
    print(f"[done] model saved to {checkpoint}")


if __name__ == "__main__":
    main()
