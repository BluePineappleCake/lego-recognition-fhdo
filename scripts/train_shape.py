"""Train the shape classifier on the Kaggle renders and log everything to W&B.

    python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
    python scripts/train_shape.py --config configs/shape_baseline.yaml
    python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
    python scripts/train_shape.py --config configs/shape_baseline.yaml --data C:\\lego\\data

--data: folder that directly contains the 40,000 PNGs (default: see legorec.paths).
To try a trained model on real photos, use scripts/predict_shape.py.
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

# values a config file does not set
DEFAULTS = {
    "size": 96,  # crop size in pixels
    "width": 32,  # channels of the first conv block
    "dropout": 0.3,
    "size_feature": False,  # give the network the measured brick size
    "size_noise": 0.05,  # ~5 % noise on that size during training
    "epochs": 30,
    "batch_size": 128,
    "lr": 3e-3,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    "hflip": False,
    "seed": 0,
    "project": "lego-recognition",
    "tags": ["renders-only"],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--sanity", action="store_true", help="overfit one batch only")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = DEFAULTS | yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    return args, cfg


def load_data(cfg, data_dir):
    """Renders -> split by pose -> preprocessed arrays (cached on disk)."""
    items = index_renders(data_dir or renders_dir())
    parts = sorted({it.part for it in items})
    part_to_idx = {p: i for i, p in enumerate(parts)}
    names = {it.part: it.name for it in items}
    split, split_hash = load_or_create_split(items)

    raw = {}
    for s in ("train", "val", "test"):
        name = f"renders_{s}_{cfg['size']}_{PREPROCESS_VERSION}_{split_hash}.npz"
        cache = processed_dir() / name
        x, px = build_cache(split[s], cfg["size"], cache)
        y = [part_to_idx[it.part] for it in split[s]]
        raw[s] = (x, px, y)
        print(f"[data] {s}: {len(y)} images")

    # size feature: standardised with the TRAINING set statistics only
    size_norm = SizeNorm.fit(raw["train"][1])
    data = {s: (x, size_norm(px), y) for s, (x, px, y) in raw.items()}
    return data, parts, names, size_norm, split_hash


def train_epoch(model, batches, opt, sched, loss_fn, cfg, size_norm, augment=True):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, z, y in batches:
        if augment:
            x = augment_batch(x, cfg["hflip"])
            # simulate that we will never measure the size perfectly on photos
            z = z + torch.randn_like(z) * cfg["size_noise"] / size_norm.sd
        out = model(x, z)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, batches):
    """-> logits (N, C), true labels (N,), mean cross-entropy loss"""
    model.eval()
    logits, ys = [], []
    for x, z, y in batches:
        logits.append(model(x, z).cpu())
        ys.append(y.cpu())
    logits, ys = torch.cat(logits), torch.cat(ys)
    return logits.numpy(), ys.numpy(), F.cross_entropy(logits, ys).item()


def main():
    args, cfg = parse_args()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, parts, names, size_norm, split_hash = load_data(cfg, args.data)
    train = Batches(*data["train"], device, cfg["batch_size"], shuffle=True)
    val = Batches(*data["val"], device, 512)
    test = Batches(*data["test"], device, 512)

    tags = list(cfg["tags"]) + (["size-feature"] if cfg["size_feature"] else [])
    run = wandb.init(
        project=cfg["project"],
        config=cfg
        | {"split_hash": split_hash, "n_classes": len(parts)}
        | {"preprocess": PREPROCESS_VERSION},
        tags=tags,
        job_type="sanity" if args.sanity else "train",
    )
    print(f"[run] device {device}, size feature {cfg['size_feature']}")

    # log 16 augmented training images: what the network actually sees
    x, _, y = next(iter(Batches(*data["train"], device, 16, shuffle=True)))
    images = (denormalize(augment_batch(x, cfg["hflip"])).cpu().numpy() * 255).astype(
        np.uint8
    )
    wandb.log(
        {
            "samples/train": [
                wandb.Image(img[0], caption=names[parts[int(c)]])
                for img, c in zip(images, y)
            ]
        }
    )

    model = SmallCNN(len(parts), cfg["width"], cfg["dropout"], cfg["size_feature"])
    model = model.to(device)
    run.summary["params"] = sum(p.numel() for p in model.parameters())

    if args.sanity:
        # a correct model + training loop must be able to memorise one batch
        one_batch = [next(iter(Batches(*data["train"], device, 32, shuffle=True)))]
        train, cfg["epochs"], cfg["label_smoothing"] = one_batch * 50, 1, 0.0

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, cfg["lr"], total_steps=cfg["epochs"] * len(train)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    if args.sanity:
        train_epoch(model, train, opt, sched, loss_fn, cfg, size_norm, augment=False)
        with torch.no_grad():
            x, z, y = one_batch[0]
            loss = loss_fn(model(x, z), y).item()
        print(f"[sanity] loss on that batch after training {loss:.4f} (should be ~0)")
        run.finish()
        return

    # ---- training: keep the checkpoint with the best validation macro-F1 ----
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"shape_{run.id}.pt"
    best = -1.0
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train, opt, sched, loss_fn, cfg, size_norm)
        logits, y, val_loss = predict(model, val)
        m = classification_metrics(logits, y)
        wandb.log(
            {"epoch": epoch, "train/loss": tr_loss, "train/top1": tr_acc}
            | {f"val/{k}": v for k, v in m.items()}
            | {"val/loss": val_loss, "lr": sched.get_last_lr()[0]}
            | {"epoch_time_s": time.time() - t0}
        )
        print(
            f"ep {epoch:02d}  train {tr_loss:.3f}/{tr_acc:.3f}  "
            f"val {val_loss:.3f}/{m['top1']:.3f}  ({time.time() - t0:.1f}s)"
        )
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "parts": parts,
                    "names": names,
                    "cfg": cfg,
                    "split_hash": split_hash,
                    "size_norm": {"mu": size_norm.mu, "sd": size_norm.sd},
                },
                ckpt,
            )

    # ---- final evaluation on the render test set with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    logits, y, _ = predict(model, test)
    pred = logits.argmax(1)
    m = classification_metrics(logits, y) | error_breakdown(y, pred, parts)
    run.summary.update({f"render_test/{k}": v for k, v in m.items()})
    labels = [names[p] for p in parts]
    cm = wandb.plot.confusion_matrix(
        y_true=y.tolist(), preds=pred.tolist(), class_names=labels
    )
    wandb.log({"render_test/confusion": cm})
    print(
        f"[render test] top1 {m['top1']:.4f}  top3 {m['top3']:.4f}  "
        f"length errors {m['errors_length']}  other errors {m['errors_other']}"
    )

    artifact = wandb.Artifact("shape-cnn", type="model", metadata=m)
    artifact.add_file(str(ckpt))
    run.log_artifact(artifact)
    run.finish()
    print(f"[done] model saved to {ckpt}")


if __name__ == "__main__":
    main()
