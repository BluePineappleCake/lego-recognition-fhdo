"""Train the shape classifier on the Kaggle renders and log everything to W&B.

    python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
    python scripts/train_shape.py --config configs/shape_baseline.yaml
    python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
    python scripts/train_shape.py --config configs/shape_baseline.yaml --data C:\\lego\\data

--data: folder that directly contains the 40,000 PNGs (default: see legorec.paths).
--real_dir: quick test on single-brick photos, layout <real_dir>/<part number>/*.jpg
            (temporary, until our labelled photo test set exists).
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from legorec.data.renders import build_cache, index_renders, load_or_create_split
from legorec.eval.metrics import classification_metrics, error_breakdown
from legorec.paths import MODELS_DIR, processed_dir, renders_dir
from legorec.shape.augment import cpu_transform, denormalize, gpu_augment
from legorec.shape.loaders import ArrayDataset, GPUBatches
from legorec.shape.model import SizeNorm, SmallCNN
from legorec.shape.preprocess import preprocess_image

DEFAULTS = {
    "size": 96,
    "width": 32,
    "dropout": 0.3,
    "size_feature": False,
    "size_noise": 0.05,  # std of log-normal noise on the size during training (~5 %)
    "epochs": 30,
    "bs": 128,
    "lr": 3e-3,
    "wd": 5e-4,
    "label_smoothing": 0.1,
    "hflip": False,
    "seed": 0,
    "loader": "auto",  # gpu: data + augmentation on the GPU, cpu: DataLoader workers
    "workers": 2,
    "project": "lego-recognition",
    "tags": ["renders-only"],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--sanity", action="store_true", help="overfit one batch, nothing else"
    )
    ap.add_argument("--real_dir", default=None)
    ap.add_argument("--real_scale", type=float, default=None)
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg = DEFAULTS | yaml.safe_load(f)
    for k in ("seed", "epochs"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg["config_file"] = Path(a.config).name
    if cfg["size_feature"] and a.real_dir and a.real_scale is None:
        ap.error(
            "size_feature + --real_dir needs --real_scale (photo px are not render px)"
        )
    return a, cfg


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    logits, ys = [], []
    for x, z, y in loader:
        logits.append(model(x.to(dev), z.to(dev)).float().cpu())
        ys.append(y.cpu())
    logits, ys = torch.cat(logits), torch.cat(ys)
    return logits.numpy(), ys.numpy(), F.cross_entropy(logits, ys).item()


def load_real(real_dir, part_to_idx, size, real_scale):
    xs, pxs, ys, skipped = [], [], [], set()
    for d in sorted(Path(real_dir).iterdir()):
        if not d.is_dir():
            continue
        if d.name not in part_to_idx:
            skipped.add(d.name)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                x, px = preprocess_image(Image.open(f), size)
                xs.append(x)
                pxs.append(px * real_scale)
                ys.append(part_to_idx[d.name])
    if skipped:
        print(f"[real] not among the training classes (ignored): {sorted(skipped)}")
    return np.stack(xs), np.array(pxs, np.float32), ys


def main():
    a, cfg = parse_args()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ----
    items = index_renders(a.data or renders_dir())
    parts = sorted({it.part for it in items})
    part_to_idx = {p: i for i, p in enumerate(parts)}
    names = {it.part: it.name for it in items}
    print(f"[data] {len(items)} images, {len(parts)} classes")
    split, shash = load_or_create_split(items)
    raw = {}
    for s in ("train", "val", "test"):
        cache = processed_dir() / f"renders_{s}_{cfg['size']}_{shash}.npz"
        x, px = build_cache(split[s], cfg["size"], cache)
        raw[s] = (x, px, [part_to_idx[it.part] for it in split[s]])
        print(f"[data] {s}: {len(px)}")
    snorm = SizeNorm.fit(raw["train"][1])
    data = {s: (x, snorm(px), y) for s, (x, px, y) in raw.items()}

    tags = list(cfg["tags"]) + (["size-feature"] if cfg["size_feature"] else [])
    run = wandb.init(
        project=cfg["project"],
        config=cfg | {"split_hash": shash, "n_classes": len(parts)},
        tags=tags,
        job_type="sanity" if a.sanity else "train",
    )
    fast = cfg["loader"] == "gpu" or (cfg["loader"] == "auto" and dev == "cuda")
    torch.backends.cudnn.benchmark = True
    print(
        f"[run] device {dev}, loader {'gpu' if fast else 'cpu'}, size {cfg['size_feature']}"
    )

    if fast:
        tr = GPUBatches(*data["train"], dev, cfg["bs"], shuffle=True, drop_last=True)
        val, test = (GPUBatches(*data[s], dev, 512) for s in ("val", "test"))
        xs, _, ys = next(iter(GPUBatches(*data["train"], dev, 16, shuffle=True)))
        samples = list(zip(denormalize(gpu_augment(xs, cfg["hflip"])).cpu(), ys.cpu()))
    else:
        ds = ArrayDataset(*data["train"], tf=cpu_transform(cfg["hflip"]))
        w = cfg["workers"]
        tr = DataLoader(
            ds,
            cfg["bs"],
            shuffle=True,
            num_workers=w,
            drop_last=True,
            persistent_workers=w > 0,
        )
        val, test = (
            DataLoader(
                ArrayDataset(*data[s]), 256, num_workers=w, persistent_workers=w > 0
            )
            for s in ("val", "test")
        )
        samples = [(denormalize(x), y) for x, _, y in (ds[i] for i in range(16))]
    # what the network actually sees: catches preprocessing bugs early
    wandb.log(
        {
            "samples/train": [
                wandb.Image(
                    (x[0].numpy() * 255).astype(np.uint8), caption=names[parts[int(y)]]
                )
                for x, y in samples
            ]
        }
    )
    if a.sanity:
        tr = [
            next(iter(DataLoader(ArrayDataset(*data["train"]), 32, shuffle=True)))
        ] * 50

    # ---- model ----
    model = SmallCNN(len(parts), cfg["width"], cfg["dropout"], cfg["size_feature"]).to(
        dev
    )
    run.summary["params"] = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs = 1 if a.sanity else cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, cfg["lr"], total_steps=epochs * len(tr)
    )
    crit = nn.CrossEntropyLoss(
        label_smoothing=0 if a.sanity else cfg["label_smoothing"]
    )
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / f"shape_{run.id}.pt"

    best, step = -1.0, 0
    for ep in range(epochs):
        model.train()
        run_loss, run_acc, n, t0 = 0.0, 0, 0, time.time()
        for x, z, y in tr:
            x, z, y = x.to(dev), z.to(dev), y.to(dev)
            if not a.sanity:
                if fast:
                    x = gpu_augment(x, cfg["hflip"])
                # simulate size-measurement error on real photos
                z = z + torch.randn_like(z) * cfg["size_noise"] / snorm.sd
            with torch.autocast(dev, enabled=dev == "cuda"):
                out = model(x, z)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            # accumulate on the GPU; .item() every step would force a sync
            run_loss += loss.detach().float() * len(y)
            run_acc += (out.argmax(1) == y).sum()
            n += len(y)
            step += 1
            if a.sanity and step % 10 == 0:
                wandb.log({"sanity/loss": loss.item()}, step=step)
        if a.sanity:
            print(f"[sanity] final batch loss {loss.item():.4f} (should be close to 0)")
            run.finish()
            return

        logits, y, vloss = predict(model, val, dev)
        m = classification_metrics(logits, y) | {"loss": vloss}
        tl, ta, et = float(run_loss) / n, float(run_acc) / n, time.time() - t0
        wandb.log(
            {"epoch": ep, "train/loss": tl, "train/top1": ta, "epoch_time_s": et}
            | {"lr": sched.get_last_lr()[0]}
            | {f"val/{k}": v for k, v in m.items()},
            step=step,
        )
        print(
            f"ep {ep:02d} train {tl:.3f}/{ta:.3f} val {vloss:.3f}/{m['top1']:.3f} ({et:.1f}s)"
        )
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "parts": parts,
                    "names": names,
                    "cfg": cfg,
                    "split_hash": shash,
                    "size_norm": {"mu": snorm.mu, "sd": snorm.sd},
                },
                ckpt,
            )

    # ---- final evaluation with the best checkpoint ----
    model.load_state_dict(
        torch.load(ckpt, map_location=dev, weights_only=False)["model"]
    )
    labels = [names[p] for p in parts]
    logits, y, tloss = predict(model, test, dev)
    pred = logits.argmax(1)
    m = (
        classification_metrics(logits, y)
        | {"loss": tloss}
        | error_breakdown(y, pred, parts)
    )
    run.summary.update({f"render_test/{k}": v for k, v in m.items()})
    cm = wandb.plot.confusion_matrix(
        y_true=y.tolist(), preds=pred.tolist(), class_names=labels
    )
    wandb.log({"render_test/confusion": cm})
    print(
        f"[render test] top1 {m['top1']:.4f} top3 {m['top3']:.4f} "
        f"length errors {m['errors_length']} other errors {m['errors_other']}"
    )

    if a.real_dir:
        xr, pxr, yr = load_real(
            a.real_dir, part_to_idx, cfg["size"], a.real_scale or 1.0
        )
        loader = DataLoader(ArrayDataset(xr, snorm(pxr), yr), 256)
        lr_, yr_, _ = predict(model, loader, dev)
        pr = lr_.argmax(1)
        mr = classification_metrics(lr_, yr_)
        run.summary.update({f"real/{k}": v for k, v in mr.items()})
        run.summary["gap/top1"] = m["top1"] - mr["top1"]  # the core experiment
        cm = wandb.plot.confusion_matrix(
            y_true=yr_.tolist(), preds=pr.tolist(), class_names=labels
        )
        wandb.log({"real/confusion": cm})
        print(
            f"[real] top1 {mr['top1']:.3f} (render {m['top1']:.3f}, gap {m['top1'] - mr['top1']:.3f})"
        )

    art = wandb.Artifact("shape-cnn", type="model", metadata=m)
    art.add_file(str(ckpt))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
