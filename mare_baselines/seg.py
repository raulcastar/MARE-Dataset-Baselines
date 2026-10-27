"""
Semantic segmentation baselines: a U-Net with an ImageNet ResNet-34 encoder
(segmentation_models_pytorch), cross-entropy + Dice with class weights.

    python -m mare_baselines.seg --task object --data /data --out /results

Tasks:
    object       rgb_central -> object (classes 1-10; 0 = unlabeled is ignored)
    shadow       rgb_central -> shadow (6 shadow classes)
    spad_object  spad        -> spad_object (the central labels on the SPAD grid)
    rear_object  rear_rgb    -> rear_object (the rear camera, which sees the wheel tracks)
    rear_shadow  rear_rgb    -> rear_shadow
    spad_shadow  spad        -> spad_shadow (255 outside the central camera is ignored)

Trains on the train maps, keeps the epoch with the best validation mIoU, and
reports per-class IoU on the test map (never seen in training) to
<out>/<task>/metrics.json, with example predictions.
"""
import argparse
import json
import os
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import data

TASKS = {
    "object": {"input": "rgb_central", "label": "object", "size": (512, 384), "classes": 11, "ignore": [0]},
    "shadow": {"input": "rgb_central", "label": "shadow", "size": (512, 384), "classes": 6, "ignore": []},
    "spad_object": {"input": "spad", "label": "spad_object", "size": (256, 256), "classes": 11, "ignore": [0]},
    "rear_object": {"input": "rear_rgb", "label": "rear_object", "size": (512, 384), "classes": 11, "ignore": [0]},
    "rear_shadow": {"input": "rear_rgb", "label": "rear_shadow", "size": (512, 384), "classes": 6, "ignore": []},
    "spad_shadow": {"input": "spad", "label": "spad_shadow", "size": (256, 256), "classes": 6, "ignore": []},
}
ARCHS = {  # name -> (description, constructor of a model with n classes)
    "unet": ("U-Net, ResNet-34 encoder (ImageNet)",
             lambda n: smp.Unet("resnet34", encoder_weights="imagenet", classes=n)),
    "deeplabv3plus": ("DeepLabV3+, ResNet-50 encoder (ImageNet)",
                      lambda n: smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", classes=n)),
    "segformer": ("SegFormer-B2 (MiT-B2 encoder, ImageNet)",
                  lambda n: smp.Segformer("mit_b2", encoder_weights="imagenet", classes=n)),
}
# Peak learning rate (AdamW) of each model's standard recipe.
LEARNING_RATES = {"unet": 3e-4, "deeplabv3plus": 3e-4, "segformer": 6e-5}
SHADOW_COLOURS = {0: (0, 0, 0), 1: (40, 120, 255), 2: (20, 20, 60), 3: (128, 128, 128), 4: (70, 70, 70),
                  5: (255, 140, 0)}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SUN_BINS = [(0, 10), (10, 20), (20, 90)]          # sun elevation (deg) bins for the per-elevation scores


class Cached(Dataset):
    """A split loaded once into memory, resized: images uint8, labels uint8 with IGNORE."""

    def __init__(self, root, split, task, limit=None):
        cfg = TASKS[task]
        items = data.frames(root, split, [cfg["input"], cfg["label"]])
        if limit:
            items = items[::max(1, len(items) // limit)][:limit]
        self.images, self.labels, self.sun = [], [], []
        for ep_dir, stem, meta in tqdm(items, desc=f"loading {split}", leave=False):
            img = data.as_rgb(data.load(ep_dir, cfg["input"], stem))
            lab = data.load(ep_dir, cfg["label"], stem).astype(np.uint8).copy()
            for i in cfg["ignore"]:
                lab[lab == i] = data.IGNORE
            self.images.append(data.resize(img, cfg["size"], nearest=False))
            self.labels.append(data.resize(lab, cfg["size"], nearest=True))
            self.sun.append(float(meta["sun"]["elevation_deg"]))
        self.train = split == "train"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, lab = self.images[i], self.labels[i]
        x = img.astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:                 # horizontal flip
                x, lab = x[:, ::-1], lab[:, ::-1]
            x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.05, 0.05)   # exposure jitter
            x = np.clip(x, 0.0, 1.0)
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), \
            torch.from_numpy(np.array(lab)).long(), i


def class_weights(labels, n):
    """Square-root median-frequency balancing over the training labels, clipped to 0.1-10
    (absent classes: 0): rare classes count more without drowning the large ones."""
    counts = np.zeros(n, np.float64)
    for lab in labels:
        v, c = np.unique(lab[lab != data.IGNORE], return_counts=True)
        counts[v] += c
    freq = counts / counts.sum()
    present = freq > 0
    w = np.zeros(n, np.float32)
    w[present] = np.sqrt(np.median(freq[present]) / freq[present])
    return np.clip(w, 0.1, 10.0) * present, freq


def confusion(pred, lab, n):
    keep = lab != data.IGNORE
    return np.bincount(n * lab[keep] + pred[keep], minlength=n * n).reshape(n, n)


def scores(cm, names):
    tp = np.diag(cm).astype(np.float64)
    gt, pr = cm.sum(1), cm.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1)
    present = gt > 0
    return {"mIoU": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "iou": {names[c]: float(iou[c]) for c in range(len(names)) if present[c]}}


@torch.no_grad()
def evaluate(model, ds, n, device, bins=False):
    model.eval()
    cm = np.zeros((n, n), np.int64)
    cms = {b: np.zeros((n, n), np.int64) for b in SUN_BINS} if bins else {}
    for x, y, idx in DataLoader(ds, batch_size=8, num_workers=0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x.to(device)).argmax(1).cpu().numpy()
        y = y.numpy()
        for k in range(len(p)):
            c = confusion(p[k].ravel(), y[k].ravel(), n)
            cm += c
            for lo, hi in cms:
                if lo <= ds.sun[int(idx[k])] < hi:
                    cms[(lo, hi)] += c
    return cm, cms


def palette(task, legend):
    lut = np.zeros((256, 3), np.uint8)
    lut[data.IGNORE] = (60, 60, 60)
    if task.endswith("shadow"):
        for k, rgb in SHADOW_COLOURS.items():
            lut[k] = rgb
    else:
        for c in legend:
            lut[c["id"]] = c["rgb"]
    return lut


@torch.no_grad()
def save_examples(model, ds, task, legend, out, device, count=6):
    model.eval()
    lut = palette(task, legend)
    picks = np.linspace(0, len(ds) - 1, count).astype(int)
    rows = []
    for i in picks:
        x, y, _ = ds[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        rows.append(np.hstack([ds.images[i], lut[y.numpy()], lut[p]]))
    Image.fromarray(np.vstack(rows)).save(os.path.join(out, "examples.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--data", default="/data")
    parser.add_argument("--out", default="/results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,
                        help="peak learning rate (default: the model's standard recipe, LEARNING_RATES)")
    parser.add_argument("--max-minutes", type=float, default=90.0, help="hard limit on training time")
    parser.add_argument("--limit", type=int, default=None, help="samples per split (quick test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=ARCHS, default="unet")
    parser.add_argument("--force", action="store_true", help="train again even if the run has finished")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = LEARNING_RATES[args.arch]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = TASKS[args.task]
    name = args.task + ("" if args.arch == "unet" else f"_{args.arch}") + ("" if args.seed == 0 else f"_seed{args.seed}")
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(os.path.join(out, "metrics.json")) and not args.force:
        print(f"{name}: already finished ({out}/metrics.json); --force to train again")
        return
    device = "cuda"

    train = Cached(args.data, "train", args.task, args.limit)
    val = Cached(args.data, "validation", args.task, args.limit and max(args.limit // 4, 8))
    test = Cached(args.data, "test", args.task, args.limit and max(args.limit // 4, 8))
    meta = data.episodes(args.data, "test")[0][1]
    legend = meta["object_classes"]
    if args.task.endswith("shadow"):
        names = [meta["shadow_classes"][str(k)] for k in range(cfg["classes"])]
    else:
        names = [c["name"] for c in sorted(legend, key=lambda c: c["id"])][:cfg["classes"]]
    weights, freq = class_weights(train.labels, cfg["classes"])
    print(f"{args.task}: {len(train)} train, {len(val)} validation, {len(test)} test samples")
    print("class weights:", {names[c]: round(float(weights[c]), 2) for c in range(cfg["classes"])})

    last = os.path.join(out, "last.pt")
    state = torch.load(last, map_location=device, weights_only=False) if os.path.isfile(last) else None
    if state and state.get("epochs", args.epochs) != args.epochs:
        print(f"{name}: continuing with --epochs {state['epochs']}, as the interrupted run started")
        args.epochs = state["epochs"]                   # the schedule spans the original length
    model = ARCHS[args.arch][1](cfg["classes"]).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=data.IGNORE)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=data.IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                        persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader),
                                                pct_start=0.1)

    # Resume an interrupted run from its last finished epoch (last.pt, written every epoch).
    best, history, start, t0 = -1.0, [], 0, time.time()
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        best, history, start = state["best"], state["history"], state["epoch"]
        t0 -= history[-1]["minutes"] * 60 if history else 0     # the time limit includes the earlier part
        print(f"{name}: resuming after epoch {start} (best validation mIoU {best:.3f})")
    for epoch in range(start, args.epochs):
        model.train()
        losses = []
        for x, y, _ in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = ce(logits.float(), y) + dice(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        cm, _ = evaluate(model, val, cfg["classes"], device)
        s = scores(cm, names)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mIoU": s["mIoU"],
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f}, validation mIoU {s['mIoU']:.3f}", flush=True)
        if s["mIoU"] > best:
            best = s["mIoU"]
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "epochs": args.epochs, "best": best, "history": history}, last + ".tmp")
        os.replace(last + ".tmp", last)                 # atomic write
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"stopping: {args.max_minutes} min limit reached")
            break

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    cm, cms = evaluate(model, test, cfg["classes"], device, bins=True)
    result = {"task": args.task, "model": ARCHS[args.arch][0], "arch": args.arch, "input": cfg["input"],
              "label": cfg["label"], "size": cfg["size"], "epochs_run": len(history),
              "best_validation_mIoU": best, "test": scores(cm, names),
              "test_by_sun_elevation": {f"{lo}-{hi} deg": scores(c, names) for (lo, hi), c in cms.items()
                                        if c.sum()},
              "train_class_frequency": {names[c]: float(freq[c]) for c in range(cfg["classes"])},
              "samples": {"train": len(train), "validation": len(val), "test": len(test)},
              "history": history, "args": vars(args)}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_examples(model, test, args.task, legend, out, device)
    print(json.dumps({"test_mIoU": result["test"]["mIoU"], "iou": result["test"]["iou"]}, indent=2))


if __name__ == "__main__":
    main()
