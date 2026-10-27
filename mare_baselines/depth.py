"""
Monocular depth baseline: Depth Anything V2-Small (Hugging Face transformers),
zero-shot and fine-tuned on the dataset.

    python -m mare_baselines.depth --data /data --out /results

Zero-shot, the model predicts relative inverse depth; each test image's
prediction is aligned to the ground truth with a least-squares scale and shift
in inverse depth (the usual protocol for relative models) before scoring.
Fine-tuned, it predicts metric depth (depth = K / output) with a scale-invariant
log loss; it is scored as it is (metric) and aligned. Ground truth: depth_m of
the central camera, pixels from MIN_DEPTH to MAX_DEPTH metres (no sky).
Results go to <out>/depth/metrics.json with example predictions.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

from . import data

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SIZE = (518, 392)                # width, height: multiples of the ViT's 14-pixel patches, 4:3 like the camera
MIN_DEPTH, MAX_DEPTH = 0.5, 80.0
K = 10.0                         # metric depth = K / output after fine-tuning
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Cached(Dataset):
    def __init__(self, root, split, limit=None):
        items = data.frames(root, split, ["rgb_central", "depth_m"])
        if limit:
            items = items[::max(1, len(items) // limit)][:limit]
        self.images, self.depths = [], []
        for ep_dir, stem, _ in tqdm(items, desc=f"loading {split}", leave=False):
            self.images.append(data.resize(data.as_rgb(data.load(ep_dir, "rgb_central", stem)), SIZE, nearest=False))
            d = data.load(ep_dir, "depth_m", stem).astype(np.float32)
            self.depths.append(data.resize(d, SIZE, nearest=True).astype(np.float16))
        self.train = split == "train"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i].astype(np.float32) / 255.0
        d = self.depths[i].astype(np.float32)
        if self.train and np.random.rand() < 0.5:
            x, d = x[:, ::-1], d[:, ::-1]
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(d))


def valid_mask(d):
    return torch.isfinite(d) & (d >= MIN_DEPTH) & (d <= MAX_DEPTH)


def depth_metrics(pred, gt):
    """AbsRel, RMSE (m), delta < 1.25 over valid pixels (numpy arrays)."""
    ratio = np.maximum(pred / gt, gt / pred)
    return {"abs_rel": float(np.mean(np.abs(pred - gt) / gt)), "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "delta1": float(np.mean(ratio < 1.25))}


def align_inverse(out, gt, m):
    """Least-squares scale and shift of a relative inverse-depth prediction to 1/gt; returns depth."""
    a = out[m].double()
    b = (1.0 / gt[m]).double()
    A = torch.stack([a, torch.ones_like(a)], 1)
    s, t = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    inv = (s * out + t).float().clamp(min=1.0 / MAX_DEPTH / 2)
    return 1.0 / inv


def predict(model, x):
    out = model(pixel_values=x).predicted_depth
    return F.interpolate(out[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


@torch.no_grad()
def evaluate(model, ds, device, metric):
    model.eval()
    acc = {"aligned": [], "metric": []}
    for x, d in DataLoader(ds, batch_size=8, num_workers=0):
        x, d = x.to(device), d.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = predict(model, x).float()
        for k in range(len(x)):
            m = valid_mask(d[k])
            if m.sum() < 100:
                continue
            gt = d[k][m].cpu().numpy()
            acc["aligned"].append(depth_metrics(align_inverse(out[k], d[k], m)[m].cpu().numpy(), gt))
            if metric:
                acc["metric"].append(depth_metrics((K / out[k].clamp(min=1e-3))[m].cpu().numpy(), gt))
    return {kind: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            for kind, rows in acc.items() if rows}


def silog(out, d):
    """Scale-invariant log loss on metric depth K / out, valid pixels only."""
    m = valid_mask(d)
    g = torch.log(K / out.clamp(min=1e-3))[m] - torch.log(d[m])
    return torch.sqrt((g ** 2).mean() - 0.85 * g.mean() ** 2 + 1e-8) * 10.0


def jet(v):
    """Depth (m) -> JET colours on a log scale, 0.5-60 m, near red; invalid pixels black."""
    ok = np.isfinite(v) & (v > 0)
    z = np.zeros(v.shape)
    z[ok] = 1.0 - np.log(np.clip(v[ok], 0.5, 60.0) / 0.5) / np.log(120.0)
    rgb = np.stack([np.clip(1.5 - np.abs(4 * z - c), 0, 1) for c in (3, 2, 1)], -1)
    rgb[~ok] = 0
    return (rgb * 255).astype(np.uint8)


@torch.no_grad()
def save_examples(model, ds, out_dir, device, count=4):
    model.eval()
    rows = []
    for i in np.linspace(0, len(ds) - 1, count).astype(int):
        x, d = ds[i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = predict(model, x[None].to(device)).float()[0]
        pred = (K / o.clamp(min=1e-3)).cpu().numpy()
        rows.append(np.hstack([ds.images[i], jet(d.numpy()), jet(pred)]))   # image | ground truth | prediction
    Image.fromarray(np.vstack(rows)).save(os.path.join(out_dir, "examples.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="/data")
    parser.add_argument("--out", default="/results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5, help="decoder; the encoder gets a tenth")
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.join(args.out, "depth")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    train = Cached(args.data, "train", args.limit)
    val = Cached(args.data, "validation", args.limit and max(args.limit // 4, 8))
    test = Cached(args.data, "test", args.limit and max(args.limit // 4, 8))
    print(f"depth: {len(train)} train, {len(val)} validation, {len(test)} test samples")

    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device)
    zero_shot = evaluate(model, test, device, metric=False)
    print("zero-shot (aligned):", zero_shot["aligned"], flush=True)

    encoder = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
                            weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                        persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr / 10, args.lr],
                                                total_steps=args.epochs * len(loader), pct_start=0.1)
    best, history, t0 = float("inf"), [], time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, d in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            x, d = x.to(device, non_blocking=True), d.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = predict(model, x)
            loss = silog(out.float(), d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = evaluate(model, val, device, metric=True)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val": v,
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f}, validation AbsRel {v['metric']['abs_rel']:.3f}",
              flush=True)
        if v["metric"]["abs_rel"] < best:
            best = v["metric"]["abs_rel"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"stopping: {args.max_minutes} min limit reached")
            break

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    fine = evaluate(model, test, device, metric=True)
    result = {"task": "monocular depth", "model": MODEL, "input": "rgb_central", "label": "depth_m",
              "size": SIZE, "depth_range_m": [MIN_DEPTH, MAX_DEPTH],
              "test": {"zero_shot_aligned": zero_shot["aligned"], "fine_tuned_metric": fine["metric"],
                       "fine_tuned_aligned": fine["aligned"]},
              "best_validation_abs_rel": best, "epochs_run": len(history),
              "samples": {"train": len(train), "validation": len(val), "test": len(test)},
              "history": history, "args": vars(args)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_examples(model, test, out_dir, device)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
