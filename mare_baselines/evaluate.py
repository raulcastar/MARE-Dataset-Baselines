"""
Final evaluation of the trained baselines, at the dataset's native resolution,
with everything needed to reproduce the numbers.

    python -m mare_baselines.evaluate --data /data --out /results [--split test]

For every run in <out> with a best.pt (segmentation runs and depth), writes
<out>/<run>/eval.json:
  - segmentation: the network sees the image at its training size; its logits
    are upsampled to the label's native size (640 x 480; SPAD 256 x 256) and
    compared with the original labels. Raw confusion matrices (overall and per
    sun-elevation bin), per-class IoU, precision and recall, mIoU.
  - depth: predictions upsampled to 640 x 480 and compared with depth_m;
    zero-shot (aligned) and fine-tuned (metric and aligned), as the mean over
    images and pooled over pixels, and the fine-tuned error by distance range.
  - provenance: code commit (MARE_GIT_COMMIT), dataset revision
    (MARE_DATASET_REVISION) and a fingerprint of the exact samples, pretrained
    weight snapshots, software versions and GPU.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import data, depth as depth_mod, seg

DISTANCE_BINS = [(0.5, 5), (5, 10), (10, 20), (20, 40), (40, 80)]


def fingerprint(items):
    """sha256 over the evaluated samples (episode/frame), so a result names its exact data."""
    h = hashlib.sha256()
    for ep_dir, stem, _ in items:
        h.update(f"{os.path.basename(ep_dir)}/{stem}\n".encode())
    return h.hexdigest()


def snapshots():
    """Hugging Face cache snapshots of the pretrained weights (repo -> revision hash)."""
    out = {}
    for repo in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*")):
        name = os.path.basename(repo)[len("models--"):].replace("--", "/")
        out[name] = sorted(os.listdir(os.path.join(repo, "snapshots"))) if os.path.isdir(
            os.path.join(repo, "snapshots")) else []
    return out


def environment():
    import segmentation_models_pytorch as smp
    import timm
    import transformers
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "segmentation_models_pytorch": smp.__version__, "timm": timm.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__}


def class_report(cm, names):
    tp = np.diag(cm).astype(np.float64)
    gt, pr = cm.sum(1), cm.sum(0)
    present = gt > 0
    per = {names[c]: {"iou": float(tp[c] / max(gt[c] + pr[c] - tp[c], 1)),
                      "precision": float(tp[c] / pr[c]) if pr[c] else None,
                      "recall": float(tp[c] / gt[c]), "pixels": int(gt[c])}
           for c in range(len(names)) if present[c]}
    return {"mIoU": float(np.mean([v["iou"] for v in per.values()])) if per else 0.0,
            "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)), "classes": per,
            "confusion_matrix": cm.tolist(), "confusion_matrix_order": names}


@torch.no_grad()
def eval_seg(run_dir, metrics, root, split, device):
    task, arch = metrics["task"], metrics.get("arch", "unet")
    cfg = seg.TASKS[task]
    meta = data.episodes(root, split)[0][1]
    if task == "shadow":
        names = [meta["shadow_classes"][str(k)] for k in range(cfg["classes"])]
    else:
        names = [c["name"] for c in sorted(meta["object_classes"], key=lambda c: c["id"])][:cfg["classes"]]
    # The weights come from best.pt, so the encoder's ImageNet weights need not be downloaded.
    model = {"unet": lambda: seg.smp.Unet("resnet34", encoder_weights=None, classes=cfg["classes"]),
             "deeplabv3plus": lambda: seg.smp.DeepLabV3Plus("resnet50", encoder_weights=None, classes=cfg["classes"]),
             "segformer": lambda: seg.smp.Segformer("mit_b2", encoder_weights=None, classes=cfg["classes"])}[arch]()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.to(device).eval()
    items = data.frames(root, split, [cfg["input"], cfg["label"]])
    n = cfg["classes"]
    cm = np.zeros((n, n), np.int64)
    cms = {b: np.zeros((n, n), np.int64) for b in seg.SUN_BINS}
    for ep_dir, stem, m in tqdm(items, desc=os.path.basename(run_dir), leave=False):
        img = data.as_rgb(data.load(ep_dir, cfg["input"], stem))
        lab = data.load(ep_dir, cfg["label"], stem).astype(np.int64)
        for i in cfg["ignore"]:
            lab[lab == i] = data.IGNORE
        x = data.resize(img, cfg["size"], nearest=False).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - seg.MEAN) / seg.STD).transpose(2, 0, 1).copy())[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = F.interpolate(logits.float(), size=lab.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()
        c = seg.confusion(pred.ravel(), lab.ravel(), n)
        cm += c
        for lo, hi in cms:
            if lo <= m["sun"]["elevation_deg"] < hi:
                cms[(lo, hi)] += c
    return {"task": task, "model": seg.ARCHS[arch][0], "arch": arch, "split": split,
            "resolution": "native (label size); network input " + "x".join(map(str, cfg["size"])),
            "samples": len(items), "samples_sha256": fingerprint(items),
            "scores": class_report(cm, names),
            "by_sun_elevation": {f"{lo}-{hi} deg": class_report(c, names) for (lo, hi), c in cms.items() if c.sum()}}


def depth_scores(pred, gt):
    """Per-image mean and pixel-pooled AbsRel, RMSE, delta1 (lists of per-image arrays)."""
    per_image = [depth_mod.depth_metrics(p, g) for p, g in zip(pred, gt)]
    p, g = np.concatenate(pred), np.concatenate(gt)
    return {"mean_over_images": {k: float(np.mean([r[k] for r in per_image])) for k in per_image[0]},
            "pooled_over_pixels": depth_mod.depth_metrics(p, g)}


@torch.no_grad()
def eval_depth(run_dir, root, split, device):
    from transformers import AutoModelForDepthEstimation
    items = data.frames(root, split, ["rgb_central", "depth_m"])
    base = AutoModelForDepthEstimation.from_pretrained(depth_mod.MODEL).to(device).eval()
    tuned = AutoModelForDepthEstimation.from_pretrained(depth_mod.MODEL)
    tuned.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    tuned.to(device).eval()
    zs, ft_m, ft_a, gts = [], [], [], []
    for ep_dir, stem, _ in tqdm(items, desc="depth", leave=False):
        img = data.as_rgb(data.load(ep_dir, "rgb_central", stem))
        gt = torch.from_numpy(data.load(ep_dir, "depth_m", stem).astype(np.float32)).to(device)
        x = data.resize(img, depth_mod.SIZE, nearest=False).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - depth_mod.MEAN) / depth_mod.STD).transpose(2, 0, 1).copy())[None].to(device)
        m = depth_mod.valid_mask(gt)
        if m.sum() < 100:
            continue
        outs = []
        for model in (base, tuned):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(pixel_values=x).predicted_depth
            outs.append(F.interpolate(o[:, None].float(), size=gt.shape, mode="bilinear", align_corners=False)[0, 0])
        zs.append(depth_mod.align_inverse(outs[0], gt, m)[m].cpu().numpy())
        ft_m.append((depth_mod.K / outs[1].clamp(min=1e-3))[m].cpu().numpy())
        ft_a.append(depth_mod.align_inverse(outs[1], gt, m)[m].cpu().numpy())
        gts.append(gt[m].cpu().numpy())
    g_all, p_all = np.concatenate(gts), np.concatenate(ft_m)
    by_distance = {}
    for lo, hi in DISTANCE_BINS:
        sel = (g_all >= lo) & (g_all < hi)
        if sel.sum():
            by_distance[f"{lo}-{hi} m"] = {**depth_mod.depth_metrics(p_all[sel], g_all[sel]),
                                           "pixel_share": float(sel.mean())}
    return {"task": "monocular depth", "model": depth_mod.MODEL, "split": split,
            "resolution": "native 640x480; network input " + "x".join(map(str, depth_mod.SIZE)),
            "depth_range_m": [depth_mod.MIN_DEPTH, depth_mod.MAX_DEPTH], "samples": len(gts),
            "samples_sha256": fingerprint(items),
            "zero_shot_aligned": depth_scores(zs, gts), "fine_tuned_metric": depth_scores(ft_m, gts),
            "fine_tuned_aligned": depth_scores(ft_a, gts), "fine_tuned_metric_by_distance": by_distance}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="/data")
    parser.add_argument("--out", default="/results")
    parser.add_argument("--split", default="test", choices=list(data.SPLITS))
    parser.add_argument("--runs", nargs="*", help="run folders to evaluate (default: all with best.pt)")
    args = parser.parse_args()
    device = "cuda"
    provenance = {"code_commit": os.environ.get("MARE_GIT_COMMIT") or None,
                  "dataset_revision": os.environ.get("MARE_DATASET_REVISION") or None,
                  "pretrained_snapshots": snapshots(), "environment": environment()}
    runs = args.runs or sorted(os.path.basename(os.path.dirname(p))
                               for p in glob.glob(os.path.join(args.out, "*", "best.pt")))
    for run in runs:
        run_dir = os.path.join(args.out, run)
        metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
        if run == "depth":
            result = eval_depth(run_dir, args.data, args.split, device)
            headline = f"AbsRel {result['fine_tuned_metric']['mean_over_images']['abs_rel']:.3f}"
        else:
            result = eval_seg(run_dir, metrics, args.data, args.split, device)
            headline = f"mIoU {result['scores']['mIoU']:.3f}"
        result["training"] = {"args": metrics["args"], "epochs_run": metrics["epochs_run"],
                              "best_validation": metrics.get("best_validation_mIoU",
                                                             metrics.get("best_validation_abs_rel"))}
        result["provenance"] = provenance
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"{run}: {headline} (native resolution, {args.split})", flush=True)


if __name__ == "__main__":
    main()
