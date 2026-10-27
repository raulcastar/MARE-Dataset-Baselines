"""
Comparison figures from the runs' metrics.json (no GPU needed).

    python -m mare_baselines.figures --out /results

    figure_iou_by_sensor.png     per-class test IoU: RGB, SPAD and rear camera
    figure_iou_by_model.png      per-class test IoU of the segmentation models on RGB (when trained)
    figure_iou_vs_frequency.png  each class's test IoU against its share of the training pixels
    figure_sun_elevation.png     test mIoU by sun elevation, per task
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plots import INK, INK_2, SURFACE, style  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]      # colour-blind-safe series colours, in order
CLASSES = ["sky", "landscape", "small_rock", "boulder", "crater", "crater_rim", "wheel_track", "sun", "sun_glare"]


def load(out, run):
    path = os.path.join(out, run, "metrics.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def grouped_bars(runs, path, title):
    """Horizontal grouped bars: one row per class, one bar per run (<= 3 runs)."""
    runs = [(label, m) for label, m in runs if m]
    if len(runs) < 2:
        return False
    classes = [c for c in CLASSES if any(c in m["test"]["iou"] for _, m in runs)]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(classes) + 1.2), facecolor=SURFACE)
    style(ax)
    h = 0.8 / len(runs)
    for k, (label, m) in enumerate(runs):
        ys = [i + (k - (len(runs) - 1) / 2) * h for i in range(len(classes))]
        vals = [m["test"]["iou"].get(c, 0.0) for c in classes]
        ax.barh(ys, vals, height=h * 0.85, color=SLOTS[k], label=f"{label} (mIoU {m['test']['mIoU']:.2f})")
    ax.set_yticks(range(len(classes)), [c.replace("_", " ") for c in classes], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("test IoU", color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def iou_vs_frequency(m, path):
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURFACE)
    style(ax)
    pts = [(m["train_class_frequency"][c] * 100, v, c) for c, v in m["test"]["iou"].items()
           if m["train_class_frequency"].get(c, 0) > 0]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5,
               zorder=3)
    for f, v, c in pts:
        ax.annotate(c.replace("_", " "), (f, v), textcoords="offset points", xytext=(6, -3), fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("share of training pixels (%, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("test IoU", color=INK_2, fontsize=9)
    ax.set_title(f"Object segmentation ({m['model'].split(',')[0]}): IoU against class frequency", color=INK,
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def sun_elevation(runs, path):
    runs = [(label, m) for label, m in runs if m and m.get("test_by_sun_elevation")]
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    style(ax)
    bins = sorted({b for _, m in runs for b in m["test_by_sun_elevation"]}, key=lambda b: float(b.split("-")[0]))
    for k, (label, m) in enumerate(runs):
        xs = [i for i, b in enumerate(bins) if b in m["test_by_sun_elevation"]]
        ys = [m["test_by_sun_elevation"][bins[i]]["mIoU"] for i in xs]
        ax.plot(xs, ys, "-o", color=SLOTS[k], linewidth=1.5, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        ax.annotate(f"{label}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(len(bins)), [f"sun {b.replace(' deg', '°')}" for b in bins], color=INK_2)
    ax.set_xlim(-0.3, len(bins) - 0.3)
    ax.set_ylabel("test mIoU", color=INK_2, fontsize=9)
    ax.set_title("Segmentation is harder with a low sun", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="/results")
    args = parser.parse_args()
    o = args.out
    made = []
    if grouped_bars([("RGB", load(o, "object")), ("SPAD", load(o, "spad_object")), ("rear RGB", load(o, "rear_object"))],
                    os.path.join(o, "figure_iou_by_sensor.png"), "Object segmentation (U-Net) per class, by camera"):
        made.append("figure_iou_by_sensor.png")
    if grouped_bars([("U-Net", load(o, "object")), ("DeepLabV3+", load(o, "object_deeplabv3plus")),
                     ("SegFormer-B2", load(o, "object_segformer"))],
                    os.path.join(o, "figure_iou_by_model.png"), "Object segmentation on RGB per class, by model"):
        made.append("figure_iou_by_model.png")
    if load(o, "object"):
        iou_vs_frequency(load(o, "object"), os.path.join(o, "figure_iou_vs_frequency.png"))
        made.append("figure_iou_vs_frequency.png")
    if sun_elevation([("objects, RGB", load(o, "object")), ("shadows, RGB", load(o, "shadow")),
                      ("objects, SPAD", load(o, "spad_object"))], os.path.join(o, "figure_sun_elevation.png")):
        made.append("figure_sun_elevation.png")
    print("\n".join(made))


if __name__ == "__main__":
    main()
