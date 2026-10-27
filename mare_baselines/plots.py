"""
Training curves from the runs' metrics.json.

    python -m mare_baselines.plots --out /results

Writes, per run, <run>/curves.png (training loss and validation score per epoch,
two panels) and, per task, curves_<task>.png comparing its models on one
validation-score panel. The best epoch (the checkpoint kept) is marked.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per model (colour-blind-safe palette), the same in every figure.
SERIES = {"unet": "#2a78d6", "deeplabv3plus": "#eb6834", "segformer": "#1baf7a", "depth": "#2a78d6"}
LABELS = {"unet": "U-Net (ResNet-34)", "deeplabv3plus": "DeepLabV3+ (ResNet-50)", "segformer": "SegFormer-B2",
          "depth": "Depth Anything V2-S"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=9)


def series(m):
    """(arch key, epochs, losses, validation scores, score name, higher is better)."""
    h = m["history"]
    epochs = [r["epoch"] for r in h]
    loss = [r["loss"] for r in h]
    if "val_mIoU" in h[0]:
        return m.get("arch", "unet"), epochs, loss, [r["val_mIoU"] for r in h], "validation mIoU", True
    return "depth", epochs, loss, [r["val"]["metric"]["abs_rel"] for r in h], "validation AbsRel (metric)", False


def run_figure(m, path, title):
    arch, epochs, loss, val, name, higher = series(m)
    colour = SERIES[arch]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    for ax in (a, b):
        style(ax)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    a.plot(epochs, loss, color=colour, linewidth=1.5)
    a.set_title("training loss", color=INK, fontsize=10, loc="left")
    b.plot(epochs, val, color=colour, linewidth=1.5)
    i = max(range(len(val)), key=lambda k: val[k] if higher else -val[k])
    b.plot(epochs[i], val[i], "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.annotate(f"best {val[i]:.3f} (epoch {epochs[i]})", (epochs[i], val[i]), textcoords="offset points",
               xytext=(0, 9 if higher else -15), ha="center", fontsize=8.5, color=INK)
    b.set_title(name, color=INK, fontsize=10, loc="left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def task_figure(runs, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    style(ax)
    for m in sorted(runs, key=lambda m: list(SERIES).index(m.get("arch", "unet"))):
        arch, epochs, _, val, name, higher = series(m)
        best = max(val) if higher else min(val)
        ax.plot(epochs, val, color=SERIES[arch], linewidth=1.5, label=f"{LABELS[arch]} (best {best:.3f})")
    ax.set_xlabel("epoch", color=INK_2, fontsize=9)
    ax.set_title(f"{title}: {name}", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="/results")
    args = parser.parse_args()
    by_task = {}
    for path in sorted(glob.glob(os.path.join(args.out, "*", "metrics.json"))):
        m = json.load(open(path))
        run = os.path.basename(os.path.dirname(path))
        run_figure(m, os.path.join(os.path.dirname(path), "curves.png"), f"{run}: {m['model']}")
        if "_seed" not in run:
            by_task.setdefault(m["task"], []).append(m)
        print("curves:", run)
    for task, runs in by_task.items():
        if len(runs) > 1:
            task_figure(runs, os.path.join(args.out, f"curves_{task}.png"), task)
            print("comparison:", task)


if __name__ == "__main__":
    main()
