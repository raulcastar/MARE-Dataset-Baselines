"""
Run a trained segmentation model on a folder of images (for example real
camera frames) and save its predictions.

    python -m mare_baselines.predict --run object --images /data --out /results/predictions

Each image is centre-cropped to the training aspect ratio (4:3), resized to the
network input, and greyscale images are repeated to three channels. Writes, per
image, <out>/<run>/<name>.png: the class index of every pixel (uint8, at the
network resolution), and <out>/<run>/<name>_input.png: the cropped input.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from . import data, seg


def crop_to(img, size):
    """Centre crop of img (H x W x 3) to the aspect ratio of size (width, height)."""
    h, w = img.shape[:2]
    target = size[0] / size[1]
    if w / h > target:
        cw = int(round(h * target))
        return img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    ch = int(round(w / target))
    return img[(h - ch) // 2:(h - ch) // 2 + ch]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder in --results, e.g. object or shadow_segformer")
    parser.add_argument("--images", default="/data", help="folder of images (searched recursively)")
    parser.add_argument("--results", default="/results")
    parser.add_argument("--out", default="/results/predictions")
    args = parser.parse_args()

    run_dir = os.path.join(args.results, args.run)
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    task, arch = metrics["task"], metrics.get("arch", "unet")
    cfg = seg.TASKS[task]
    model = {"unet": lambda: seg.smp.Unet("resnet34", encoder_weights=None, classes=cfg["classes"]),
             "deeplabv3plus": lambda: seg.smp.DeepLabV3Plus("resnet50", encoder_weights=None, classes=cfg["classes"]),
             "segformer": lambda: seg.smp.Segformer("mit_b2", encoder_weights=None, classes=cfg["classes"])}[arch]()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.to("cuda").eval()

    out = os.path.join(args.out, args.run)
    os.makedirs(out, exist_ok=True)
    paths = sorted(p for ext in ("png", "jpg", "jpeg") for p in
                   glob.glob(os.path.join(args.images, "**", f"*.{ext}"), recursive=True))
    for path in paths:
        img = data.as_rgb(np.asarray(Image.open(path)))
        img = data.resize(crop_to(img, cfg["size"]), cfg["size"], nearest=False)
        x = (img.astype(np.float32) / 255.0 - seg.MEAN) / seg.STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
        name = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(pred).save(os.path.join(out, f"{name}.png"))
        Image.fromarray(img).save(os.path.join(out, f"{name}_input.png"))
    print(f"{args.run}: {len(paths)} images -> {out}")


if __name__ == "__main__":
    main()
