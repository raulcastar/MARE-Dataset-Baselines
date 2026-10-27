"""
Samples of the MARE Lunar Rover Dataset, from its folder layout:

    <root>/samples/<episode>/episode.json
    <root>/samples/<episode>/<field>/<frame>.png|.npy      (rgb_central, object, depth_m, ...)

This is the generator's output, and what unpack_hf.py makes of the Hugging Face
shards. Splits are by map, as on the Hub, so a model is tested on terrain it
never saw.
"""
import json
import os

import numpy as np
from PIL import Image

SPLITS = {"train": ["map_000", "map_001", "map_002", "map_003", "map_004", "map_005"],
          "validation": ["map_006"], "test": ["map_007"]}
IGNORE = 255


def episodes(root, split):
    """Episode folders of a split, with their metadata."""
    out = []
    samples = os.path.join(root, "samples")
    for ep in sorted(os.listdir(samples)):
        meta = json.load(open(os.path.join(samples, ep, "episode.json")))
        name = meta["map"]["name"] if isinstance(meta.get("map"), dict) else meta.get("map")
        if name in SPLITS[split]:
            out.append((os.path.join(samples, ep), meta))
    if not out:
        raise FileNotFoundError(f"no {split} episodes under {samples}")
    return out


def frames(root, split, fields):
    """(episode dir, frame stem, episode meta) of every sample that has all `fields`."""
    out = []
    for ep_dir, meta in episodes(root, split):
        names = None
        for f in fields:
            have = {n.rsplit(".", 1)[0] for n in os.listdir(os.path.join(ep_dir, f))} \
                if os.path.isdir(os.path.join(ep_dir, f)) else set()
            names = have if names is None else names & have
        out += [(ep_dir, stem, meta) for stem in sorted(names)]
    return out


def load(ep_dir, field, stem):
    path = os.path.join(ep_dir, field, stem)
    if os.path.isfile(path + ".npy"):
        return np.load(path + ".npy")
    return np.asarray(Image.open(path + ".png"))


def resize(a, size, nearest):
    """Resizes an image or label array to size = (width, height)."""
    if a.shape[1] == size[0] and a.shape[0] == size[1]:
        return a
    if a.dtype in (np.float32, np.float64):
        return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.NEAREST))
    return np.asarray(Image.fromarray(a).resize(size, Image.NEAREST if nearest else Image.BILINEAR))


def as_rgb(a):
    return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a[..., :3]
