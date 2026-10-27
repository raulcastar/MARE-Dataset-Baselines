"""
Export the trained checkpoints for a Hugging Face model repository.

    python -m mare_baselines.export_hub --results /results --out /results/hub_export

For every run with best.pt and eval.json, writes <out>/<run>/model.safetensors
(the same weights, verified tensor by tensor) and <out>/<run>/config.json (task,
model, input and label streams, class names, test scores at native resolution,
training settings and provenance), and the model card <out>/README.md.
mare_baselines.predict loads either layout.
"""
import argparse
import glob
import json
import os

import torch
from safetensors.torch import load_file, save_file

from . import seg

REPO = "raulcastar/mare-lunar-baselines"
DATASET = "raulcastar/mare-lunar-rover-dataset"
CARD = """---
license: other
license_name: mixed
datasets:
- {dataset}
tags:
- semantic-segmentation
- depth-estimation
- lunar
- planetary-robotics
- synthetic-data
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
---

# MARE Lunar Rover Baselines

Segmentation and depth models trained on the synthetic
[MARE Lunar Rover Dataset](https://huggingface.co/datasets/{dataset}), recorded in the
[MARE simulator](https://github.com/raulcastar/MARE-Lunar-Simulator). They are the baselines of the
[MARE paper](https://arxiv.org/abs/2610.28465); the training and evaluation code is in
[MARE-Dataset-Baselines](https://github.com/raulcastar/MARE-Dataset-Baselines).

All models were trained on maps 000-005, selected on map 006 and scored on map 007, terrain never seen
in training. Scores are at the dataset's native resolution.

| Folder | Task | Model | Input | Test |
|---|---|---|---|---|
{rows}

Each folder holds `model.safetensors` and `config.json` (class names, input size, per-class test IoU,
training settings, code commit and dataset revision).

## Usage

Segmentation models are [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
networks; inputs are RGB images resized to `input_size` (width, height) and normalized with the ImageNet mean
and standard deviation.

```python
import json
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

config = json.load(open(hf_hub_download("{repo}", "object/config.json")))
model = smp.Unet("resnet34", encoder_weights=None, classes=config["classes"])
model.load_state_dict(load_file(hf_hub_download("{repo}", "object/model.safetensors")))
model.eval()
```

Use `smp.DeepLabV3Plus("resnet50", ...)` and `smp.Segformer("mit_b2", ...)` for the other architectures.
The depth model is
[Depth Anything V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) fine-tuned for
metric depth: load the base model with `transformers.AutoModelForDepthEstimation`, then
`depth/model.safetensors`; metric depth in metres is `10 / predicted_depth`.

With the training repository, `mare_baselines.predict` runs any of these models on a folder of images:

```bash
hf download {repo} --local-dir results/hub
docker compose run --rm train mare_baselines.predict --run object --results /results/hub --images /data
```

## License

The U-Net, DeepLabV3+ and depth weights are released under CC BY 4.0. The SegFormer weights derive from
NVIDIA's MiT-B2 ImageNet weights and are subject to the
[NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) (non-commercial use).

## Citation

```bibtex
@misc{{castilla2026mare,
  title         = {{{{MARE}}: An Open-Source Lunar Rover Simulator with Shadow Ground Truth for Perception and Autonomy}},
  author        = {{Castilla-Arquillo, Ra{{\\'u}}l and Olivares-Mendez, Miguel}},
  year          = {{2026}},
  eprint        = {{2610.28465}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.RO}},
  url           = {{https://arxiv.org/abs/2610.28465}}
}}
```
"""


def card_row(run, config):
    if run == "depth":
        m = config["test"]["fine_tuned_metric"]
        return (f"| `{run}` | monocular depth | Depth Anything V2-Small, fine-tuned | `{config['input']}` | "
                f"AbsRel {m['abs_rel']:.3f}, RMSE {m['rmse']:.2f} m, δ<1.25 {m['delta1']:.3f} |")
    task = "shadows" if config["task"].endswith("shadow") else "objects"
    return f"| `{run}` | {task} | {config['model']} | `{config['input']}` | mIoU {config['test']['mIoU']:.3f} |"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="/results")
    parser.add_argument("--out", default="/results/hub_export")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results, "*", "best.pt"))):
        run_dir = os.path.dirname(path)
        run = os.path.basename(run_dir)
        if not os.path.isfile(os.path.join(run_dir, "eval.json")):
            print(f"{run}: no eval.json, skipped")
            continue
        metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
        evaluation = json.load(open(os.path.join(run_dir, "eval.json")))
        out = os.path.join(args.out, run)
        os.makedirs(out, exist_ok=True)

        state = torch.load(path, map_location="cpu")
        state = {k: v.contiguous() for k, v in state.items()}
        target = os.path.join(out, "model.safetensors")
        save_file(state, target)
        saved = load_file(target)
        assert saved.keys() == state.keys() and all(torch.equal(saved[k], state[k]) for k in state), run

        if run == "depth":
            config = {"task": "monocular depth", "base_model": evaluation["model"], "input": "rgb_central",
                      "label": "depth_m", "output": "metric depth in metres = 10 / predicted_depth",
                      "depth_range_m": evaluation["depth_range_m"],
                      "test": {k: evaluation[k]["mean_over_images"]
                               for k in ("zero_shot_aligned", "fine_tuned_metric", "fine_tuned_aligned")}}
        else:
            cfg = seg.TASKS[metrics["task"]]
            config = {"task": metrics["task"], "arch": metrics.get("arch", "unet"), "model": evaluation["model"],
                      "input": cfg["input"], "label": cfg["label"], "input_size": list(cfg["size"]),
                      "classes": cfg["classes"], "class_names": evaluation["scores"]["confusion_matrix_order"],
                      "ignored_labels": cfg["ignore"],
                      "test": {"mIoU": evaluation["scores"]["mIoU"],
                               "iou": {c: v["iou"] for c, v in evaluation["scores"]["classes"].items()}}}
        config.update({"split": evaluation["split"], "resolution": evaluation["resolution"],
                       "training": evaluation["training"],
                       "code_commit": evaluation["provenance"]["code_commit"],
                       "dataset_revision": evaluation["provenance"]["dataset_revision"]})
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        rows.append(card_row(run, config))
        print(f"{run}: {os.path.getsize(target) / 1e6:.1f} MB, weights verified")

    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(CARD.format(dataset=DATASET, repo=REPO, rows="\n".join(rows)))


if __name__ == "__main__":
    main()
