# MARE-Dataset-Baselines

Baseline models for the [MARE Lunar Rover Dataset](https://huggingface.co/datasets/raulcastar/mare-lunar-rover-dataset):
synthetic lunar rover camera frames with pixel-accurate depth, object and shadow
labels, recorded in the [MARE simulator](https://github.com/raulcastar/MARE-Lunar-Simulator)
([paper](https://arxiv.org/abs/2610.28465)).

| Baseline | Input → target | Model | Script |
|---|---|---|---|
| Object segmentation (10 classes) | `rgb_central` → `object` | U-Net, ResNet-34 encoder (ImageNet) | `mare_baselines.seg --task object` |
| Shadow segmentation (6 classes) | `rgb_central` → `shadow` | U-Net, ResNet-34 encoder (ImageNet) | `mare_baselines.seg --task shadow` |
| Object segmentation on the SPAD camera | `spad` → `spad_object` | U-Net, ResNet-34 encoder (ImageNet) | `mare_baselines.seg --task spad_object` |
| Object segmentation on the rear camera | `rear_rgb` → `rear_object` | U-Net, ResNet-34 encoder (ImageNet) | `mare_baselines.seg --task rear_object` |
| Shadow segmentation on the rear and SPAD cameras | `rear_rgb` → `rear_shadow`, `spad` → `spad_shadow` | U-Net, ResNet-34 encoder (ImageNet) | `mare_baselines.seg --task rear_shadow` / `spad_shadow` |
| Model comparison on objects and shadows | as above | DeepLabV3+ (ResNet-50), SegFormer-B2 | `mare_baselines.seg --arch deeplabv3plus` / `segformer` |
| Monocular depth | `rgb_central` → `depth_m` | Depth Anything V2-Small, zero-shot and fine-tuned | `mare_baselines.depth` |

Every model trains on the six training maps, keeps the epoch with the best
validation score (map_006) and is reported on the test map (map_007), terrain
it never saw. Results, with per-class scores and example predictions, are in
[`results/`](results).

## Results

Test map (map_007, 585 frames, sun elevations of 8°, 16° and 20°), at the
dataset's native resolution, from `results/*/eval.json`.

**Segmentation, per-class IoU.** Best front-camera model in bold.

| Class | U-Net | DeepLabV3+ | SegFormer-B2 | Rear (U-Net) | SPAD (U-Net) |
|---|---|---|---|---|---|
| *Objects* | | | | | |
| Sky | **0.99** | **0.99** | **0.99** | 0.99 | 0.99 |
| Landscape | **0.90** | **0.90** | **0.90** | 0.88 | 0.88 |
| Small rock | **0.63** | 0.60 | 0.61 | 0.64 | 0.43 |
| Boulder | 0.43 | 0.39 | **0.49** | 0.54 | 0.33 |
| Crater | 0.46 | 0.46 | **0.48** | 0.25 | 0.29 |
| Crater rim | 0.21 | 0.19 | **0.23** | 0.15 | 0.17 |
| Wheel track | 0.14 | 0.15 | **0.16** | 0.94 | 0.10 |
| Sun | **0.75** | 0.65 | 0.58 | 0.74 | 0.47 |
| Sun glare | **0.84** | 0.81 | 0.82 | 0.86 | 0.69 |
| **mIoU** | **0.60** | 0.57 | 0.59 | 0.67 | 0.48 |
| mIoU, sun 0–10° | **0.57** | 0.54 | 0.56 | 0.65 | 0.42 |
| mIoU, sun 10–20° | **0.68** | 0.66 | 0.66 | 0.68 | 0.57 |
| *Shadows* | | | | | |
| Sky | **1.00** | **1.00** | **1.00** | 0.99 | 0.99 |
| Sunlit ground | **0.98** | 0.96 | 0.96 | 0.98 | 0.93 |
| Shadowed ground | **0.71** | 0.52 | 0.51 | 0.61 | 0.33 |
| Sunlit object | **0.72** | 0.67 | 0.66 | 0.76 | 0.48 |
| Shadowed object | **0.74** | 0.70 | 0.70 | 0.76 | 0.56 |
| Rover shadow | **0.99** | 0.98 | 0.98 | 0.98 | 0.97 |
| **mIoU** | **0.85** | 0.80 | 0.80 | 0.85 | 0.71 |
| mIoU, sun 0–10° | **0.82** | 0.77 | 0.77 | 0.82 | 0.67 |
| mIoU, sun 10–20° | **0.87** | 0.82 | 0.82 | 0.87 | 0.73 |

**Monocular depth** (Depth Anything V2-Small, ground truth 0.5–80 m):

| Setting | AbsRel | RMSE | δ < 1.25 |
|---|---|---|---|
| Zero-shot, scale and shift aligned per image | 0.177 | 12.61 m | 0.841 |
| Fine-tuned, metric | **0.027** | **1.63 m** | **0.984** |
| Fine-tuned, aligned | 0.029 | 3.84 m | 0.974 |

The fine-tuned error grows with distance: AbsRel 0.016 below 5 m (89 % of the
pixels), 0.079 at 5–10 m, 0.117 at 10–20 m, 0.164 at 20–40 m and 0.173 at
40–80 m.

**Real images.** `results/predictions_spice_hl3/` holds the predictions of the
object and shadow models on 84 frames of
[SPICE-HL3](https://zenodo.org/records/13970078), applied without adaptation;
there are no labels for them, so they are a qualitative check.

## Trained checkpoints

The trained models are on Hugging Face at
[raulcastar/mare-lunar-baselines](https://huggingface.co/raulcastar/mare-lunar-baselines), one folder per run
with the weights (`model.safetensors`) and their configuration and test scores (`config.json`). To run one on
your own images:

```bash
hf download raulcastar/mare-lunar-baselines --local-dir results/hub
MARE_DATA=/path/to/images docker compose run --rm train mare_baselines.predict --run object --results /results/hub
```

`mare_baselines.export_hub` produces that folder from the training results.

## Running them

Needs Docker with GPU access (NVIDIA Container Toolkit) and the dataset.

**1. The dataset.** Either the generator's output folder (it has `samples/`), or
the Hugging Face download unpacked into that layout:

```bash
hf download raulcastar/mare-lunar-rover-dataset --repo-type dataset --local-dir mare_hf
mkdir -p mare_data
MARE_DATA="$PWD/mare_data" docker compose run --rm -v "$PWD/mare_hf:/hf:ro" -v "$PWD/mare_data:/out" \
    train mare_baselines.unpack_hf /hf /out          # writes mare_data/samples/...
```

**2. Train and evaluate**, one baseline or all of them:

```bash
export MARE_DATA=/path/to/mare_data
docker compose run --rm train mare_baselines.seg --task object
./run_all.sh                                          # the U-Net and depth baselines, one after another
./run_models.sh                                       # DeepLabV3+ and SegFormer, then the figures
```

Each run writes `results/<task>/metrics.json` (test scores, per class and by sun
elevation for segmentation; history; arguments), `examples.png` (input | ground
truth | prediction) and `best.pt` (the checkpoint, not in git). Options: `--epochs`,
`--batch`, `--lr`, `--max-minutes` (a hard limit on training time), `--limit N`
(N samples per split, for a quick test).

**3. Final evaluation** (the numbers to report): every trained model on the test
map at the dataset's native resolution, with the full record of how they were
obtained, in `results/<run>/eval.json`:

```bash
MARE_DATASET_REVISION=<dataset commit on the Hub> ./evaluate.sh
```

It records per-class IoU, precision and recall and the raw confusion matrices
(overall and by sun elevation); for depth, zero-shot and fine-tuned errors per
image and pooled over pixels, and by distance range; and the provenance: this
repository's commit (flagged if the code has uncommitted changes, so commit
first), the dataset revision and a hash of the exact samples evaluated, the
pretrained weight snapshots, and the software versions and GPU. The training
settings and per-epoch history are in `metrics.json`. GPU training is not
bit-exact between runs, even with a fixed seed: report the mean and spread of
several seeds (`--seed`) for final numbers.

**4. Figures:** `docker compose run --rm train mare_baselines.plots` (training
curves) and `mare_baselines.figures` (per-class and per-camera comparisons, class
frequency, sun elevation), from the saved results.

Without Docker: `pip install torch torchvision` (CUDA build) and
`pip install -r requirements.txt`, then `python -m mare_baselines.seg --task object
--data /path/to/mare_data --out results`.

## Details

- **Segmentation:** images resized to 512 × 384 (SPAD: 256 × 256); horizontal
  flips and exposure jitter; cross-entropy with square-root median-frequency
  class weights plus Dice loss; AdamW, one-cycle schedule, bf16; peak learning
  rate 3e-4 for the CNNs (U-Net, DeepLabV3+) and 6e-5 for SegFormer, each
  model's standard recipe. Object class 0
  (unlabeled) and the SPAD's rows outside the central camera (255) are ignored.
  mIoU is over the classes present in the test map.
- **Depth:** images at 518 × 392; ground truth from 0.5 to 80 m (no sky).
  Zero-shot, Depth Anything predicts relative inverse depth, aligned per image to
  the ground truth with a least-squares scale and shift before scoring. Fine-tuned,
  it predicts metric depth (10 / output) with a scale-invariant log loss, scored
  as it is (metric) and aligned. Metrics: AbsRel, RMSE, δ < 1.25.
- Seeds are fixed (`--seed`), but GPU training is not bit-exact between runs.

## License

The code is released under the [MIT License](LICENSE). The dataset is CC BY 4.0. The trained
U-Net, DeepLabV3+ and depth weights are CC BY 4.0; the SegFormer weights derive from NVIDIA's MiT-B2
ImageNet weights and follow the [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
(non-commercial use).

## Citation

If you use the dataset or these baselines, please cite the MARE paper:

```bibtex
@misc{castilla2026mare,
  title         = {{MARE}: An Open-Source Lunar Rover Simulator with Shadow Ground Truth for Perception and Autonomy},
  author        = {Castilla-Arquillo, Ra{\'u}l and Olivares-Mendez, Miguel},
  year          = {2026},
  eprint        = {2610.28465},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.28465}
}
```
