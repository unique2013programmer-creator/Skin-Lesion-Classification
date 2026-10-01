# ISIC 2019 Skin Lesion Classification (ConvNeXt + TTA + Fold Ensembling v2)

[Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
[PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
[timm](https://img.shields.io/badge/timm-%E2%89%A50.9.12-orange)
[albumentations](https://img.shields.io/badge/albumentations-%E2%89%A51.3.1-blueviolet)
[Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-green)
[Task](https://img.shields.io/badge/Task-8--class%20classification-lightgrey)

A PyTorch pipeline for **8-class dermoscopic skin lesion classification** on the ISIC 2019 dataset. It fine-tunes an ImageNet-22k pretrained **ConvNeXt-Base** backbone using layer-wise learning-rate decay, progressive unfreezing, soft-target Focal Loss with MixUp/CutMix, and EMA weights. It evaluates on **stratified 5-fold cross-validation** with **out-of-fold (OOF) metrics**, **8x dihedral Test-Time Augmentation**, and **per-class logit adjustment** tuned for Macro F1. Inference is handled by a multi-fold `EnsemblePredictor`.


## Table of Contents

- [Key Architectural Features & Highlights](#key-architectural-features--highlights)
- [Dataset & Target Classes](#dataset--target-classes)
- [Configuration & Hyperparameters](#configuration--hyperparameters-config)
- [Evaluation & Performance Expectations](#evaluation--performance-expectations)
- [Installation & Prerequisites](#installation--prerequisites)
- [Execution Workflow](#execution-workflow)
- [Directory Structure](#directory-structure)
- [Roadmap](#roadmap)


## Key Architectural Features & Highlights

### ConvNeXt backbone with custom head
- Backbone: `convnext_base.fb_in22k_ft_in1k`, created through `timm.create_model(..., num_classes=0)`. The original classifier is removed and global pooling is kept.
- Stochastic depth through `drop_path_rate=0.2`.
- Custom classification head: `LayerNorm → Dropout(0.3) → Linear(num_features, 8)`, with `trunc_normal_(std=0.01)` weight init and zero bias.
- `channels_last` memory format on CUDA.

### Layer-wise LR decay + progressive unfreezing (no optimizer state resets)
- Parameters are mapped to **layer group IDs**: `0` = stem, `1..4` = ConvNeXt stages, `99` = head (plus final norms and any unmatched params).
- The same group IDs drive **both** the LR decay and the unfreeze schedule, so the two stay consistent.
- Per-group LR = `base_lr × layer_decay^(depth + 1 − group_id)`. The head trains at `base_lr`.
- Norm and bias parameters get **no weight decay**.
- The `AdamW` optimizer is built **once** over every parameter group. Unfreezing only toggles `requires_grad`, so AdamW momentum for layers that are already training is never discarded.

| Epoch (0-indexed) | Lowest trainable group | Trainable region |
|---|---|---|
| 0 | 99 | Head only |
| 1 | 4 | Stage 4 + head |
| 3 | 3 | Stages 3–4 + head |
| 5 | 2 | Stages 2–4 + head |
| 7 | 0 | Full network |

### Soft-target Focal Loss (MixUp / CutMix / label smoothing)
- `SoftTargetFocalLoss` takes a **(B, C) probability matrix** rather than integer labels:
  `loss = −Σ soft_target · α_c · (1 − p)^γ · log p`
- Label smoothing, MixUp, and CutMix all pass through a single soft-target path. This avoids the common shortcut of interpolating two hard-target losses, which is incorrect once label smoothing is involved.
- CutMix **recomputes λ from the actual (clipped) box area**, so the image and target mixing stay in sync.
- Each batch gets MixUp or CutMix with probability `mix_prob = 0.5`, split 50/50 between the two.

### Model EMA
- `ModelEMA` (decay `0.9995`) is updated after every optimizer step.
- **Raw and EMA weights are both validated each epoch.** The one with the higher Macro F1 is chosen, and its weights are checkpointed whenever the fold's best score improves.

### 8x dihedral TTA + stratified 5-fold ensembling
- TTA covers the full dihedral group (4 × `rot90` × 2 horizontal flips) and runs on **GPU tensors**, so it adds no image decoding cost.
- Validation during training runs **without** TTA for speed. OOF evaluation and inference use 8x TTA.
- `EnsemblePredictor` averages softmax probabilities over every fold checkpoint and every TTA view.

### OOF evaluation + per-class logit adjustment
- Each best fold checkpoint is reloaded to re-predict its own validation fold with TTA. The folds are combined into one OOF matrix.
- `tune_class_priors` runs **coordinate ascent** (3 rounds, grid `[-1.5, 1.5]` with 31 steps) over per-class log-priors to maximize Macro F1 on the **OOF predictions only**. The tuned vector is saved to `checkpoints/class_priors.npy` and reused at inference.

### Version-tolerant wrappers
- **albumentations:** `_try_build` tries the modern signature first and falls back to the legacy one, covering argument renames across 1.3 → 1.4 → 2.x for `CoarseDropout`, `Affine`, and `RandomResizedCrop`.
- **torch AMP:** `make_grad_scaler` / `autocast_ctx` use `torch.amp` when available and fall back to `torch.cuda.amp` otherwise.
- Gradients are unscaled **before** gradient clipping (`max_grad_norm = 1.0`).


## Dataset & Target Classes

- **Dataset:** ISIC 2019 Training set (Kaggle mirror `andrewmvd/isic-2019`)
- **Files used:** `ISIC_2019_Training_GroundTruth.csv` and `ISIC_2019_Training_Input/`
- **Samples:** 25,331 images. All 25,331 pass label normalization: rows are kept only if they have exactly one positive label among the 8 target classes (this drops `UNK`-only and multi-label rows) and a matching image file.

The ground-truth CSV and image directory are **discovered automatically**: the CSV is chosen by filename keywords, and the image directory is whichever folder holds the most `.jpg/.jpeg/.png` files. This handles layout changes across versions of the Kaggle mirror.

| Index | Class | Description | Count |
|---|---|---|---|
| 0 | MEL | Melanoma | 4,522 |
| 1 | NV | Melanocytic nevus | 12,875 |
| 2 | BCC | Basal cell carcinoma | 3,323 |
| 3 | AK | Actinic keratosis | 867 |
| 4 | BKL | Benign keratosis | 2,624 |
| 5 | DF | Dermatofibroma | 239 |
| 6 | VASC | Vascular lesion | 253 |
| 7 | SCC | Squamous cell carcinoma | 628 |

NV outnumbers DF by **about 54:1**.

### Class imbalance mitigation

| Strategy | Implementation |
|---|---|
| `WeightedRandomSampler` | Square-root inverse sampling (`1/√count`), with replacement and `num_samples = len(train_df)`. Rebalances the data without letting tiny classes dominate each epoch. |
| Square-root inverse class weighting | `α_c = (N / (C · n_c))^0.5`, normalized to mean 1, used as the per-class α in the loss. Full inverse weighting combined with focal loss over-corrects and hurts NV precision. |
| Focal Loss | `γ = 2.0`, which down-weights easy (mostly NV) examples. |
| Stratified K-Fold | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Folds are assigned once so every experiment shares the same splits. |


## Configuration & Hyperparameters (`Config`)

`Config` is a `dataclass` and the only place pipeline settings are defined.

### Data & Model

| Parameter | Value | Notes |
|---|---|---|
| `image_size` | `320` | 320×320 input |
| `batch_size` | `16` | Train loader, `drop_last=True` |
| `val_batch_size` | `32` | Used by the OOF loaders. The per-fold validation loader uses `batch_size`. |
| `num_workers` | `4` | |
| `num_folds` | `5` | Stratified |
| `folds_to_run` | `(0,)` | Set to `(0, 1, 2, 3, 4)` for the full ensemble |
| `model_name` | `convnext_base.fb_in22k_ft_in1k` | via `timm` |
| `pretrained` | `True` | |
| `drop_path_rate` | `0.2` | Stochastic depth |
| `head_dropout` | `0.3` | |

### Optimization

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | `AdamW` | `betas=(0.9, 0.999)` |
| `base_lr` | `3e-4` | Head LR. Backbone groups are scaled down by layer decay. |
| `layer_decay` | `0.75` | Resulting LRs: stem 7.12e-5 → stage 4 2.25e-4 → head 3.0e-4 |
| `weight_decay` | `0.05` | Not applied to norms or biases |
| Scheduler | `CosineAnnealingWarmRestarts` | Stepped per iteration (`epoch + step/steps`) |
| `scheduler_t0` / `scheduler_tmult` | `5` / `1` | |
| `min_lr` | `1e-6` | `eta_min` |
| `warmup_epochs` | `1` | Defined but not used by the training loop |
| `epochs` | `15` | |
| `max_grad_norm` | `1.0` | Clipping over trainable params |
| `mixed_precision` | `True` | AMP (CUDA only) |
| `channels_last` | `True` | CUDA only |
| `unfreeze_schedule` | `{0: 99, 1: 4, 3: 3, 5: 2, 7: 0}` | epoch → lowest trainable group |

### Regularization, Imbalance & Augmentation

| Parameter | Value |
|---|---|
| `label_smoothing` | `0.05` |
| `mixup_alpha` | `0.4` |
| `cutmix_alpha` | `1.0` |
| `mix_prob` | `0.5` |
| `focal_gamma` | `2.0` |
| `class_weight_power` | `0.5` (0 = none, 0.5 = sqrt-inverse, 1 = full inverse) |
| `use_weighted_sampler` | `True` |

**Training transforms (albumentations):** `LongestMaxSize(1.15×)` → `PadIfNeeded(reflect)` → `RandomResizedCrop(scale=0.7–1.0, ratio=0.8–1.25)` → `HorizontalFlip` / `VerticalFlip` / `RandomRotate90` / `Transpose` (p=0.5 each) → `Affine(scale 0.85–1.15, translate ±8%, rotate ±35°, shear ±12°, p=0.7)` → `OneOf[ColorJitter, RandomBrightnessContrast, HueSaturationValue]` (p=0.8) → `OneOf[GaussianBlur, MotionBlur, GaussNoise]` (p=0.25) → `CoarseDropout` (1–8 holes, p=0.5) → ImageNet `Normalize` → `ToTensorV2`.

**Validation transforms:** `LongestMaxSize` → `PadIfNeeded(reflect)` → `CenterCrop(320)` → `Normalize` → `ToTensorV2`.

### EMA, Early Stopping & Inference

| Parameter | Value |
|---|---|
| `use_ema` | `True` |
| `ema_decay` | `0.9995` |
| `early_stopping_patience` | `5` (on validation Macro F1, `min_delta=1e-4`) |
| `use_tta` | `True` |
| `tta_transforms` | `8` (full dihedral group) |
| `checkpoint_dir` | `./checkpoints` |
| Seed | `42` (fold *k* reseeds with `42 + k`) |


## Evaluation & Performance Expectations

### Metrics reported
`compute_metrics` reports **Accuracy, Balanced Accuracy, Macro F1, Weighted F1, quadratic-weighted Cohen's κ, and Macro OvR ROC-AUC**. The notebook also produces per-class classification reports, raw and row-normalized confusion matrices, and per-class / micro / macro ROC curves. **Macro F1** is the model-selection and checkpointing criterion.

### Reference expectations

| Metric | Typical range for strong ISIC 2019 solutions |
|---|---|
| Weighted Accuracy / Weighted F1 | **0.93 – 0.96** |
| Macro F1 (all 8 classes) | **0.80 – 0.88** |

**Why Macro F1 is lower:** Macro F1 weights every class equally, so the tiny **DF (239), VASC (253), and SCC (628)** classes carry as much weight as NV (12,875). A handful of errors on a 47-image validation slice moves the score noticeably. Results that claim **0.90+ Macro F1 across every class usually come from data leaking across folds**, for example multiple images of the same lesion landing in both train and validation. This pipeline reports both weighted and macro metrics so the difference is visible. All quoted numbers come from OOF predictions, where every prediction is made by a model that never saw that image.

### Results recorded in the notebook (fold 0 only, OOF, 8x TTA)

Only `folds_to_run = (0,)` has been executed so far (5,067 validation images). Best weights: **raw** (not EMA).

| Metric | argmax | + prior adjustment |
|---|---|---|
| Accuracy | 0.8642 | **0.8838** |
| Balanced Accuracy | 0.8562 | **0.8633** |
| Macro F1 | 0.8446 | **0.8636** |
| Weighted F1 | 0.8667 | **0.8835** |
| Cohen's κ (quadratic) | 0.8512 | **0.8649** |
| Macro ROC-AUC | 0.9744 | 0.9744 |

Micro ROC-AUC: **0.9807**. Logit adjustment added about **+1.9 Macro F1 points** with no extra training.

<details>
<summary>Per-class results (fold 0, argmax, TTA)</summary>

| Class | Precision | Recall | F1 | ROC-AUC | Support |
|---|---|---|---|---|---|
| MEL | 0.7140 | 0.8331 | 0.7690 | 0.9537 | 905 |
| NV | 0.9471 | 0.8687 | 0.9062 | 0.9772 | 2575 |
| BCC | 0.8967 | 0.9263 | 0.9112 | 0.9911 | 665 |
| AK | 0.7120 | 0.7816 | 0.7452 | 0.9785 | 174 |
| BKL | 0.8134 | 0.8552 | 0.8338 | 0.9810 | 525 |
| DF | 0.8696 | 0.8511 | 0.8602 | 0.9397 | 47 |
| VASC | 0.9038 | 0.9400 | 0.9216 | 0.9999 | 50 |
| SCC | 0.8264 | 0.7937 | 0.8097 | 0.9739 | 126 |

Tuned log-priors: `MEL −0.3, NV +0.3, BCC −0.4, AK −0.1, BKL 0.0, DF −0.5, VASC +0.2, SCC −1.0`
</details>

> **Note:** These numbers are from a single fold. The notebook estimates that the full 5-fold ensemble adds about +2–3 Macro F1 points.


## Installation & Prerequisites

### Environment
Tested in the notebook with **torch 2.10.0+cu128, timm 1.0.26, albumentations 2.0.8** on a CUDA GPU. The version-tolerant wrappers are meant to support older releases as well (albumentations ≥ 1.3.1, timm ≥ 0.9.12).

### 1. Install dependencies

`requirements.txt`:

```text
torch
timm>=0.9.12
albumentations>=1.3.1
opencv-python-headless
scikit-learn>=1.3
numpy
pandas
matplotlib
seaborn
kaggle
ipywidgets
```

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Kaggle API authentication

The notebook checks for credentials in this order:

**Option A: `kaggle.json`**
```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Option B: environment variables.** The notebook writes `~/.kaggle/kaggle.json` (mode `0600`) from these automatically:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### 3. Dataset location

| Environment | Data path |
|---|---|
| Kaggle | Attach `andrewmvd/isic-2019` as an input dataset. Cells 5 and 10 set `CFG.data_dir` to the first folder under `/kaggle/input`. |
| Local / Colab | Put the dataset under `./data/isic_2019/` (the default `Config.data_dir`) and **skip the two `/kaggle/input` cells**, since they assume a Kaggle runtime. |

Discovery is recursive, so any nesting under `data_dir` works as long as the ground-truth CSV and image folder are inside it.


## Execution Workflow

Run the notebook top to bottom. The stages:

```text
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 1. Environment &     │──▶│ 2. Data discovery &  │──▶│ 3. Stratified 5-fold │
│    Kaggle auth       │   │    label normalize   │   │    assignment        │
└──────────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                                  ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 6. Ensemble TTA      │◀──│ 5. OOF eval + TTA +  │◀──│ 4. Per-fold training │
│    inference         │   │    logit adjustment  │   │    (raw vs EMA)      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

**1. Environment & credentials** (Sections 1–3). Install packages, resolve Kaggle credentials, set seeds, and build `CFG`.

**2. Data discovery** (Section 4). Find the CSV and image directory automatically, convert one-hot labels to a single label, resolve image paths, and plot the class distribution.

**3. Fold assignment** (Section 5). Add a stratified `fold` column once, up front. Then smoke-test the augmentation pipelines, the loss sanity check, and the model forward pass (Sections 6–8).

**4. Training** (Sections 11–14). For each fold in `CFG.folds_to_run`:
```python
# Start with one fold to verify the loop end to end, then run the full ensemble:
CFG.folds_to_run = (0, 1, 2, 3, 4)
fold_results = [train_fold(df_data, fold, CFG) for fold in CFG.folds_to_run]
```
Each epoch applies the unfreeze schedule, trains with MixUp/CutMix and AMP, validates both raw and EMA weights, saves `checkpoints/convnext_base_fold{k}.pth` when Macro F1 improves, and applies early stopping.

**5. OOF metrics** (Sections 15–18). Reload each best checkpoint, re-predict its validation fold with 8x TTA, build the OOF matrix, then produce the classification report, confusion matrices, ROC-AUC, and Macro F1-tuned class priors (saved to `checkpoints/class_priors.npy`).

**6. Ensemble inference** (Section 19).
```python
ensemble = EnsemblePredictor(
    [result["checkpoint"] for result in fold_results], CFG, class_priors=priors
)
table = ensemble.predict_image("path/to/lesion.jpg")  # returns class probabilities, plots by default
```


## Directory Structure

```text
.
├── ISIC_2019_Skin_Lesion_Classification.ipynb   # End-to-end pipeline
├── requirements.txt
├── README.md
├── data/
│   └── isic_2019/                               # Default Config.data_dir
│       ├── ISIC_2019_Training_GroundTruth.csv
│       └── ISIC_2019_Training_Input/
│           └── ISIC_*.jpg
└── checkpoints/
    ├── convnext_base_fold0.pth                  # state_dict, config, fold, macro_f1, source (raw/ema)
    ├── convnext_base_fold1.pth
    ├── ...
    └── class_priors.npy                         # OOF-tuned per-class log-priors
```


## Roadmap

Ranked in the notebook by return on GPU hours:

1. **Run all 5 folds.** Ensembling is the largest single gain, typically +2–3 Macro F1 points.
2. **Backbone diversity.** Average `convnext_base` with `tf_efficientnetv2_m` or `swin_base_patch4_window7_224`.
3. **Resolution ladder.** Fine-tune the final epochs at 384 or 448.
4. **External data.** Add HAM10000 / BCN20000 for DF/VASC/SCC, deduplicated by lesion ID to avoid leakage.
5. **Group-aware splits.** Switch to `StratifiedGroupKFold` on lesion ID. ISIC contains multiple images per lesion, and ignoring that inflates every metric.
6. **Metadata fusion.** Concatenate age, sex, and anatomical site into the head.
7. **Calibration.** Apply temperature scaling on OOF before deployment, because focal loss and class weighting both hurt calibration.
```
