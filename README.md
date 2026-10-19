# Processing-Robust AI Image Source Attribution

A reproducible **single-model, single-view** solution for CE7454 Project 1:
AI-Generated Image Detection and Source Attribution.

**Final CodaBench test accuracy: 97.133% (0.97133), final rank: 30.**
These are the author's confirmed final server results. Local validation is a
separate measurement; test labels and a test confusion matrix are unavailable.

The model joins RGB with neighboring-pixel residuals in one modified ResNet-18,
pools spatial means and standard deviations, and is trained from scratch
through a crop, interpolation, JPEG and noise curriculum. It contains
**11,173,635 parameters**, below the challenge's 12.3M limit. Inference uses the
unchanged 512 × 512 RGB image. There is no ensemble, TTA or test-time fitting.

## Release contents

| Location | Contents |
|---|---|
| `src/ai_attribution/` | Model, exact preprocessing, sampling, training, evaluation and prediction |
| `configs/` | Six curriculum stages, three final ablations, original 12k/3k manifest |
| `results/` | Recorded validation bank, training curves, final score and release checks |
| `assets/` | Official final filenames and model/prediction checksums |
| `tools/` | Curriculum replay, three-GPU comparison, asset verification and paper plots |
| `tests/` | Meaningful CPU tests for sampling, model, optimization and submission rules |
| `docs/` | Detailed reproducibility protocol and experimental interpretation |

The companion **`ai-image-source-attribution_model_assets.zip`** contains the
selected final checkpoint, its 110-epoch parent, and the accepted predictions.
Extract its contents into this repository's root. Model weights are kept out
of the source ZIP and Git history. Datasets are obtained from the challenge;
no images or course lecture files are redistributed.

When publishing this repository, attach the companion ZIP as a GitHub Release
asset so readers can download it separately from the source checkout.

## 1. Create the environment

Run commands from the repository root. Linux/WSL with an NVIDIA GPU is the
primary training platform. The release's complete 1,500-image inference was
verified with Python 3.12, PyTorch 2.8.0 + CUDA 12.8, torchvision 0.23.0,
NumPy 2.2.6 and Pillow 12.3.0.

```bash
conda env create -f environment.yml
conda activate ai-attribution
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python tools/verify_assets.py
```

The CUDA wheel includes the needed runtime; no separate CUDA toolkit install
is required for this project. Use a compatible NVIDIA driver. For CPU-only
inference, replace `cu128` with `cpu` in the PyTorch installation URL and use
`--device cpu --workers 0`. CPU training is not practical for the full recipe.

## 2. Prepare the provided images

Use the competition's original **development archive containing the labeled
training images**, and the actual final release **`test_images.zip`**. The
earlier `test.zip` is not a substitute for the final release. Archive paths
below are examples; replace them with your downloaded files.

```bash
python -m ai_attribution.prepare \
  --train-archive /path/to/dev_phase.zip \
  --test-archive /path/to/test_images.zip --output data
```

A standalone `train.zip` is also accepted. Nested archives are supported.
Preparation checks every labeled image's decoded RGB hash against the fixed
manifest and requires all official final filenames. It produces:

```text
data/train/             # 15,000 labeled images, shared by train/holdout manifest
data/test/              # 1,500 final RGB images
checkpoints/final_model.pt
checkpoints/parent_110.pt
artifacts/final_test_predictions.json
artifacts/final_test_predictions.zip
```

Only the manifest's **12,000 training rows** receive gradient updates. The
other 3,000 are the fixed labeled holdout. Each class has 4,000/1,000 rows.
Hidden development/test images are never training examples.

## 3. Reproduce the accepted predictions

```bash
python -m ai_attribution.predict \
  --checkpoint checkpoints/final_model.pt --images data/test \
  --output outputs/final_test.zip --device cuda:0 --batch-size 32 \
  --verify-reverse --reference-json artifacts/final_test_predictions.json
python -m ai_attribution.validate outputs/final_test.zip
```

Expected: **1,500 predictions**, exact agreement with the accepted
filename-to-label mapping, and canonical prediction JSON SHA256:

```text
6f45f62afe5cce9d7a7a01abe6909e01a2c46bc693546e0438fc9ad6f4bbdaa1
```

Class mapping is `0 = Real`, `1 = Model A`, `2 = Model B`. The ZIP contains
only `predictions.json` at its root. Reverse-order, half-batch replay checks
stability; it is a verification pass, not prediction averaging. The generated
ZIP's container metadata differs from the historical ZIP, while its JSON
bytes and labels match. The exact original submitted ZIP is preserved in
`artifacts/`. No command uploads to CodaBench.

## 4. Evaluate labeled validation

```bash
python -m ai_attribution.evaluate --checkpoint checkpoints/final_model.pt \
  --condition clean --output outputs/clean_validation.json
python -m ai_attribution.evaluate --checkpoint checkpoints/final_model.pt \
  --condition all --output outputs/stress_validation.json
```

| Selected model | Recorded accuracy |
|---|---:|
| Clean labeled holdout | 99.300% |
| Center Lanczos / JPEG-90 | 99.633% |
| Center bilinear / JPEG-90 | 99.067% |
| Worst of 13 primary stress conditions | 98.033% |
| Final CodaBench test (different images) | **97.133%** |

Validation evaluates 27 conditions **separately on the same 3,000 images**.
It does not average their predictions. The 13 primary conditions determine
the final candidate ranking; all other conditions are diagnostics. See
[experimental analysis](docs/EXPERIMENTS.md).

## 5. Reproduce training

To replay the final ten epochs from the provided project-trained parent:

```bash
CUDA_VISIBLE_DEVICES=0 python -m ai_attribution.train \
  --config configs/06_final.json --parent checkpoints/parent_110.pt \
  --output runs/final_replay --device cuda:0
```

To replay the retained 120-epoch lineage from random initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/replay_curriculum.py --output runs/full_replay
```

To reproduce the paired final comparison on three available GPUs:

```bash
python tools/run_ablation.py --gpus 0 1 2 --parent checkpoints/parent_110.pt \
  --output runs/final_ablation
```

These are expensive training commands. A final branch took approximately
17 minutes on an RTX 6000 Ada; full replay includes 120 gradient epochs plus
validation and takes longer. Physical training batch size is **64**. Do not
replace it with gradient accumulation: BatchNorm would see different batches.
The selected noise stage is **15 epochs of a 20-epoch learning-rate schedule**.
Seed 7454 and fixed private augmentation streams are encoded in the configs.

Training replay preserves the documented recipe. A fresh complete 120-epoch
retrain was **not** performed during release preparation, and bitwise
equivalence across different CUDA/PyTorch hardware stacks is not promised.
Exact accepted-label reproduction is verified for the supplied final weights.
See [reproducibility details](docs/REPRODUCIBILITY.md).

## Checks and figures

```bash
python -m unittest discover -s tests -v
python -m pip install -e '.[dev,figures]'
python -m ruff check .
python -m ruff format --check .
python tools/make_figures.py --output outputs/figures
```

Release checks preserve all original learned tensors and buffers, reproduce
all 1,500 accepted predictions, compare augmentation views with the original
experiment code, and verify 27 validation renderers pixel for pixel. Receipts
are in `results/refactor_validation.json` and `results/reproduction_check.json`.

## References and attribution

- [Tan et al., CVPR 2024: Neighboring Pixel Relationships](https://arxiv.org/abs/2312.10461).
- [He et al., CVPR 2016: Deep Residual Learning](https://arxiv.org/abs/1512.03385).
- [Wang et al., CVPR 2020: CNN-Generated Images Are Surprisingly Easy to Spot](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html).
- [Loshchilov and Hutter, ICLR 2019: AdamW](https://arxiv.org/abs/1711.05101).

This is a course-project implementation inspired by these methods, not a
CVPR publication or a claim to invent NPR/ResNet. `torchvision` provides the
ResNet implementation with `weights=None`. Original project code is released
under MIT; dependencies retain their respective licenses. Dataset access and
redistribution remain governed by the competition provider.
