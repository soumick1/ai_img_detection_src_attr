# Reproducibility protocol

## What has actually been verified

The final clean checkpoint was loaded in evaluation mode and run on all
1,500 original final images. Its filename-keyed labels matched the accepted
historical submission exactly. A second pass in reverse order with a smaller
batch agreed. No model parameter or buffer changed. Every learned tensor and
buffer is bitwise equal to the original checkpoint before repackaging.

Release validation also compared normalization, 432 training views (72 per
stage), every sample key in three full 12,000-row sampler epochs, and all 27
validation renderers against the frozen original experiment sources. Training
view comparisons use a synthetic seeded JPEG, not a new accuracy measurement.
CPU tests exercise an actual optimizer update, tail-batch accounting, sampler
quotas, descriptor math, stage scheduling and official prediction rules.

A fresh full 120-epoch training replay was not run to prepare this release.
Recorded accuracy belongs to the released trained weights. Replaying the
recipe can yield small numerical differences on another runtime/GPU. Exact
intermediate optimizer/scaler resume is not supported: stages deliberately
create a fresh AdamW optimizer, and model checkpoints are inference/state
artifacts, not interruption-resume snapshots.

## Fixed data and stage schedule

Manifest SHA256:
`99d8b47c5ec2f26e9d962ec869804792f53f827f1ae91fda1975d2e33adcb18a`.
Do not regenerate the split for the recorded experiments. Exact decoded RGB
duplicates are grouped into one split. Preparation verifies 15,000 labeled
pixel hashes. Training is restricted to 12,000 rows; validation uses 3,000.
No development or final-test images receive updates or normalization fitting.

| Config | Updates in epochs | LR schedule length | Parent epochs | Augmentation epochs |
|---|---:|---:|---:|---|
| `01_scratch.json` | 60 | 60 | 0 | 0–59 |
| `02_processing.json` | 20 | 20 | 60 | 0–19 |
| `03_noise.json` | 15 | **20** | 80 | 0–14 |
| `04_bilinear.json` | 5 | 5 | 95 | 0–4 |
| `05_strong_noise.json` | 10 | 10 | 100 | 0–9 |
| `06_final.json` | 10 | 10 | 110 | **10–19** |

The historical noise run lasted 20 epochs, but epoch 15 was retained. The
clean replay stops after 15 while computing LR against its original
20-epoch horizon. Omitting this distinction changes subsequent training.

Common settings: seed 7454, physical batch 64, 188 updates/epoch (tail 32),
AdamW weight decay 0.001, label smoothing 0.05, norm clip 5, AMP, deterministic
cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each continuation resets AdamW.
The final model contains 20 BatchNorm layers with counters 22,560. The
supplied parent contains counters 20,680. Workers use seeded initialization;
each augmentation's geometry and noise are controlled by private item seeds.

## Final branch and comparison

Final configs differ only in name and post-noise endpoint mass 0.25/0.50/0.75.
All start from the same 110-epoch parent and run ten epochs. Every physical
batch has half noisy rows, one quarter forced clean bilinear rows and one
quarter remaining base-processing rows; noisy and forced rows are disjoint.
The tail preserves the same ratios. Noise order is half pre, half post.
Pre-noise sigma is endpoint 4 with probability 0.25, otherwise uniform 0–4.
Post-noise uses the selected endpoint probability, otherwise uniform 0–4.

`run_ablation.py` starts one independent process per GPU, waits for all three
to complete, and ranks their labeled validation results. Its selection does
not inspect hidden test predictions. The all-conditions evaluation is run
once after the final serialized state is reloaded; no best-epoch search is
performed inside these ten-epoch branches.

Eligibility: at least 98% accuracy and 97% recall per class in all 13 primary
conditions. Ranking: maximum minimum correct count, maximum total correct,
minimum summed losses, then fixed endpoint order 0.25, 0.50, 0.75. The selected
recorded branch is 0.75. Keep the three physical models separate; no ensemble
is created.

## Checkpoints, prediction hashes and portability

`assets/checkpoints.json` identifies clean release artifacts and original
file hashes. New clean checkpoint containers remove old experiment-wrapper
dependencies; all tensors and buffers are unchanged. New file hashes are
therefore expected and should not be confused with changed weights.

The accepted JSON has SHA256
`6f45f62afe5cce9d7a7a01abe6909e01a2c46bc693546e0438fc9ad6f4bbdaa1`.
It is serialized as an indented filename-to-integer mapping, official
filename order, followed by a newline. The historical ZIP hash is
`526661b24f6a88ffb197fc37e82e240a41f5a972f810deaa387dc65b7be6724f`.
Fresh ZIPs use fixed timestamps and can differ in compressed-container bytes
while retaining identical JSON. Validation checks coverage and integer labels,
not an assumed accuracy. The exact historical ZIP is included separately.

Run from the repository root after an editable install. Use `--workers 0`
when debugging multiprocessing, especially on Windows. A GPU with enough
memory for the physical training batch is needed; recorded training used
RTX 6000 Ada cards with 48GB-class memory. Three GPUs are useful for independent
final ablations; one GPU is sufficient for sequential curriculum training.
