# Experimental record and interpretation

## Final endpoint comparison

| Post-noise endpoint mass | Worst primary accuracy | Lowest primary class recall | Passes gate |
|---|---:|---:|---|
| 0.25 | 97.9667% | 97.5% | No |
| 0.50 | 98.0000% | 97.4% | Yes |
| **0.75** | **98.0333%** | 97.4% | Yes |

These candidates share the same parent, ten-epoch training budget and paired
random streams. The worst-condition advantage of 0.75 over 0.25 is only two
images out of 3,000. It is not evidence of a statistically decisive general
advantage. The 0.25 branch actually has slightly better minimum class recall.

`results/validation.json` contains all 27 condition measurements for each
candidate, including confusion matrices and error filenames. The primary
thirteen are seven JPEG-90 crop/kernel views plus pre/post Gaussian noise at
sigma 1, 2 and 4. Clean input and extreme noise are diagnostics. All conditions
reuse the same 3,000 images: they are not 81,000 independent samples.

## Selected-model sensitivity

| Condition | Accuracy | Real recall | Model A recall | Model B recall |
|---|---:|---:|---:|---:|
| Clean | 99.300% | 98.9% | 99.3% | 99.7% |
| Bilinear / JPEG-90 | 99.067% | 97.6% | 99.8% | 99.8% |
| Pre-noise sigma 4 | 98.267% | 98.9% | 97.4% | 98.5% |
| Post-noise sigma 4 | 98.033% | 97.6% | 97.5% | 99.0% |
| Pre-noise sigma 8 | 79.367% | 99.7% | 65.4% | 73.0% |

Severe pre-processing noise mainly changes synthetic images into predicted
real images. This is a measured limitation outside the training sigma range.
The recorded clean and post-sigma4 error sets have 12 filenames in common:
47 new errors and nine corrected clean errors, rather than a nested error set.

`results/training_history.json` contains the original ten-epoch histories of
the final three branches. Curves are training loss/accuracy measured online
under augmentation. Validation was performed only at the final endpoint;
there is no per-epoch validation curve to plot. Loss includes label smoothing.

## Development-to-test transfer

The earlier compact detector scored 99.556% on the development leaderboard,
but its first final-test score was approximately 33.46%, as reported by the
author. The recovery changed architecture, training duration and processing
curriculum. This is not a matched ablation isolating augmentation alone, and
the hidden test processing pipeline was not established from labeled evidence.

The accepted final result is 0.97133, rank 30, confirmed by the author. With
1,500 images, its rounded accuracy implies 1,457 correct and 43 errors. There
are no final-test labels, per-class test metrics or test confusion matrix.
Repeated selection on one local holdout can be optimistic; its 99.3% clean
accuracy exceeds final test accuracy by approximately 2.167 percentage points.

No paper figure or recorded table substitutes local stress accuracy for the
final server score. Raw training data and final hidden labels are not included.
