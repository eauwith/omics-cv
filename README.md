# omics-cv

Leakage-aware cross-validation harnesses and omics autoencoders, extracted from
[`origin-predict`](https://github.com/eauwith/origin-predict) so the same
evaluation machinery can be shared across projects instead of reimplemented.

The problem it exists to solve: in omics studies the sampling unit is almost
never the row. Repeat samples from one subject, cells from one donor, or slides
from one block will leak across a naive `KFold` split and inflate performance.
These harnesses make the grouping explicit.

## Install

```bash
pip install -e .
```

## Use

```python
from omics_cv.cv import LOSOHarness, NestedCVHarness, MetricsEvaluator, CrossValidationRunner
from omics_cv.embeddings import OmicsAutoencoder, AutoencoderTrainer

# Leave-one-subject-out: no subject appears in both train and test
harness = LOSOHarness(group_column="animal_id")
for train_idx, test_idx in harness.split(X, y, groups):
    ...
```

| Component | Purpose |
| --- | --- |
| `StratifiedKFoldHarness` | class-balanced folds |
| `LOSOHarness` | leave-one-subject/group-out — use whenever rows repeat per subject |
| `NestedCVHarness` | inner loop for tuning, outer loop for honest error |
| `MetricsEvaluator` | AUROC, AUPRC, F1, Spearman/Pearson under one interface |
| `CrossValidationRunner` | drives a splitter + estimator + evaluator |
| `OmicsAutoencoder` / `VariationalOmicsAutoencoder` | dense and variational latent encoders |
| `AutoencoderTrainer` | training loop with denoising support |

## Provenance

History is preserved from `origin-predict` via `git filter-repo`; the original
path was `src/infrastructure/`.
