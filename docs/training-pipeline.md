# Training Pipeline Walkthrough

This page explains the training pipeline from extension export to promoted model.

## Overview

The pipeline is:

1. collect avatar sightings in the extension
2. export them as JSON
3. ingest those exports into the local catalog
4. download avatar images and positive collection samples
5. score the catalog with the current model
6. review the highest-value cases
7. build stable train/validation/test splits
8. train a candidate model
9. compare it against the current production model
10. export it back into the extension if it wins

## Data Sources

The model trains on two main sources:

1. **Exported avatars**
   Real avatars collected from X. This is the highest-value data because it matches the production environment.

2. **Collection corpora**
   Positive NFT collections stored under `cache/collections/`, currently:
   - `milady-maker`
   - `remilio`
   - `pixelady`

   These are useful positives, but they are cleaner than real exported avatars, so they are downweighted during training.

## Step 1: Export From The Extension

The extension collects avatar sightings while you browse X. An export contains:

- avatar URL
- handles
- display names
- source surfaces
- seen count
- example profile / tweet / notification URLs
- whitelist state

Exports are usually dropped into `cache/ingest/`.

## Step 2: Ingest Into The Catalog

Run:

```bash
uv run milady ingest
```

This command:

- scans `cache/ingest/*.json`
- archives those manifests into `cache/exports/raw/`
- merges the exported rows into `cache/dataset/avatar_catalog.sqlite`

Ingest deduplicates by normalized avatar URL. If the same avatar URL appears in multiple exports, the pipeline merges metadata into one catalog row instead of creating duplicates.

At this point the catalog has sighting metadata, but some rows may still be missing downloaded image bytes.

## Step 3: Download Images

Run:

```bash
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-collections
```

These commands do two things:

- `download-avatars` fetches exported avatar images into `cache/avatars/files/`
- `download-collections` maintains the positive NFT corpora under `cache/collections/`

Downloaded avatars are deduplicated by image SHA. If many avatar URLs point to the same image, the catalog still ends up with one stored image object.

Collection downloads are tracked in:

- `cache/collections/manifest.json`

## Step 4: Score The Catalog

Run:

```bash
uv run milady score
```

If you omit `--run-id`, this uses the currently promoted production run.

Scoring writes per-image model outputs into the catalog:

- model run id
- score
- predicted label
- threshold for that run
- distance from the threshold

This step writes both score records and automatic `model` labels for non-manual exported avatars, unless you pass `--score-only`. It also ranks the catalog so review effort can focus on the most useful cases.

## Step 5: Review High-Value Cases

Build and open the review app:

```bash
pnpm run build:review
uv run milady review
```

The review app is the labeling and audit layer. It is meant to direct human attention to the cases that most improve the model.

### Run-Pinned Review

The UI starts with a `Run` selector. Pick the scored run you want to improve. Queue ranking, disagreement flags, and batch defaults are all tied to that selected run.

The `Run` selector is populated from scored runs in the local catalog. A run that has been promoted into the extension still will not appear there until you score the catalog with it at least once.

### Main Queues

The most useful queues are:

- **Boundary unlabeled**
  Items closest to the current threshold. These are the best gold-label candidates because they shape the decision boundary.

- **Hard negatives**
  Items already labeled `not_milady` that the model still scores highly. These are strong failure examples.

- **Model disagreements**
  Items where the human label disagrees with the selected run.

- **Residual unlabeled**
  The highest-scoring items left in the unlabeled pool. As the model improves, this often becomes a low-priority backlog rather than a stream of likely positives.

- **Unreviewed**
  The general backlog. This is lower-yield than the score-driven queues.

### Label Sources

The pipeline now uses three simple states for exported avatars:

- **`manual`**
  Any human-reviewed label. Individual and batch review both write `manual`, and these are treated as gold labels.

- **`model`**
  Automatic labels refreshed by `uv run milady score`. These are trainable and currently use the same training weight as `manual` labels.

- **`unclear`**
  A review outcome for ambiguous items. These stay in the catalog but are excluded from training and evaluation.

### Batch Review

Batch mode shows nine items at a time and defaults each tile to the model’s predicted label when available. The normal action is confirm or slightly correct, not label from scratch.

Committing a batch writes `manual` labels.

### Individual Review

Individual review is for:

- hard boundary cases
- ambiguous items
- failures
- anything you want treated as full gold signal

Those labels are also written as `manual`.

### Model Labels

After scoring, the pipeline refreshes automatic labels from the same run. `uv run milady score` clears previous `model` labels and rewrites the current run’s non-manual exported backlog. This keeps the auto-label lane tied to the latest model instead of accumulating stale pseudo-labels over time.

If you want to write only `model_scores` without touching automatic labels, use:

```bash
uv run milady score --score-only
```

## Step 6: Build The Dataset

Run:

```bash
uv run milady build-dataset
```

This materializes the training dataset.

The builder:

- loads exported avatars and collection positives
- computes or reuses cached fingerprints
- groups exact and near-duplicate images together
- assigns stable train/validation/test splits
- writes JSONL files under `cache/dataset/splits/`

The builder also maintains:

- `cache/dataset/offline_cache.sqlite`
  image fingerprints and preprocessing caches
- `cache/dataset/split_manifest.json`
  split assignments and dataset metadata

### Deduplication

The dataset builder groups images by:

- raw SHA
- normalized pixel digest
- perceptual hash proximity

This prevents obvious duplicates and near-duplicates from leaking across splits.

### Split Policy

The split policy is intentionally asymmetric:

- blind `val` and `test` prioritize manually labeled exported avatars
- held-out collection positives are included as a fixed extra positive slice
- routine training uses the larger mix of exported labels, model labels, and collection positives

## Step 7: Trust Tiers And Weights

The training set uses two trust tiers for exported labels:

- **Gold**
  all human-reviewed exported labels
- **Trusted**
  automatic `model` labels and collection corpus samples

Current weights are:

- `manual`: `1.0`
- `model`: `1.0`
- collection corpus positives: `0.5`

The intended effect is:

- human-reviewed and model-refreshed exported labels train at full strength
- collection positives still help, but they do not dominate the exported corpus

## Step 8: Train A Candidate

Run:

```bash
uv run milady train --run-id <candidate-run-id>
```

Training reads the split JSONL files and writes a run directory under:

- `cache/models/mobilenet_v3_small/<run-id>/`

That run directory contains:

- checkpoints
- `summary.json`
- validation and test metrics
- dataset split metadata for the run

The trainer uses the same image preprocessing logic as the extension runtime so offline metrics match deployed behavior.

## Step 9: Compare Before Promotion

After training, rescore and compare:

```bash
uv run milady score --run-id <candidate-run-id>
uv run milady eval --run-id <current-prod-run-id> --run-id <candidate-run-id>
uv run milady export-errors --compare-dir <compare-dir>
```

This is the decision point before promotion.

`eval` re-evaluates runs side by side on the same evaluation set. `export-errors` turns false positives and false negatives into image folders so you can inspect what changed.

This is where you find out whether the next improvement should come from:

- more hard negatives
- more edited positives
- threshold changes
- data weighting changes

## Step 10: Promote The Winner

If the candidate is better, export it back into the extension:

```bash
uv run milady export-onnx --run-id <candidate-run-id>
pnpm run build
```

This updates:

- `public/models/milady-mobilenetv3-small.onnx`
- `public/generated/milady-mobilenetv3-small.meta.json`

The extension then uses those artifacts at runtime.

## Typical Command Loop

```bash
uv run milady ingest
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-collections
uv run milady score
pnpm run build:review
uv run milady review
uv run milady build-dataset
uv run milady train --run-id <candidate-run-id>
uv run milady score --run-id <candidate-run-id>
uv run milady eval --run-id <current-prod-run-id> --run-id <candidate-run-id>
uv run milady export-errors --compare-dir <compare-dir>
uv run milady export-onnx --run-id <candidate-run-id>
pnpm run build
```

## Summary

The training pipeline is:

- collect and export avatar sightings
- ingest them into the catalog
- download avatars and positive collections
- score the catalog with the current model
- review high-value cases
- build stable splits
- train a candidate
- compare it against the current production model
- export it back into the extension if it wins
