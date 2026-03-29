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

The model trains on three main sources:

1. **Exported avatars**
   Real avatars collected from X. This is the highest-value data because it matches the production environment.

2. **Collection corpora**
   Positive NFT collections stored under `cache/collections/`, currently:
   - `milady-maker`
   - `remilio`
   - `pixelady`

   These are useful positives, but they are cleaner than real exported avatars, so they are downweighted during training.

3. **Silver labels**
   Weak labels generated automatically from extreme-confidence model predictions. These are mainly used for obvious negatives.

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

This step does not create labels by itself. It ranks the catalog so review effort can focus on the most useful cases.

## Step 5: Review High-Value Cases

Build and open the review app:

```bash
pnpm run build:review
uv run milady review
```

The review app is the labeling and audit layer. It is meant to direct human attention to the cases that most improve the model.

### Run-Pinned Review

The UI starts with a `Run` selector. Pick the scored run you want to improve. Queue ranking, disagreement flags, and batch defaults are all tied to that selected run.

### Main Queues

The most useful queues are:

- **Uncertain unlabeled**
  Items closest to the current threshold. These are the best gold-label candidates because they shape the decision boundary.

- **High-score false positives**
  Items already labeled `not_milady` that the model still scores highly. These are strong failure examples.

- **Human vs model**
  Items where the human label disagrees with the selected run.

- **High-score unlabeled**
  Strong model-positive candidates that have not been reviewed yet.

- **Unlabeled**
  The general backlog. This is lower-yield than the score-driven queues.

### Review Trust Levels

The review pipeline distinguishes between three kinds of exported labels:

- **`manual`**
  Written by individual review. These are treated as gold labels.

- **`model_reviewed`**
  Written by batch review. These are human-confirmed model suggestions and are treated as trusted labels.

- **`silver`**
  Fully automatic weak labels.

Fast batch confirmation is useful, but it should not count as the same quality of signal as careful manual adjudication.

### Batch Review

Batch mode shows nine items at a time and defaults each tile to the model’s predicted label when available. The normal action is confirm or slightly correct, not label from scratch.

Committing a batch writes `model_reviewed` labels.

### Individual Review

Individual review is for:

- hard boundary cases
- ambiguous items
- failures
- anything you want treated as full gold signal

Those labels are written as `manual`.

## Step 6: Optional Silver Labels

Run:

```bash
uv run milady label-silver --run-id <run-id>
```

Today this is mainly used for extremely low-score negatives. Silver labels are:

- train-only
- weakly weighted
- excluded from blind validation and test

## Step 7: Build The Dataset

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
- routine training uses the larger mix of exported labels, reviewed labels, silver labels, and collection positives

## Step 8: Trust Tiers And Weights

The training set uses three trust tiers:

- **Gold**
  full manual exported labels
- **Trusted**
  human-confirmed batch labels and collection corpus samples
- **Weak**
  silver labels

Current weights are:

- `manual`: `1.0`
- `model_reviewed`: `0.7`
- `silver`: `0.35`
- collection corpus positives: `0.5`

The intended effect is:

- gold labels dominate
- trusted labels matter, but less
- weak labels shape the boundary without steering it too strongly

## Step 9: Train A Candidate

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

## Step 10: Compare Before Promotion

After training, rescore and compare:

```bash
uv run milady score --run-id <candidate-run-id>
uv run milady compare --run-id <current-prod-run-id> --run-id <candidate-run-id>
uv run milady export-errors --compare-dir <compare-dir>
```

This is the decision point before promotion.

`compare` re-evaluates runs side by side on the same evaluation set. `export-errors` turns false positives and false negatives into image folders so you can inspect what changed.

This is where you find out whether the next improvement should come from:

- more hard negatives
- more edited positives
- threshold changes
- data weighting changes

## Step 11: Promote The Winner

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
uv run milady compare --run-id <current-prod-run-id> --run-id <candidate-run-id>
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
