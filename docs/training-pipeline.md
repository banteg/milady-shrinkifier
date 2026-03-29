# Training Pipeline Walkthrough

This page explains how Milady Shrinkifier improves its classifier over time, from raw browser exports to a new extension build.

The short version is:

1. the extension collects avatar sightings while you browse X,
2. those sightings are exported and ingested into a local catalog,
3. avatar images and NFT collection samples are downloaded,
4. the current model scores the catalog,
5. the review tool turns those scores into high-value labeling work,
6. the dataset builder turns everything into train/validation/test splits,
7. a new model is trained and compared against the current production model,
8. if it wins, it is exported back into the extension.

Everything happens locally. The browser extension, the catalog, the review app, training, evaluation, and ONNX export all run on your machine.

## Why The Pipeline Exists

The extension ships a compact image classifier that decides whether a profile picture looks Milady-like. That model needs two things to improve:

- more examples of real avatars seen on X
- better labels around the decision boundary

The pipeline is designed to get both without introducing a remote service or a cloud training system. The extension gathers the raw material. The offline tools organize it, rank it, label it, train on it, and ship a new model back into the extension.

## The Main Data Sources

The model learns from three broad sources:

1. **Exported avatars**
   These are real avatars collected by the extension while you browse. They are the most important source because they match the production environment.

2. **Collection corpora**
   These are curated positive NFT collections stored under `cache/collections/`, such as:
   - `milady-maker`
   - `remilio`
   - `pixelady`

   These help define the positive class, but they are cleaner than real exported avatars, so they are downweighted during training.

3. **Automatic weak labels**
   These are conservative silver labels generated from extreme-confidence model predictions. Today they are mostly used for obvious negatives.

## Step 1: Collect And Export Avatars From The Extension

While the extension runs on X, it scans visible avatars locally and keeps track of:

- avatar URL
- account handles
- display names
- source surfaces
- how often the avatar was seen
- example profile, tweet, and notification URLs
- whether the account was whitelisted

Nothing is uploaded anywhere. When you export from the popup, the extension writes a JSON manifest. Those manifests are usually dropped into `cache/ingest/`.

## Step 2: Ingest Exports Into The Local Catalog

Run:

```bash
uv run milady ingest
```

This command:

- scans `cache/ingest/*.json`
- archives those manifests into `cache/exports/raw/`
- merges their contents into the local SQLite catalog at `cache/dataset/avatar_catalog.sqlite`

Ingest deduplicates by normalized avatar URL. If the same avatar URL appears in multiple exports, the pipeline merges metadata instead of creating separate records. Handles, display names, source surfaces, and seen counts are combined into one catalog entry.

At this stage the catalog knows about avatar sightings, but it may not have the image bytes yet.

## Step 3: Download Avatar Images And Collection Positives

Run:

```bash
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-collections
```

These commands do two different jobs:

- `download-avatars` fetches the exported avatar images into `cache/avatars/files/`
- `download-collections` maintains the positive NFT corpora under `cache/collections/`

Downloaded avatars are deduplicated by image SHA. If many different avatar URLs point to the same underlying image, the catalog will still end up with one stored image object.

Collection downloads are tracked in one manifest:

- `cache/collections/manifest.json`

That manifest tells the training pipeline which positive collection samples currently exist on disk.

## Step 4: Score The Catalog With The Current Model

Run:

```bash
uv run milady score
```

If you do not pass `--run-id`, this uses the currently promoted production run automatically.

Scoring writes per-image model outputs into the catalog, including:

- the model run id
- the score
- the predicted label
- the threshold associated with that run
- distance from the threshold

This does not create labels by itself. It creates ranking information that the review tool can use to focus human attention where it matters most.

## Step 5: Review High-Value Cases

Build and open the review app:

```bash
pnpm run build:review
uv run milady review
```

The review app is the human-in-the-loop part of the system. It is no longer designed as a generic “label everything manually” interface. It is meant to help you spend time on the cases that most improve the model.

### Run-Pinned Review

The review UI starts with a `Run` selector. Pick the scored run you want to improve. All queue ranking, disagreement flags, and batch defaults are tied to that selected run.

### The Most Useful Queues

The main review queues are:

- **Uncertain unlabeled**
  Items closest to the current threshold. These are the best gold-label candidates because they shape the decision boundary.

- **High-score false positives**
  Items already labeled `not_milady` that the model still scores highly. These are excellent failure-harvesting examples.

- **Human vs model**
  Items where the current human label disagrees with the selected model run.

- **High-score unlabeled**
  Strong model-positive candidates that have not been reviewed yet.

- **Unlabeled**
  The general backlog. This is lower-yield than the score-driven queues once the model is reasonably good.

### Label Trust Levels

The review pipeline now distinguishes between different levels of trust:

- **`manual`**
  Written by individual review. These are treated as gold labels.

- **`model_reviewed`**
  Written by batch review. These are human-confirmed model suggestions and are treated as trusted labels rather than gold.

- **`silver`**
  Fully automatic weak labels created by the silver-label tool.

This split matters because not every human interaction should count as the same quality of training signal. Fast batch confirmation is valuable, but it is not the same as careful manual adjudication of a difficult case.

### Batch Review

Batch mode shows nine items at a time. It is model-proposal-first:

- each tile defaults to the model’s predicted label when available
- the human mainly confirms or corrects
- committing the batch writes `model_reviewed` labels

This is for throughput. It is useful when the model is already mostly right and you want humans spending less time on obvious cases.

### Individual Review

Individual review is for higher-value decisions:

- hard boundary cases
- ambiguous items
- failures
- anything you want treated as full gold signal

Those labels are written as `manual`.

## Step 6: Optional Silver Labels

The pipeline also supports conservative fully automatic labels:

```bash
uv run milady label-silver --run-id <run-id>
```

Today this is mainly used for extremely low-score negatives. Those labels are:

- train-only
- weakly weighted
- excluded from blind validation and test

Silver labels are meant to reduce low-value human work, not replace review of important cases.

## Step 7: Build The Dataset

Run:

```bash
uv run milady build-dataset
```

This is where the pipeline turns raw files and labels into training data.

### What The Builder Does

It:

- loads exported avatars and collection positives
- computes or reuses cached fingerprints
- groups exact and near-duplicate images together
- assigns stable train/validation/test splits
- writes JSONL files under `cache/dataset/splits/`

The builder also maintains:

- `cache/dataset/offline_cache.sqlite`
  for image fingerprints and preprocessing caches
- `cache/dataset/split_manifest.json`
  for stable split assignments and dataset metadata

### Deduplication And Grouping

The dataset builder groups images by:

- raw SHA
- normalized pixel digest
- perceptual hash proximity

That prevents obvious duplicates and near-duplicates from leaking across splits.

### Split Policy

The split policy is intentionally asymmetric:

- blind `val` and `test` prioritize manually labeled exported avatars
- held-out collection positives are included as a fixed extra positive slice
- routine training uses the much larger mix of exported labels, reviewed labels, silver labels, and collection positives

This keeps evaluation honest while still allowing broader training input.

## Step 8: Trust Tiers And Weights

The training set uses three trust tiers:

- **Gold**
  Full manual exported labels
- **Trusted**
  Human-confirmed batch labels and collection corpus samples
- **Weak**
  Silver labels

Current training weights are:

- `manual`: `1.0`
- `model_reviewed`: `0.7`
- `silver`: `0.35`
- collection corpus positives: `0.5`

The idea is simple:

- gold labels should dominate
- trusted labels should matter, but less
- weak labels should shape the boundary without steering it too strongly

## Step 9: Train A Candidate Model

Run:

```bash
uv run milady train --run-id <candidate-run-id>
```

Training reads the split JSONL files and fine-tunes the classifier on your local machine. Each run writes a directory under:

- `cache/models/mobilenet_v3_small/<run-id>/`

That run directory contains:

- checkpoints
- a `summary.json`
- validation and test metrics
- dataset split metadata used for the run

The trainer uses the same image preprocessing logic as the extension runtime so offline metrics match deployed behavior as closely as possible.

## Step 10: Compare Before Promotion

After training, rescore and compare:

```bash
uv run milady score --run-id <candidate-run-id>
uv run milady compare --run-id <current-prod-run-id> --run-id <candidate-run-id>
uv run milady export-errors --compare-dir <compare-dir>
```

This step is mandatory in practice, even if it is just a local habit and not a hard enforcement rule.

### Why Compare Matters

A run can look good in its own training summary but still be worse than the current production model when both are evaluated on the same set.

`compare` solves that by re-evaluating runs side by side on the same evaluation set.

### Error Export

`export-errors` turns false positives and false negatives into image folders so you can inspect what each model is missing or misfiring on.

That is one of the highest-value feedback loops in the system, because it tells you whether the next gains should come from:

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

## What “Good” Looks Like

At this stage of the project, the model is already good enough that the highest-value work is usually not “label more random avatars.”

The best improvements now usually come from:

- reviewing uncertain items
- reviewing high-score false positives
- harvesting missed positives from compare error folders
- keeping evaluation honest
- promoting only after direct comparison

That is why the pipeline has shifted from broad manual labeling toward model-guided review.

## Typical End-To-End Command Loop

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

The important conceptual split is:

- the extension gathers raw sightings
- the scorer ranks what matters
- the review app turns those rankings into better labels
- the dataset builder turns those labels into stable splits
- the trainer produces a candidate
- compare decides whether it is actually better
- export ships it back into the extension

That loop is the whole product.
