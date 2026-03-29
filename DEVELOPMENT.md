# Development

## Core Commands

```bash
pnpm run build
pnpm run build:review
pnpm run typecheck
pnpm run test
uv run milady check-pfp <avatar-url-or-local-file>
```

## Chrome Debugging

Launch Chrome with a persistent local debug profile:

```bash
pnpm run debug:chrome:launch-local-profile
```

Attach to that running Chrome session:

```bash
pnpm run debug:chrome:attach
```

Keep the CDP attachment open:

```bash
pnpm run debug:chrome:attach:keep-open
```

## Paths

- Runtime model artifacts live in `public/models/` and `public/generated/`.
- Training runs, labels, downloaded avatars, and dataset manifests live under ignored `cache/`.

## Training Pipeline

The extension exports collected avatars as JSON manifests. The offline pipeline ingests those exports into a local SQLite catalog under `cache/`, downloads avatar images, supports manual labeling, then trains and exports a MobileNetV3-Small classifier back into the extension runtime. The review app supports both individual labeling and 9-up batch labeling.

For a fuller narrative walkthrough of the pipeline and label sources used during review and training, see [docs/training-pipeline.md](docs/training-pipeline.md).

Split policy:
- blind `val` / `test` prioritize manual export labels and held-out collection positives
- routine training uses real exported avatars, conservative model labels, and a reduced-weight collection corpus
- `manual` labels are gold and `model` labels are trusted at lower weight

Typical loop:

```bash
uv run milady ingest
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-collections
uv run milady score --run-id <current-best-run-id>
pnpm run build:review
uv run milady review
uv run milady build-dataset
uv run milady train --run-id <candidate-run-id>
uv run milady score --run-id <run-id>
uv run milady compare --run-id <current-best-run-id> --run-id <candidate-run-id>
uv run milady export-errors --compare-dir <compare-dir>
uv run milady export-onnx --run-id <run-id>
pnpm run build
```

`uv run milady ingest` scans `cache/ingest/*.json` by default and archives those manifests into `cache/exports/raw/` as it ingests them. You can still pass explicit JSON paths when needed.

Recommended review order after scoring:
- `Hard negatives`
- `Model disagreements`
- `Boundary unlabeled`
- `Notifications`
- `High-impact`
- `Unreviewed`
- `Residual unlabeled`

In the review UI, pick a scored `run_id` first. Queue ranking, disagreement flags, and 9-up batch defaults are all tied to that selected run. Both individual and batch review write `manual` labels. `uv run milady score` refreshes automatic `model` labels from the same scored run unless you pass `--score-only`.

One subtlety: the review app only lists runs that have catalog scores in `model_scores`. Promoting a run into the extension does not make it appear in review by itself; run `uv run milady score --run-id <promoted-run-id>` first if you want to inspect that model in the review UI.
