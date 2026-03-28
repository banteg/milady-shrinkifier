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

## Alternate Profile Helpers

Launch against a system Chrome profile named `Extension`:

```bash
pnpm run debug:chrome:launch-extension-profile
```

Legacy extension debug helpers:

```bash
pnpm run debug:extension
pnpm run debug:extension:default-profile
pnpm run debug:extension:extension-profile
pnpm run debug:extension:seed-extension-profile
```

## Paths

- Runtime model artifacts live in `public/models/` and `public/generated/`.
- Training runs, labels, downloaded avatars, and dataset manifests live under ignored `cache/`.

## Training Pipeline

The extension exports collected avatars as JSON manifests. The offline pipeline ingests those exports into a local SQLite catalog under `cache/`, downloads avatar images, supports manual labeling, then trains and exports a MobileNetV3-Small classifier back into the extension runtime. The review app supports both individual labeling and 9-up batch labeling.

Split policy:
- blind `val` / `test` only use manual export labels (`label_source=manual`)
- official Milady images, derivative samples, and heuristic-assisted labels are train-only
- heuristic-assisted labels stay in the training pool with a reduced sample weight

Typical loop:

```bash
uv run milady ingest
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-collections
uv run milady label-heuristic
pnpm run build:review
uv run milady review
uv run milady build-dataset
uv run milady train
uv run milady score --run-id <run-id>
uv run milady export-onnx --run-id <run-id>
pnpm run build
```

`uv run milady ingest` scans `cache/ingest/*.json` by default and archives those manifests into `cache/exports/raw/` as it ingests them. You can still pass explicit JSON paths when needed.
