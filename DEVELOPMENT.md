# Development

## Core Commands

```bash
pnpm run build
pnpm run typecheck
pnpm run test
uv run python scripts/check_pfp_url.py <avatar-url>
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

## Asset Preparation

Download the Milady image corpus:

```bash
pnpm run download:images
pnpm run download:images:aria2
```

Generate the legacy local model assets:

```bash
uv run python scripts/build_prototype_model.py
```

## Training Pipeline

The extension exports collected avatars as JSON manifests. The offline pipeline ingests those exports into a local SQLite catalog under `cache/`, downloads avatar images, supports manual labeling, then trains and exports a MobileNetV3-Small classifier back into the extension runtime.

Typical loop:

```bash
uv run python scripts/ingest_avatar_exports.py cache/milady-shrinkifier-avatars-<timestamp>.json
uv run python scripts/download_avatar_catalog.py
uv run python scripts/download_avatar_catalog.py --retry-failed
uv run python scripts/label_heuristic_matches.py
uv run python scripts/review_avatars.py
uv run python scripts/build_training_dataset.py
uv run python scripts/train_classifier.py
uv run python scripts/score_avatar_catalog.py --run-id <run-id>
uv run python scripts/export_classifier_onnx.py --run-id <run-id>
pnpm run build
```
