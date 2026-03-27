# Development

## Core Commands

```bash
pnpm run build
pnpm run build:review
pnpm run typecheck
pnpm run test
uv run milady check-pfp <avatar-url>
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

## Training Pipeline

The extension exports collected avatars as JSON manifests. The offline pipeline ingests those exports into a local SQLite catalog under `cache/`, downloads avatar images, supports manual labeling, then trains and exports a MobileNetV3-Small classifier back into the extension runtime.

Typical loop:

```bash
uv run milady ingest-avatars cache/milady-shrinkifier-avatars-<timestamp>.json
uv run milady download-avatars
uv run milady download-avatars --retry-failed
uv run milady download-derivatives
uv run milady label-heuristic
pnpm run build:review
uv run milady review
uv run milady build-dataset
uv run milady train
uv run milady score --run-id <run-id>
uv run milady export-onnx --run-id <run-id>
pnpm run build
```
