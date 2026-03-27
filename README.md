# Milady Shrinkifier

*Protecting your timeline from the egregore since 2026.*

## Why

Some people find that a significant percentage of their timeline consists of accounts using aesthetically identical chibi avatars posting aesthetically identical content. This extension addresses that.

## What It Does

The extension scans author avatars locally with a bundled ONNX classifier. When a match is found, you choose what happens:

- `Hide` collapses matched posts behind a click-to-reveal row.
- `Fade` renders matched posts at 50% opacity.
- `Debug` shows visual markers and detector score badges.
- `Off` does nothing.

Everything runs locally. No server calls, no telemetry, no data leaves your browser unless you explicitly export collected avatar data yourself.

## Other Features

- `Stats` shows live session counts for seen posts, matched posts, match rate, exemptions, errors, and last match time.
- `Accounts` keeps a running list of caught handles, grouped into `Exempt` and `Caught`.
- `Data` collects normalized avatar URLs and metadata for offline dataset building.
- The `Export` action dumps collected avatars as JSON for labeling and model improvement.

## Install

There is no Chrome Web Store release yet.

To run it locally:

1. Install JavaScript dependencies:
   ```bash
   pnpm install
   ```
2. Install Python dependencies for the classifier and labeling pipeline:
   ```bash
   uv sync
   ```
3. Build the extension:
   ```bash
   pnpm run build
   ```
4. Load `dist/` as an unpacked extension in Chrome.

## Privacy

All detection happens on-device using a bundled ONNX model. No images are uploaded anywhere. Collected data is stored in local browser storage and is never transmitted unless you explicitly export it.

## Notes

- Development, debugging, and training workflow commands live in `DEVELOPMENT.md`.
- Runtime model artifacts live in `public/models/` and `public/generated/`.
- Training runs, labels, downloaded avatars, and dataset manifests live under ignored `cache/`.
- The review app supports both individual labeling and 9-up batch labeling.
- The extension runtime is ONNX-only.
