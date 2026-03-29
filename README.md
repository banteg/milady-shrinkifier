# Milady Shrinkifier

*protecting your timeline from the egregore since 2026*

![hero](assets/hero.png)

## Why

Some people find that a significant percentage of their timeline consists of accounts using aesthetically identical chibi avatars posting aesthetically identical content. This extension addresses that.

## How It Works

A bundled ONNX classifier scans avatars as you scroll. When it spots a match, you pick what happens:

- **Hide** — collapsed behind a click-to-reveal row.
- **Fade** — visible but at half opacity.
- **Debug** — borders and confidence scores on every post.
- **Off** — does nothing.

The popup tracks session stats (posts scanned, match rate, last sighting), keeps a list of detected accounts you can exempt individually, and collects avatar data you can export for offline labeling.

Everything runs locally. No server calls, no telemetry, nothing leaves your browser unless you explicitly export it.

## Install

There is no Chrome Web Store release. Install from [GitHub Releases](https://github.com/banteg/milady-shrinkifier/releases) instead:

1. Download the latest `milady-shrinkifier-vX.Y.Z-unpacked.zip` from the [Releases page](https://github.com/banteg/milady-shrinkifier/releases).
2. Unzip it somewhere permanent on disk.
3. Open [`chrome://extensions`](chrome://extensions).
4. Enable `Developer mode`.
5. Click `Load unpacked`.
6. Select the unzipped folder.

## Accuracy

All scores come from the current blind set: manually labeled exported avatars plus a fixed held-out slice of collection positives.

- **Precision** — when the extension filters a post, how often it's right.
- **Recall** — of the Milady-style avatars in the test set, how many it catches.
- **Validation** set — used during training to choose thresholds and checkpoints; `778` images (`95` milady, `683` not_milady).
- **Test** set — held back for the final blind score; `445` images (`78` milady, `367` not_milady).

| Version | Run | Training mix | Precision | Recall |
| --- | --- | --- | --- | --- |
| `v0.2.2` | `20260327T142224Z` | Milady Maker + `2,596` manually tagged avatars | `0.9706` | `0.8462` |
| `v0.3.0` | `20260327T212453Z` | + Remilio, Pixelady + `2,967` manually tagged avatars | `1.0000` | `0.9359` |
| `v0.4.0` | `20260328T144735Z` | + `5,715` manually tagged avatars | `1.0000` | `0.9744` |
| `v0.5.0` | `20260328T223931Z` | + `6,773` manually tagged avatars | `1.0000` | `0.9744` |
| `v0.6.0` | `20260329T124912Z` | + `7,370` human-reviewed avatars | `1.0000` | `0.9231` |

All rows were re-evaluated on the same blind set on March 29, 2026, so they are directly comparable.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for build commands and debugging.

For a user-facing walkthrough of how the offline training loop works end to end, see [docs/training-pipeline.md](docs/training-pipeline.md).
