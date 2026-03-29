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

All scores below come from the current exported evaluation corpus: deduped exported avatars labeled `manual` or `model`.

- **Precision** — when the extension filters a post, how often it's right.
- **Recall** — of the Milady-style avatars in the evaluation set, how many it catches.
- **Evaluation corpus** — `8,496` exported avatars (`390` milady, `8,106` not_milady).
- This is a broader product-facing snapshot than the blind split, but it is not a blind benchmark.

| Version | Run | Training mix | Precision | Recall |
| --- | --- | --- | --- | --- |
| `v0.2.2` | `20260327T142224Z` | Milady Maker + `2,596` manually tagged avatars | `0.9957` | `0.5872` |
| `v0.3.0` | `20260327T212453Z` | + Remilio, Pixelady + `2,967` manually tagged avatars | `0.9964` | `0.7051` |
| `v0.4.0` | `20260328T144735Z` | + `5,715` manually tagged avatars | `0.9965` | `0.7205` |
| `v0.5.0` | `20260328T223931Z` | + `6,773` manually tagged avatars | `1.0000` | `0.9077` |
| `v0.6.0` | `20260329T124912Z` | + `7,370` human-reviewed avatars | `0.9971` | `0.8769` |

All rows were re-evaluated on the same exported evaluation corpus on March 29, 2026, so they are directly comparable.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for build commands and debugging.

For a user-facing walkthrough of how the offline training loop works end to end, see [docs/training-pipeline.md](docs/training-pipeline.md).
