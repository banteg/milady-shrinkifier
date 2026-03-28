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

Headline metrics are measured on the current blind test set of manually labeled exported avatars only. Official corpus images, derivative collections, and heuristic labels are used for training, but not for the main score.

| Promoted run | Train / val / test | Training labels | Precision | Recall | False positives | False negatives |
| --- | --- | --- | --- | --- | --- | --- |
| `20260327T142224Z` | `10,781 / 1,346 / 1,350` | `8,185` milady, `2,596` not_milady | `0.8333` | `0.6250` | `2` | `6` |
| `20260327T212453Z` | `13,593 / 1,698 / 1,701` | `10,626` milady, `2,967` not_milady | `1.0000` | `0.6875` | `0` | `5` |
| `20260328T144735Z` | `17,878 / 600 / 386` | `13,322` milady, `4,556` not_milady | `1.0000` | `0.8125` | `0` | `3` |

These numbers were re-evaluated on the same current blind set on March 28, 2026, so they are directly comparable across production model revisions.

The blind evaluation sets are manually labeled exported avatars only:

- validation: `600` images (`27` positives, `573` negatives)
- test: `386` images (`16` positives, `370` negatives)

## Notes

- Development, debugging, and training workflow commands live in [DEVELOPMENT.md](DEVELOPMENT.md).
- Runtime model artifacts live in [`public/models/`](public/models/) and [`public/generated/`](public/generated/).
- Training runs, labels, downloaded avatars, and dataset manifests live under ignored [`cache/`](cache/).
- The review app supports both individual labeling and 9-up batch labeling.
- The extension runtime is ONNX-only.
