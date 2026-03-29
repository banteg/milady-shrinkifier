import { describe, expect, it } from "vitest";
import sharp from "sharp";

import {
  computeCoverCropRegion,
  computeNormalizedTensorFromRgbBuffer,
  resolveRuntimeModelConfig,
} from "../src/shared/model-config";
import { computeNodeImageFeatures } from "../src/shared/node-image";
import type { ModelMetadata } from "../src/shared/types";

describe("runtime preprocessing", () => {
  it("resolves runtime model config from metadata with defaults", () => {
    const metadata: ModelMetadata = {
      generated_at: "2026-03-29T00:00:00Z",
      threshold: 0.5,
    };
    expect(resolveRuntimeModelConfig(metadata)).toEqual({
      inputSize: 128,
      channels: 3,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      positiveIndex: 1,
    });
  });

  it("computes distinct top and center cover regions", () => {
    expect(computeCoverCropRegion(4, 6, 4, 4, "top")).toEqual({
      left: 0,
      top: 0,
      width: 4,
      height: 4,
    });
    expect(computeCoverCropRegion(4, 6, 4, 4, "center")).toEqual({
      left: 0,
      top: 1,
      width: 4,
      height: 4,
    });
  });

  it("normalizes rgb buffers using config statistics", () => {
    const tensor = computeNormalizedTensorFromRgbBuffer(
      new Uint8Array([255, 128, 0]),
      {
        inputSize: 1,
        channels: 3,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.25, 0.5],
        positiveIndex: 1,
      },
    );
    expect(Array.from(tensor)).toEqual([1, expect.closeTo(0.0078431377, 6), -1]);
  });

  it("uses metadata-driven input size in node preprocessing", async () => {
    const image = await sharp({
      create: {
        width: 8,
        height: 12,
        channels: 3,
        background: { r: 255, g: 0, b: 0 },
      },
    }).png().toBuffer();

    const features = await computeNodeImageFeatures(
      image,
      resolveRuntimeModelConfig({
        generated_at: "2026-03-29T00:00:00Z",
        threshold: 0.5,
        input_size: 32,
        channels: 3,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      }),
      "center",
    );

    expect(features.modelShape).toEqual([1, 3, 32, 32]);
    expect(features.modelTensor).toHaveLength(3 * 32 * 32);
  });
});
