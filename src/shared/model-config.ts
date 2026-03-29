import type { ModelMetadata } from "./types";

export type CropVariant = "center" | "top";

export interface RuntimeModelConfig {
  inputSize: number;
  channels: number;
  mean: [number, number, number];
  std: [number, number, number];
  positiveIndex: number;
}

export interface CoverCropRegion {
  left: number;
  top: number;
  width: number;
  height: number;
}

const DEFAULT_RUNTIME_MODEL_CONFIG: RuntimeModelConfig = {
  inputSize: 128,
  channels: 3,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  positiveIndex: 1,
};

export function resolveRuntimeModelConfig(metadata: ModelMetadata): RuntimeModelConfig {
  const inputSize = Number.isInteger(metadata.input_size) && metadata.input_size! > 0
    ? metadata.input_size!
    : DEFAULT_RUNTIME_MODEL_CONFIG.inputSize;
  const channels = Number.isInteger(metadata.channels) && metadata.channels! > 0
    ? metadata.channels!
    : DEFAULT_RUNTIME_MODEL_CONFIG.channels;
  const mean = isTriple(metadata.mean) ? metadata.mean : DEFAULT_RUNTIME_MODEL_CONFIG.mean;
  const std = isTriple(metadata.std) ? metadata.std : DEFAULT_RUNTIME_MODEL_CONFIG.std;
  const positiveIndex = Number.isInteger(metadata.positive_index) && metadata.positive_index! >= 0
    ? metadata.positive_index!
    : DEFAULT_RUNTIME_MODEL_CONFIG.positiveIndex;
  return {
    inputSize,
    channels,
    mean,
    std,
    positiveIndex,
  };
}

export function runtimeModelShape(config: RuntimeModelConfig): [1, number, number, number] {
  return [1, config.channels, config.inputSize, config.inputSize];
}

export function computeCoverCropRegion(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  variant: CropVariant,
): CoverCropRegion {
  const targetAspect = targetWidth / targetHeight;
  const sourceAspect = sourceWidth / sourceHeight;

  if (sourceAspect > targetAspect) {
    const width = Math.max(1, Math.min(sourceWidth, Math.round(sourceHeight * targetAspect)));
    const left = Math.max(0, Math.round((sourceWidth - width) / 2));
    return {
      left,
      top: 0,
      width,
      height: sourceHeight,
    };
  }

  const height = Math.max(1, Math.min(sourceHeight, Math.round(sourceWidth / targetAspect)));
  const top = variant === "top"
    ? 0
    : Math.max(0, Math.round((sourceHeight - height) / 2));
  return {
    left: 0,
    top,
    width: sourceWidth,
    height,
  };
}

export function computeNormalizedTensorFromRgbBuffer(
  buffer: ArrayLike<number>,
  config: RuntimeModelConfig,
): Float32Array {
  const pixelCount = config.inputSize * config.inputSize;
  const tensor = new Float32Array(config.channels * pixelCount);

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const offset = pixelIndex * 3;
    tensor[pixelIndex] = (buffer[offset] / 255 - config.mean[0]) / config.std[0];
    tensor[pixelCount + pixelIndex] = (buffer[offset + 1] / 255 - config.mean[1]) / config.std[1];
    tensor[pixelCount * 2 + pixelIndex] = (buffer[offset + 2] / 255 - config.mean[2]) / config.std[2];
  }

  return tensor;
}

function isTriple(value: unknown): value is [number, number, number] {
  return Array.isArray(value)
    && value.length === 3
    && value.every((entry) => typeof entry === "number" && Number.isFinite(entry));
}
