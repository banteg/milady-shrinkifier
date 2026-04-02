import {
  computeCoverCropRegion,
  computeNormalizedTensorFromRgbBuffer,
  type RuntimeModelConfig,
} from "./model-config";

export async function loadCorsImage(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.decoding = "async";
  const loaded = new Promise<void>((resolve, reject) => {
    image.addEventListener("load", () => resolve(), { once: true });
    image.addEventListener("error", () => reject(new Error(`Avatar image failed to load: ${url}`)), {
      once: true,
    });
  });
  image.src = url;
  await loaded;
  return image;
}

export async function computeBrowserImageFeatures(
  image: HTMLImageElement,
  config: RuntimeModelConfig,
): Promise<Float32Array> {
  const classifierCanvas = document.createElement("canvas");
  classifierCanvas.width = config.inputSize;
  classifierCanvas.height = config.inputSize;
  const classifierContext = classifierCanvas.getContext("2d", { willReadFrequently: true });
  if (!classifierContext) {
    throw new Error("Unable to create classifier context");
  }
  drawCoverImage(classifierContext, image, config.inputSize, config.inputSize);
  const classifierPixels = classifierContext
    .getImageData(0, 0, config.inputSize, config.inputSize).data;

  return rgbaToModelTensor(classifierPixels, config);
}

function drawCoverImage(
  context: CanvasRenderingContext2D,
  image: CanvasImageSource,
  targetWidth: number,
  targetHeight: number,
): void {
  const imageWidth = "naturalWidth" in image ? image.naturalWidth : targetWidth;
  const imageHeight = "naturalHeight" in image ? image.naturalHeight : targetHeight;
  const region = computeCoverCropRegion(imageWidth, imageHeight, targetWidth, targetHeight, "center");

  context.clearRect(0, 0, targetWidth, targetHeight);
  context.drawImage(
    image,
    region.left,
    region.top,
    region.width,
    region.height,
    0,
    0,
    targetWidth,
    targetHeight,
  );
}

function rgbaToModelTensor(buffer: Uint8ClampedArray, config: RuntimeModelConfig): Float32Array {
  const pixelCount = config.inputSize * config.inputSize;
  const rgbBuffer = new Uint8Array(pixelCount * 3);

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const rgbaOffset = pixelIndex * 4;
    const rgbOffset = pixelIndex * 3;
    rgbBuffer[rgbOffset] = buffer[rgbaOffset];
    rgbBuffer[rgbOffset + 1] = buffer[rgbaOffset + 1];
    rgbBuffer[rgbOffset + 2] = buffer[rgbaOffset + 2];
  }

  return computeNormalizedTensorFromRgbBuffer(rgbBuffer, config);
}
