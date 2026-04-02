import * as ort from "onnxruntime-web";

import type { WorkerRequest, WorkerResponse } from "./shared/types";

interface InitMessage {
  modelUrl: string;
  wasmPath: string;
  positiveIndex: number;
  inputShape: [number, number, number, number];
}

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let positiveIndex = 1;
let inputShape: [number, number, number, number] | null = null;
let runQueue: Promise<void> = Promise.resolve();

self.addEventListener("message", (event: MessageEvent<InitMessage | WorkerRequest>) => {
  const data = event.data;

  if ("modelUrl" in data) {
    ort.env.wasm.wasmPaths = data.wasmPath;
    positiveIndex = data.positiveIndex;
    inputShape = data.inputShape;
    sessionPromise = ort.InferenceSession.create(data.modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    runQueue = Promise.resolve();
    return;
  }

  runQueue = runQueue
    .then(() => handleInferenceRequest(data))
    .catch((error: unknown) => {
      postErrorResponse(data.id, error);
    });
});

async function handleInferenceRequest(request: WorkerRequest): Promise<void> {
  if (!sessionPromise) {
    throw new Error("Worker used before model initialization");
  }
  if (!inputShape) {
    throw new Error("Worker used before input shape initialization");
  }

  const session = await sessionPromise;
  const tensor = new ort.Tensor("float32", request.tensor, inputShape);
  const output = await runModel(session, tensor);
  const score = scoreFromOutput(output, 0, output.length);
  self.postMessage({
    id: request.id,
    score,
  } satisfies WorkerResponse);
}

function postErrorResponse(id: string, error: unknown): void {
  const response: WorkerResponse = {
    id,
    error: error instanceof Error ? error.message : String(error),
  };
  self.postMessage(response);
}

async function runModel(session: ort.InferenceSession, tensor: ort.Tensor): Promise<number[]> {
  const outputName = session.outputNames[0];
  const result = await session.run({
    input: tensor,
  });
  return Array.from(result[outputName].data as Iterable<number>);
}

function scoreFromOutput(output: number[], offset: number, classStride: number): number {
  return classStride === 1
    ? Number(output[offset] ?? 0)
    : Number(output[offset + positiveIndex] ?? output[offset] ?? 0);
}
