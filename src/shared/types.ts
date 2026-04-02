export type FilterMode = "off" | "hide" | "fade" | "debug";

export interface ExtensionSettings {
  mode: FilterMode;
  whitelistHandles: string[];
}

export interface DetectionStats {
  tweetsScanned: number;
  avatarsChecked: number;
  cacheHits: number;
  postsMatched: number;
  modelMatches: number;
  errors: number;
  lastMatchAt: string | null;
}

export interface MatchedAccount {
  handle: string;
  displayName: string | null;
  postsMatched: number;
  lastMatchedAt: string | null;
}

export type MatchedAccountMap = Record<string, MatchedAccount>;

export interface CollectedAvatar {
  normalizedUrl: string;
  originalUrl: string;
  handles: string[];
  displayNames: string[];
  sourceSurfaces: string[];
  seenCount: number;
  firstSeenAt: string;
  lastSeenAt: string;
  exampleProfileUrl: string | null;
  exampleNotificationUrl: string | null;
  exampleTweetUrl: string | null;
  whitelisted: boolean;
}

export type CollectedAvatarMap = Record<string, CollectedAvatar>;

export interface ModelMetadata {
  architecture: string;
  class_names: [string, string];
  input_size: number;
  channels: number;
  mean: [number, number, number];
  std: [number, number, number];
  positive_index: number;
  generated_at: string;
  threshold: number;
  run_id: string;
}

export interface DetectionResult {
  matched: boolean;
  source: "onnx" | null;
  score: number | null;
  debugLabel?: string | null;
}

export interface WorkerInferenceRequest {
  id: string;
  tensor: Float32Array;
}

export interface WorkerSuccessResponse {
  id: string;
  score: number;
}

export interface WorkerErrorResponse {
  id: string;
  error: string;
}

export type WorkerRequest = WorkerInferenceRequest;
export type WorkerResponse = WorkerSuccessResponse | WorkerErrorResponse;
