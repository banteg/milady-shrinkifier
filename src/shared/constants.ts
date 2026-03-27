import type {
  CollectedAvatarMap,
  DetectionStats,
  ExtensionSettings,
  MatchedAccountMap,
} from "./types";

export const DEFAULT_SETTINGS: ExtensionSettings = {
  mode: "off",
  whitelistHandles: [],
};

export const DEFAULT_STATS: DetectionStats = {
  tweetsScanned: 0,
  avatarsChecked: 0,
  cacheHits: 0,
  postsMatched: 0,
  phashMatches: 0,
  onnxMatches: 0,
  errors: 0,
  lastMatchAt: null,
};

export const DEFAULT_MATCHED_ACCOUNTS: MatchedAccountMap = {};
export const DEFAULT_COLLECTED_AVATARS: CollectedAvatarMap = {};

export const HASH_MATCH_THRESHOLD = 8;
export const HASH_ONNX_THRESHOLD = 18;
export const COLOR_DISTANCE_THRESHOLD = 120;
export const MODEL_INPUT_LENGTH = 32 * 32;
export const HASH_URL = "generated/milady-maker.hashes.json";
export const MODEL_METADATA_URL = "generated/milady-prototype.meta.json";
export const MODEL_URL = "models/milady-prototype.onnx";
