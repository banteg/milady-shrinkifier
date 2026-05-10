import {
  DEFAULT_COLLECTED_AVATARS,
  DEFAULT_MATCHED_ACCOUNTS,
  DEFAULT_SETTINGS,
  DEFAULT_STATS,
} from "./constants";
import { normalizeHandle } from "./account-core";
import type {
  AvatarStorageRequest,
  AvatarStorageResponse,
  CollectedAvatar,
  CollectedAvatarMap,
  CollectedAvatarRecord,
  CollectedAvatarSummary,
  DetectionStats,
  ExtensionSettings,
  FilterMode,
  MatchedAccountMap,
} from "./types";
import { isFilterMode } from "./types";

export async function loadSettings(): Promise<ExtensionSettings> {
  const stored = await chrome.storage.sync.get({
    mode: DEFAULT_SETTINGS.mode,
    whitelistHandles: DEFAULT_SETTINGS.whitelistHandles,
  });
  return {
    mode: normalizeFilterMode(stored.mode),
    whitelistHandles: normalizeWhitelistHandles(stored.whitelistHandles),
  };
}

export async function saveSettings(settings: ExtensionSettings): Promise<void> {
  await chrome.storage.sync.set({
    mode: settings.mode,
    whitelistHandles: normalizeWhitelistHandles(settings.whitelistHandles),
  });
}

export async function loadStats(): Promise<DetectionStats> {
  const stored = await chrome.storage.local.get({
    stats: DEFAULT_STATS,
  });
  return normalizeStats(stored.stats);
}

export async function saveStats(stats: DetectionStats): Promise<void> {
  await chrome.storage.local.set({
    stats,
  });
}

export async function loadMatchedAccounts(): Promise<MatchedAccountMap> {
  const stored = await chrome.storage.local.get({
    matchedAccounts: DEFAULT_MATCHED_ACCOUNTS,
  });
  return normalizeMatchedAccounts(stored.matchedAccounts);
}

export async function saveMatchedAccounts(matchedAccounts: MatchedAccountMap): Promise<void> {
  await chrome.storage.local.set({
    matchedAccounts,
  });
}

export async function loadCollectedAvatars(): Promise<CollectedAvatarMap> {
  const response = await sendAvatarStorageRequest({ type: "avatars:list" });
  if (!("avatars" in response)) {
    throw new Error("Avatar storage returned no avatar list");
  }
  return normalizeCollectedAvatarArray(response.avatars);
}

export async function loadCollectedAvatarSummary(): Promise<CollectedAvatarSummary> {
  const response = await sendAvatarStorageRequest({ type: "avatars:summary" });
  if (!("summary" in response)) {
    throw new Error("Avatar storage returned no summary");
  }
  return response.summary;
}

export async function saveCollectedAvatars(collectedAvatars: CollectedAvatarMap): Promise<void> {
  await sendAvatarStorageRequest({
    type: "avatars:replace",
    avatars: Object.values(collectedAvatars),
  });
}

export async function resetStats(): Promise<void> {
  await saveStats(DEFAULT_STATS);
}

export async function resetMatchedAccounts(): Promise<void> {
  await saveMatchedAccounts(DEFAULT_MATCHED_ACCOUNTS);
}

export async function resetCollectedAvatars(): Promise<void> {
  await sendAvatarStorageRequest({ type: "avatars:reset" });
}

export async function saveCollectedAvatarBatch(avatars: CollectedAvatar[]): Promise<void> {
  if (avatars.length === 0) {
    return;
  }
  await sendAvatarStorageRequest({ type: "avatars:putMany", avatars });
}

export async function recordCollectedAvatarBatch(records: CollectedAvatarRecord[]): Promise<void> {
  if (records.length === 0) {
    return;
  }
  await sendAvatarStorageRequest({ type: "avatars:recordMany", records });
}

export function normalizeFilterMode(value: unknown): FilterMode {
  return isFilterMode(value) ? value : DEFAULT_SETTINGS.mode;
}

export function normalizeWhitelistHandles(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return DEFAULT_SETTINGS.whitelistHandles;
  }

  return Array.from(
    new Set(
      value
        .filter((handle): handle is string => typeof handle === "string")
        .map((handle) => normalizeHandle(handle))
        .filter((handle) => handle.length > 0),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

export function normalizeStats(value: unknown): DetectionStats {
  if (!isRecord(value)) {
    return DEFAULT_STATS;
  }

  return {
    tweetsScanned: readNumber(value.tweetsScanned),
    avatarsChecked: readNumber(value.avatarsChecked),
    cacheHits: readNumber(value.cacheHits),
    postsMatched: readNumber(value.postsMatched),
    modelMatches: readNumber(value.modelMatches),
    errors: readNumber(value.errors),
    lastMatchAt: readNullableString(value.lastMatchAt),
  };
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function normalizeMatchedAccounts(value: unknown): MatchedAccountMap {
  if (!isRecord(value)) {
    return DEFAULT_MATCHED_ACCOUNTS;
  }

  const normalized: MatchedAccountMap = {};

  for (const [key, entry] of Object.entries(value)) {
    if (!isRecord(entry)) {
      continue;
    }

    const handle = normalizeHandle(
      readString(entry.handle) ?? key,
    );
    if (!handle) {
      continue;
    }

    normalized[handle] = {
      handle,
      displayName: readNullableString(entry.displayName),
      postsMatched: readNumber(entry.postsMatched),
      lastMatchedAt: readNullableString(entry.lastMatchedAt),
    };
  }

  return normalized;
}

export function normalizeCollectedAvatars(value: unknown): CollectedAvatarMap {
  if (!isRecord(value)) {
    return DEFAULT_COLLECTED_AVATARS;
  }

  const normalized: CollectedAvatarMap = {};

  for (const [key, entry] of Object.entries(value)) {
    if (!isRecord(entry)) {
      continue;
    }

    const normalizedUrl = readString(entry.normalizedUrl) ?? key;
    if (!normalizedUrl) {
      continue;
    }

    normalized[normalizedUrl] = {
      normalizedUrl,
      originalUrl: readString(entry.originalUrl) ?? normalizedUrl,
      handles: uniqueStrings(entry.handles, normalizeHandle),
      displayNames: uniqueStrings(entry.displayNames),
      sourceSurfaces: uniqueStrings(entry.sourceSurfaces),
      seenCount: readNumber(entry.seenCount),
      firstSeenAt: readString(entry.firstSeenAt) ?? new Date(0).toISOString(),
      lastSeenAt: readString(entry.lastSeenAt) ?? new Date(0).toISOString(),
      exampleProfileUrl: readNullableString(entry.exampleProfileUrl),
      exampleNotificationUrl: readNullableString(entry.exampleNotificationUrl),
      exampleTweetUrl: readNullableString(entry.exampleTweetUrl),
      whitelisted: entry.whitelisted === true,
    };
  }

  return normalized;
}

export function normalizeCollectedAvatarArray(value: unknown): CollectedAvatarMap {
  if (!Array.isArray(value)) {
    return DEFAULT_COLLECTED_AVATARS;
  }

  const normalized: CollectedAvatarMap = {};
  for (const entry of value) {
    if (!isRecord(entry)) {
      continue;
    }
    const normalizedUrl = readString(entry.normalizedUrl);
    if (!normalizedUrl) {
      continue;
    }
    const entryMap = normalizeCollectedAvatars({ [normalizedUrl]: entry });
    const normalizedEntry = entryMap[normalizedUrl];
    if (normalizedEntry) {
      normalized[normalizedUrl] = normalizedEntry;
    }
  }
  return normalized;
}

async function sendAvatarStorageRequest(
  request: AvatarStorageRequest,
): Promise<Exclude<AvatarStorageResponse, { ok: false }>> {
  const response = await chrome.runtime.sendMessage(request) as AvatarStorageResponse | undefined;
  if (!response) {
    throw new Error("Avatar storage did not respond");
  }
  if (!response.ok) {
    throw new Error(response.error);
  }
  return response;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object";
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function uniqueStrings(
  value: unknown,
  map: (entry: string) => string = (entry) => entry.trim(),
): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return Array.from(
    new Set(
      value
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => map(entry))
        .filter((entry) => entry.length > 0),
    ),
  ).sort((left, right) => left.localeCompare(right));
}
