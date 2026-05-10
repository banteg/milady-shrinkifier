import {
  normalizeCollectedAvatarArray,
  normalizeCollectedAvatars,
} from "./shared/storage";
import { normalizeHandle } from "./shared/account-core";
import type {
  AvatarStorageRequest,
  AvatarStorageResponse,
  CollectedAvatar,
  CollectedAvatarMap,
  CollectedAvatarRecord,
  CollectedAvatarSummary,
} from "./shared/types";

const DB_NAME = "milady-shrinkifier";
const DB_VERSION = 1;
const AVATAR_STORE = "avatars";
const MIGRATION_VERSION_KEY = "collectedAvatarsMigrationVersion";
const CURRENT_MIGRATION_VERSION = 1;

let dbPromise: Promise<IDBDatabase> | null = null;
let migrationPromise: Promise<void> | null = null;

chrome.runtime.onMessage.addListener((
  request: AvatarStorageRequest,
  _sender,
  sendResponse: (response: AvatarStorageResponse) => void,
) => {
  if (!isAvatarStorageRequest(request)) {
    return false;
  }

  void handleAvatarStorageRequest(request)
    .then((response) => sendResponse(response))
    .catch((error: unknown) => {
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  return true;
});

async function handleAvatarStorageRequest(
  request: AvatarStorageRequest,
): Promise<AvatarStorageResponse> {
  await ensureMigrated();

  switch (request.type) {
    case "avatars:list":
      return {
        ok: true,
        avatars: await listAvatars(),
      };
    case "avatars:summary":
      return {
        ok: true,
        summary: await summarizeAvatars(),
      };
    case "avatars:recordMany":
      await recordAvatars(request.records);
      return { ok: true };
    case "avatars:putMany":
      await putAvatars(normalizeCollectedAvatarArray(request.avatars));
      return { ok: true };
    case "avatars:replace":
      await replaceAvatars(normalizeCollectedAvatarArray(request.avatars));
      return { ok: true };
    case "avatars:reset":
      await clearAvatars();
      return { ok: true };
  }
}

async function ensureMigrated(): Promise<void> {
  if (!migrationPromise) {
    migrationPromise = migrateLegacyCollectedAvatars();
  }
  await migrationPromise;
}

async function migrateLegacyCollectedAvatars(): Promise<void> {
  const stored = await chrome.storage.local.get({
    [MIGRATION_VERSION_KEY]: 0,
    collectedAvatars: undefined,
  });
  const migrationVersion = typeof stored[MIGRATION_VERSION_KEY] === "number"
    ? stored[MIGRATION_VERSION_KEY]
    : 0;
  if (migrationVersion >= CURRENT_MIGRATION_VERSION) {
    return;
  }

  const legacyAvatars = normalizeCollectedAvatars(stored.collectedAvatars);
  if (Object.keys(legacyAvatars).length > 0) {
    await putAvatars(legacyAvatars);
  }
  await chrome.storage.local.set({
    [MIGRATION_VERSION_KEY]: CURRENT_MIGRATION_VERSION,
  });
}

async function openDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.addEventListener("upgradeneeded", () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(AVATAR_STORE)) {
          db.createObjectStore(AVATAR_STORE, { keyPath: "normalizedUrl" });
        }
      });
      request.addEventListener("success", () => resolve(request.result));
      request.addEventListener("error", () => reject(request.error ?? new Error("Failed to open avatar DB")));
    });
  }
  return dbPromise;
}

async function listAvatars(): Promise<CollectedAvatar[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(AVATAR_STORE, "readonly");
    const store = transaction.objectStore(AVATAR_STORE);
    const request = store.getAll();
    request.addEventListener("success", () => {
      resolve(Object.values(normalizeCollectedAvatarArray(request.result)));
    });
    request.addEventListener("error", () => reject(request.error ?? new Error("Failed to list avatars")));
  });
}

async function summarizeAvatars(): Promise<CollectedAvatarSummary> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const summary: CollectedAvatarSummary = {
      avatarCount: 0,
      totalSightings: 0,
    };
    const transaction = db.transaction(AVATAR_STORE, "readonly");
    const store = transaction.objectStore(AVATAR_STORE);
    const request = store.openCursor();
    request.addEventListener("success", () => {
      const cursor = request.result;
      if (!cursor) {
        resolve(summary);
        return;
      }
      const avatar = Object.values(normalizeCollectedAvatarArray([cursor.value]))[0];
      if (avatar) {
        summary.avatarCount += 1;
        summary.totalSightings += avatar.seenCount;
      }
      cursor.continue();
    });
    request.addEventListener("error", () => reject(request.error ?? new Error("Failed to summarize avatars")));
  });
}

async function putAvatars(avatars: CollectedAvatarMap): Promise<void> {
  const db = await openDb();
  await runAvatarTransaction("readwrite", (store) => {
    for (const avatar of Object.values(avatars)) {
      store.put(avatar);
    }
  }, db);
}

async function recordAvatars(records: CollectedAvatarRecord[]): Promise<void> {
  for (const record of records) {
    await recordAvatar(record);
  }
}

async function recordAvatar(record: CollectedAvatarRecord): Promise<void> {
  const normalizedRecord = normalizeCollectedAvatarRecord(record);
  if (!normalizedRecord) {
    return;
  }

  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(AVATAR_STORE, "readwrite");
    const store = transaction.objectStore(AVATAR_STORE);
    const request = store.get(normalizedRecord.normalizedUrl);
    transaction.addEventListener("complete", () => resolve());
    transaction.addEventListener("error", () => reject(transaction.error ?? new Error("Avatar DB transaction failed")));
    transaction.addEventListener("abort", () => reject(transaction.error ?? new Error("Avatar DB transaction aborted")));
    request.addEventListener("success", () => {
      const existing = Object.values(normalizeCollectedAvatarArray([request.result]))[0];
      store.put(mergeAvatarRecord(existing, normalizedRecord));
    });
    request.addEventListener("error", () => reject(request.error ?? new Error("Failed to read avatar")));
  });
}

async function replaceAvatars(avatars: CollectedAvatarMap): Promise<void> {
  const db = await openDb();
  await runAvatarTransaction("readwrite", (store) => {
    store.clear();
    for (const avatar of Object.values(avatars)) {
      store.put(avatar);
    }
  }, db);
}

async function clearAvatars(): Promise<void> {
  const db = await openDb();
  await runAvatarTransaction("readwrite", (store) => {
    store.clear();
  }, db);
  await chrome.storage.local.remove(["collectedAvatars", "collectedAvatarsRevision"]);
  await chrome.storage.local.set({
    [MIGRATION_VERSION_KEY]: CURRENT_MIGRATION_VERSION,
  });
}

function normalizeCollectedAvatarRecord(record: CollectedAvatarRecord): CollectedAvatarRecord | null {
  const normalizedUrl = typeof record.normalizedUrl === "string" ? record.normalizedUrl : "";
  if (!normalizedUrl) {
    return null;
  }

  return {
    normalizedUrl,
    originalUrl: typeof record.originalUrl === "string" && record.originalUrl ? record.originalUrl : normalizedUrl,
    handle: normalizeHandle(record.handle),
    displayName: typeof record.displayName === "string" && record.displayName ? record.displayName.trim() : null,
    whitelisted: record.whitelisted === true,
    exampleTweetUrl: typeof record.exampleTweetUrl === "string" ? record.exampleTweetUrl : null,
    exampleNotificationUrl: typeof record.exampleNotificationUrl === "string" ? record.exampleNotificationUrl : null,
    sourceSurface: typeof record.sourceSurface === "string" && record.sourceSurface
      ? record.sourceSurface
      : "unknown",
  };
}

function mergeAvatarRecord(
  existing: CollectedAvatar | undefined,
  record: CollectedAvatarRecord,
): CollectedAvatar {
  const now = new Date().toISOString();
  return {
    normalizedUrl: record.normalizedUrl,
    originalUrl: record.originalUrl || existing?.originalUrl || record.normalizedUrl,
    handles: mergeUniqueStrings(existing?.handles, record.handle, true),
    displayNames: mergeUniqueStrings(existing?.displayNames, record.displayName, false),
    sourceSurfaces: mergeUniqueStrings(existing?.sourceSurfaces, record.sourceSurface, false),
    seenCount: (existing?.seenCount ?? 0) + 1,
    firstSeenAt: existing?.firstSeenAt ?? now,
    lastSeenAt: now,
    exampleProfileUrl: existing?.exampleProfileUrl ?? (record.handle ? toProfileUrl(record.handle) : null),
    exampleNotificationUrl: existing?.exampleNotificationUrl ?? record.exampleNotificationUrl,
    exampleTweetUrl: existing?.exampleTweetUrl ?? record.exampleTweetUrl,
    whitelisted: record.whitelisted || existing?.whitelisted === true,
  };
}

function mergeUniqueStrings(
  existing: string[] | undefined,
  incoming: string | null,
  normalizeHandles: boolean,
): string[] {
  const values = new Set(existing ?? []);
  const normalized = incoming
    ? (normalizeHandles ? normalizeHandle(incoming) : incoming.trim())
    : "";
  if (normalized) {
    values.add(normalized);
  }
  return Array.from(values).sort((left, right) => left.localeCompare(right));
}

function toProfileUrl(handle: string): string {
  return `https://x.com/${handle}`;
}

async function runAvatarTransaction(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => void,
  db: IDBDatabase,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(AVATAR_STORE, mode);
    transaction.addEventListener("complete", () => resolve());
    transaction.addEventListener("error", () => reject(transaction.error ?? new Error("Avatar DB transaction failed")));
    transaction.addEventListener("abort", () => reject(transaction.error ?? new Error("Avatar DB transaction aborted")));
    run(transaction.objectStore(AVATAR_STORE));
  });
}

function isAvatarStorageRequest(value: unknown): value is AvatarStorageRequest {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof value.type === "string" &&
    value.type.startsWith("avatars:")
  );
}
