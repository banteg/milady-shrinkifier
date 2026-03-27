import { DEFAULT_SETTINGS } from "./shared/constants";
import {
  loadCollectedAvatars,
  loadMatchedAccounts,
  loadSettings,
  loadStats,
  resetCollectedAvatars,
  resetMatchedAccounts,
  resetStats,
  saveSettings,
} from "./shared/storage";
import type {
  CollectedAvatar,
  CollectedAvatarMap,
  DetectionStats,
  FilterMode,
  MatchedAccount,
  MatchedAccountMap,
} from "./shared/types";

const styles = document.createElement("style");
styles.textContent = `
  :root {
    color-scheme: dark;
    font-family: "Avenir Next", "Segoe UI", sans-serif;
  }

  body {
    margin: 0;
    background:
      radial-gradient(circle at top right, rgba(255, 109, 74, 0.18), transparent 34%),
      linear-gradient(180deg, #15121b 0%, #0e0b12 100%);
    color: #f7f1e8;
    min-width: 300px;
  }

  .popup {
    padding: 18px 16px 14px;
  }

  h1 {
    margin: 0 0 6px;
    font-size: 19px;
    font-weight: 650;
  }

  .lede,
  .footnote {
    margin: 0;
    color: rgba(247, 241, 232, 0.7);
    font-size: 12px;
    line-height: 1.5;
  }

  .section-note {
    margin: 0 0 10px;
    color: rgba(247, 241, 232, 0.56);
    font-size: 11px;
    line-height: 1.45;
  }

  .mode-form {
    display: grid;
    gap: 0;
    margin: 18px 0 14px;
    border-top: 1px solid rgba(247, 241, 232, 0.12);
  }

  .mode-form label {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 2px;
    border-bottom: 1px solid rgba(247, 241, 232, 0.08);
    font-size: 13px;
  }

  .mode-form input {
    accent-color: #ff6d4a;
  }

  .stats,
  .accounts {
    margin: 0;
    padding: 14px 0 0;
    border-top: 1px solid rgba(247, 241, 232, 0.12);
  }

  .stats-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }

  h2 {
    margin: 0;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(247, 241, 232, 0.56);
  }

  #reset-stats {
    border: 0;
    padding: 0;
    background: transparent;
    color: rgba(247, 241, 232, 0.7);
    font: inherit;
    font-size: 12px;
    cursor: pointer;
  }

  .section-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .section-actions button {
    border: 0;
    padding: 0;
    background: transparent;
    color: rgba(247, 241, 232, 0.7);
    font: inherit;
    font-size: 12px;
    cursor: pointer;
  }

  .section-actions button:disabled {
    color: rgba(247, 241, 232, 0.3);
    cursor: default;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 16px;
    margin: 0;
  }

  .stats-grid div {
    min-width: 0;
  }

  .stats-grid dt {
    color: rgba(247, 241, 232, 0.5);
    font-size: 10px;
    margin: 0 0 2px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .stats-grid dd {
    margin: 0;
    font-size: 15px;
    font-weight: 620;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .accounts-list {
    display: grid;
    gap: 0;
  }

  .account-row {
    display: block;
    width: 100%;
    padding: 9px 0;
    border: 0;
    border-bottom: 1px solid rgba(247, 241, 232, 0.08);
    background: transparent;
    color: inherit;
    text-align: left;
    cursor: pointer;
  }

  .account-meta {
    min-width: 0;
  }

  .account-handle {
    margin: 0;
    font-size: 14px;
    font-weight: 620;
  }

  .account-subline {
    margin: 3px 0 0;
    color: rgba(247, 241, 232, 0.54);
    font-size: 12px;
  }

  .account-row[data-whitelisted="true"] .account-handle,
  .account-row[data-whitelisted="true"] .account-subline {
    text-decoration: line-through;
    text-decoration-thickness: 1.5px;
    color: rgba(247, 241, 232, 0.42);
  }

  .account-row:focus-visible {
    outline: 1px solid rgba(255, 109, 74, 0.9);
    outline-offset: 2px;
  }
`;
document.head.append(styles);

let currentSettings = DEFAULT_SETTINGS;
let currentMatchedAccounts: MatchedAccountMap = {};
let currentCollectedAvatars: CollectedAvatarMap = {};

void init();

async function init(): Promise<void> {
  const form = document.getElementById("mode-form");
  const resetButton = document.getElementById("reset-stats");
  const accountsList = document.getElementById("accounts-list");
  const exportAvatarsButton = document.getElementById("export-avatars");
  const resetAvatarsButton = document.getElementById("reset-avatars");
  if (
    !(form instanceof HTMLFormElement) ||
    !(resetButton instanceof HTMLButtonElement) ||
    !(accountsList instanceof HTMLDivElement) ||
    !(exportAvatarsButton instanceof HTMLButtonElement) ||
    !(resetAvatarsButton instanceof HTMLButtonElement)
  ) {
    return;
  }

  const [settings, stats, matchedAccounts, collectedAvatars] = await Promise.all([
    loadSettings(),
    loadStats(),
    loadMatchedAccounts(),
    loadCollectedAvatars(),
  ]);
  currentSettings = settings;
  currentMatchedAccounts = matchedAccounts;
  currentCollectedAvatars = collectedAvatars;
  setModeSelection(form, currentSettings.mode);
  renderStats(stats);
  renderMatchedAccounts(currentMatchedAccounts, currentSettings.whitelistHandles);
  renderCollectedAvatarSummary(currentCollectedAvatars);

  form.addEventListener("change", async () => {
    const mode = getSelectedMode(form);
    currentSettings = {
      ...currentSettings,
      mode,
    };
    await saveSettings(currentSettings);
  });

  resetButton.addEventListener("click", async () => {
    await Promise.all([resetStats(), resetMatchedAccounts()]);
    renderStats(await loadStats());
    currentMatchedAccounts = await loadMatchedAccounts();
    renderMatchedAccounts(currentMatchedAccounts, currentSettings.whitelistHandles);
  });

  resetAvatarsButton.addEventListener("click", async () => {
    await resetCollectedAvatars();
    currentCollectedAvatars = await loadCollectedAvatars();
    renderCollectedAvatarSummary(currentCollectedAvatars);
  });

  exportAvatarsButton.addEventListener("click", () => {
    exportCollectedAvatars(currentCollectedAvatars, currentSettings.whitelistHandles);
  });

  accountsList.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const row = target.closest<HTMLElement>("[data-handle]");
    if (!row) {
      return;
    }

    const handle = row.dataset.handle;
    if (!handle) {
      return;
    }

    const nextWhitelist = currentSettings.whitelistHandles.includes(handle)
      ? currentSettings.whitelistHandles.filter((entry) => entry !== handle)
      : [...currentSettings.whitelistHandles, handle].sort((left, right) =>
          left.localeCompare(right),
        );

    currentSettings = {
      ...currentSettings,
      whitelistHandles: nextWhitelist,
    };
    await saveSettings(currentSettings);
    renderMatchedAccounts(currentMatchedAccounts, currentSettings.whitelistHandles);
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local") {
      if (changes.stats) {
        renderStats(normalizeStats(changes.stats.newValue));
      }
      if (changes.matchedAccounts) {
        currentMatchedAccounts = normalizeMatchedAccounts(changes.matchedAccounts.newValue);
        renderMatchedAccounts(currentMatchedAccounts, currentSettings.whitelistHandles);
      }
      if (changes.collectedAvatars) {
        currentCollectedAvatars = normalizeCollectedAvatars(changes.collectedAvatars.newValue);
        renderCollectedAvatarSummary(currentCollectedAvatars);
      }
    }

    if (area === "sync") {
      if (changes.mode) {
        const mode = getStoredMode(changes.mode.newValue);
        currentSettings = {
          ...currentSettings,
          mode,
        };
        setModeSelection(form, mode);
      }
      if (changes.whitelistHandles) {
        currentSettings = {
          ...currentSettings,
          whitelistHandles: normalizeWhitelistHandles(changes.whitelistHandles.newValue),
        };
        renderMatchedAccounts(currentMatchedAccounts, currentSettings.whitelistHandles);
      }
    }
  });
}

function getSelectedMode(form: HTMLFormElement): FilterMode {
  const selected = new FormData(form).get("mode");
  if (
    selected === "hide" ||
    selected === "fade" ||
    selected === "debug" ||
    selected === "off"
  ) {
    return selected;
  }
  return DEFAULT_SETTINGS.mode;
}

function getStoredMode(value: unknown): FilterMode {
  if (value === "hide" || value === "fade" || value === "debug" || value === "off") {
    return value;
  }
  return DEFAULT_SETTINGS.mode;
}

function setModeSelection(form: HTMLFormElement, mode: FilterMode): void {
  const input = form.querySelector<HTMLInputElement>(`input[name="mode"][value="${mode}"]`);
  if (input) {
    input.checked = true;
  }
}

function renderStats(stats: DetectionStats): void {
  writeStat("tweetsScanned", formatNumber(stats.tweetsScanned));
  writeStat("avatarsChecked", formatNumber(stats.avatarsChecked));
  writeStat("cacheHits", formatNumber(stats.cacheHits));
  writeStat("postsMatched", formatNumber(stats.postsMatched));
  writeStat("phashMatches", formatNumber(stats.phashMatches));
  writeStat("onnxMatches", formatNumber(stats.onnxMatches));
  writeStat("errors", formatNumber(stats.errors));
  writeStat("lastMatchAt", stats.lastMatchAt ? formatDate(stats.lastMatchAt) : "Never");
}

function renderMatchedAccounts(
  matchedAccounts: MatchedAccountMap,
  whitelistHandles: string[],
): void {
  const accountsList = document.getElementById("accounts-list");
  const empty = document.getElementById("accounts-empty");
  if (!(accountsList instanceof HTMLDivElement) || !(empty instanceof HTMLParagraphElement)) {
    return;
  }

  const entries = Object.values(matchedAccounts).sort(compareAccounts);
  empty.hidden = entries.length > 0;
  accountsList.replaceChildren();

  for (const account of entries) {
    const row = document.createElement("button");
    const isWhitelisted = whitelistHandles.includes(account.handle);
    row.type = "button";
    row.className = "account-row";
    row.dataset.handle = account.handle;
    row.dataset.whitelisted = String(isWhitelisted);
    row.title = isWhitelisted
      ? `@${account.handle} bypasses the filter`
      : `Click to let @${account.handle} bypass the filter`;

    const meta = document.createElement("div");
    meta.className = "account-meta";

    const handle = document.createElement("p");
    handle.className = "account-handle";
    handle.textContent = `@${account.handle}`;

    const subline = document.createElement("p");
    subline.className = "account-subline";
    subline.textContent = isWhitelisted
      ? `${formatNumber(account.postsMatched)} matched posts, bypassed`
      : `${formatNumber(account.postsMatched)} matched posts`;

    meta.append(handle, subline);
    row.append(meta);
    accountsList.append(row);
  }
}

function renderCollectedAvatarSummary(collectedAvatars: CollectedAvatarMap): void {
  const avatars = Object.values(collectedAvatars);
  writeCollectionStat("uniqueAvatars", formatNumber(avatars.length));
  writeCollectionStat(
    "totalSightings",
    formatNumber(avatars.reduce((total, avatar) => total + avatar.seenCount, 0)),
  );

  const exportButton = document.getElementById("export-avatars");
  const resetButton = document.getElementById("reset-avatars");
  if (exportButton instanceof HTMLButtonElement) {
    exportButton.disabled = avatars.length === 0;
  }
  if (resetButton instanceof HTMLButtonElement) {
    resetButton.disabled = avatars.length === 0;
  }
}

function writeStat(name: keyof DetectionStats, value: string): void {
  const node = document.querySelector<HTMLElement>(`[data-stat="${name}"]`);
  if (node) {
    node.textContent = value;
  }
}

function writeCollectionStat(name: "uniqueAvatars" | "totalSightings", value: string): void {
  const node = document.querySelector<HTMLElement>(`[data-collection-stat="${name}"]`);
  if (node) {
    node.textContent = value;
  }
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Never" : date.toLocaleString();
}

function normalizeStats(value: unknown): DetectionStats {
  if (!value || typeof value !== "object") {
    return emptyStats();
  }

  const candidate = value as Partial<DetectionStats>;
  return {
    tweetsScanned: readNumber(candidate.tweetsScanned),
    avatarsChecked: readNumber(candidate.avatarsChecked),
    cacheHits: readNumber(candidate.cacheHits),
    postsMatched: readNumber(candidate.postsMatched),
    phashMatches: readNumber(candidate.phashMatches),
    onnxMatches: readNumber(candidate.onnxMatches),
    errors: readNumber(candidate.errors),
    lastMatchAt: typeof candidate.lastMatchAt === "string" ? candidate.lastMatchAt : null,
  };
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function normalizeWhitelistHandles(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return Array.from(
    new Set(
      value
        .filter((handle): handle is string => typeof handle === "string")
        .map((handle) => handle.trim().replace(/^@+/, "").toLowerCase())
        .filter((handle) => handle.length > 0),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

function normalizeMatchedAccounts(value: unknown): MatchedAccountMap {
  if (!value || typeof value !== "object") {
    return {};
  }

  const normalized: MatchedAccountMap = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const candidate = entry as Partial<MatchedAccount>;
    const handle = typeof candidate.handle === "string" && candidate.handle.length > 0
      ? candidate.handle
      : key;
    const normalizedHandle = handle.trim().replace(/^@+/, "").toLowerCase();
    if (!normalizedHandle) {
      continue;
    }
    normalized[normalizedHandle] = {
      handle: normalizedHandle,
      displayName: typeof candidate.displayName === "string" ? candidate.displayName : null,
      postsMatched: readNumber(candidate.postsMatched),
      lastMatchedAt: typeof candidate.lastMatchedAt === "string" ? candidate.lastMatchedAt : null,
    };
  }
  return normalized;
}

function normalizeCollectedAvatars(value: unknown): CollectedAvatarMap {
  if (!value || typeof value !== "object") {
    return {};
  }

  const normalized: CollectedAvatarMap = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }

    const candidate = entry as Partial<CollectedAvatar>;
    const normalizedUrl =
      typeof candidate.normalizedUrl === "string" && candidate.normalizedUrl.length > 0
        ? candidate.normalizedUrl
        : key;
    if (!normalizedUrl) {
      continue;
    }

    normalized[normalizedUrl] = {
      normalizedUrl,
      originalUrl:
        typeof candidate.originalUrl === "string" && candidate.originalUrl.length > 0
          ? candidate.originalUrl
          : normalizedUrl,
      handles: normalizeStringArray(candidate.handles, true),
      displayNames: normalizeStringArray(candidate.displayNames, false),
      seenCount: readNumber(candidate.seenCount),
      firstSeenAt:
        typeof candidate.firstSeenAt === "string" ? candidate.firstSeenAt : new Date(0).toISOString(),
      lastSeenAt:
        typeof candidate.lastSeenAt === "string" ? candidate.lastSeenAt : new Date(0).toISOString(),
      exampleProfileUrl:
        typeof candidate.exampleProfileUrl === "string" ? candidate.exampleProfileUrl : null,
      exampleTweetUrl: typeof candidate.exampleTweetUrl === "string" ? candidate.exampleTweetUrl : null,
      heuristicMatch:
        typeof candidate.heuristicMatch === "boolean" ? candidate.heuristicMatch : null,
      heuristicSource:
        candidate.heuristicSource === "phash" || candidate.heuristicSource === "onnx"
          ? candidate.heuristicSource
          : null,
      heuristicScore:
        typeof candidate.heuristicScore === "number" && Number.isFinite(candidate.heuristicScore)
          ? candidate.heuristicScore
          : null,
      heuristicTokenId:
        typeof candidate.heuristicTokenId === "number" && Number.isFinite(candidate.heuristicTokenId)
          ? candidate.heuristicTokenId
          : null,
      whitelisted: candidate.whitelisted === true,
    };
  }

  return normalized;
}

function normalizeStringArray(value: unknown, normalizeHandles: boolean): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return Array.from(
    new Set(
      value
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => normalizeHandles ? entry.trim().replace(/^@+/, "").toLowerCase() : entry.trim())
        .filter((entry) => entry.length > 0),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

function exportCollectedAvatars(
  collectedAvatars: CollectedAvatarMap,
  whitelistHandles: string[],
): void {
  const avatars = Object.values(collectedAvatars).sort(compareCollectedAvatars);
  if (avatars.length === 0) {
    return;
  }

  const payload = {
    version: 1,
    exportedAt: new Date().toISOString(),
    avatarCount: avatars.length,
    totalSightings: avatars.reduce((total, avatar) => total + avatar.seenCount, 0),
    whitelistHandles,
    avatars,
  };

  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `milady-shrinkifier-avatars-${timestampForFilename(new Date())}.json`;
  anchor.click();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 0);
}

function compareAccounts(left: MatchedAccount, right: MatchedAccount): number {
  if (right.postsMatched !== left.postsMatched) {
    return right.postsMatched - left.postsMatched;
  }
  return left.handle.localeCompare(right.handle);
}

function compareCollectedAvatars(left: CollectedAvatar, right: CollectedAvatar): number {
  if (right.seenCount !== left.seenCount) {
    return right.seenCount - left.seenCount;
  }
  return left.normalizedUrl.localeCompare(right.normalizedUrl);
}

function timestampForFilename(value: Date): string {
  return value.toISOString().replace(/[:.]/g, "-");
}

function emptyStats(): DetectionStats {
  return {
    tweetsScanned: 0,
    avatarsChecked: 0,
    cacheHits: 0,
    postsMatched: 0,
    phashMatches: 0,
    onnxMatches: 0,
    errors: 0,
    lastMatchAt: null,
  };
}
