import {
  CLASSIFIER_MODEL_METADATA_URL,
  CLASSIFIER_MODEL_URL,
  DEFAULT_SETTINGS,
} from "./shared/constants";
import {
  loadCorsImage,
  computeBrowserImageFeatures,
} from "./shared/browser-image";
import { resolveRuntimeModelConfig, type RuntimeModelConfig } from "./shared/model-config";
import {
  normalizeProfileImageUrl,
} from "./shared/image-core";
import {
  loadCollectedAvatars,
  loadMatchedAccounts,
  loadSettings,
  loadStats,
  normalizeCollectedAvatars,
  normalizeMatchedAccounts,
  normalizeStats,
  normalizeWhitelistHandles,
  saveCollectedAvatars,
  saveMatchedAccounts,
  saveStats,
} from "./shared/storage";
import type {
  CollectedAvatarMap,
  DetectionStats,
  DetectionResult,
  ExtensionSettings,
  MatchedAccountMap,
  ModelMetadata,
  WorkerRequest,
  WorkerResponse,
} from "./shared/types";

const STYLE_ID = "milady-shrinkifier-style";
const ARTICLE_SELECTOR = 'article[data-testid="tweet"]';
const NOTIFICATION_SELECTOR = 'article[data-testid="notification"]';
const cache = new Map<string, Promise<DetectionResult>>();
const processed = new WeakMap<HTMLElement, string>();
const processedNotifications = new WeakMap<HTMLElement, string>();
const placeholders = new WeakMap<HTMLElement, HTMLDivElement>();
const revealed = new WeakMap<HTMLElement, string>();
const observedTweets = new WeakSet<HTMLElement>();
const observedNotifications = new WeakSet<HTMLElement>();
const visibleTweets = new Set<HTMLElement>();
const visibleNotifications = new Set<HTMLElement>();

let settings: ExtensionSettings = DEFAULT_SETTINGS;
let modelMetadataPromise: Promise<ResolvedModel> | null = null;
let workerPromise: Promise<Worker> | null = null;
let pendingWorker = new Map<string, { resolve: (score: number) => void; reject: (error: Error) => void }>();
let visibilityObserver: IntersectionObserver | null = null;
let scanScheduled = false;
let delayedScanTimer: number | null = null;
let stats: DetectionStats | null = null;
let matchedAccounts: MatchedAccountMap | null = null;
let collectedAvatars: CollectedAvatarMap | null = null;
let localStateWriteScheduled = false;

interface ResolvedModel {
  metadata: ModelMetadata;
  modelUrl: string;
  config: RuntimeModelConfig;
}

void boot();

async function boot(): Promise<void> {
  injectStyles();
  [settings, stats, matchedAccounts, collectedAvatars] = await Promise.all([
    loadSettings(),
    loadStats(),
    loadMatchedAccounts(),
    loadCollectedAvatars(),
  ]);
  observeStorage();
  visibilityObserver = new IntersectionObserver(handleVisibilityChange, {
    rootMargin: "200px 0px",
  });
  observeTrackableElements(document.body);
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of Array.from(record.addedNodes)) {
        if (node instanceof HTMLElement) {
          observeTrackableElements(node);
        }
      }
    }
    scheduleProcessVisibleTweets();
    scheduleDelayedProcessVisibleTweets();
  });
  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
  scheduleProcessVisibleTweets();
  scheduleDelayedProcessVisibleTweets();
}

async function processVisibleTweets(): Promise<void> {
  const tweets = Array.from(visibleTweets).filter((tweet) => tweet.isConnected);
  const notifications = Array.from(visibleNotifications).filter((notification) => notification.isConnected);
  for (const tweet of Array.from(visibleTweets)) {
    if (!tweet.isConnected) {
      visibleTweets.delete(tweet);
    }
  }
  for (const notification of Array.from(visibleNotifications)) {
    if (!notification.isConnected) {
      visibleNotifications.delete(notification);
    }
  }
  await Promise.allSettled([
    ...tweets.map((tweet) => processTweet(tweet)),
    ...notifications.map((notification) => processNotificationGroup(notification)),
  ]);
}

function handleVisibilityChange(entries: IntersectionObserverEntry[]): void {
  for (const entry of entries) {
    const element = entry.target as HTMLElement;
    const targetSet = element.matches(ARTICLE_SELECTOR) ? visibleTweets : visibleNotifications;
    if (entry.isIntersecting) {
      targetSet.add(element);
      continue;
    }
    targetSet.delete(element);
  }
  scheduleProcessVisibleTweets();
  scheduleDelayedProcessVisibleTweets();
}

function observeTrackableElements(root: ParentNode): void {
  if (!visibilityObserver) {
    return;
  }

  if (root instanceof HTMLElement) {
    observeTrackableElement(root);
  }

  for (const tweet of Array.from(root.querySelectorAll<HTMLElement>(ARTICLE_SELECTOR))) {
    observeTrackableElement(tweet);
  }
  for (const notification of Array.from(root.querySelectorAll<HTMLElement>(NOTIFICATION_SELECTOR))) {
    observeTrackableElement(notification);
  }
}

function observeTrackableElement(element: HTMLElement): void {
  if (!visibilityObserver) {
    return;
  }

  if (element.matches(ARTICLE_SELECTOR)) {
    if (observedTweets.has(element)) {
      return;
    }
    observedTweets.add(element);
    visibilityObserver.observe(element);
    return;
  }

  if (element.matches(NOTIFICATION_SELECTOR)) {
    if (observedNotifications.has(element)) {
      return;
    }
    observedNotifications.add(element);
    visibilityObserver.observe(element);
  }
}

function scheduleProcessVisibleTweets(): void {
  if (scanScheduled) {
    return;
  }
  scanScheduled = true;
  queueMicrotask(async () => {
    scanScheduled = false;
    await processVisibleTweets();
  });
}

function scheduleDelayedProcessVisibleTweets(): void {
  if (delayedScanTimer !== null) {
    window.clearTimeout(delayedScanTimer);
  }

  delayedScanTimer = window.setTimeout(() => {
    delayedScanTimer = null;
    scheduleProcessVisibleTweets();
  }, 350);
}

function reapplyModeToVisibleTweets(): void {
  for (const tweet of Array.from(visibleTweets)) {
    if (!tweet.isConnected) {
      visibleTweets.delete(tweet);
      continue;
    }
    applyMode(tweet, processed.get(tweet));
  }
}

async function processTweet(tweet: HTMLElement): Promise<void> {
  try {
    const avatar = findAvatar(tweet);
    const author = findAuthor(tweet);
    if (!avatar) {
      setWhitelistState(tweet, false);
      tweet.dataset.miladyShrinkifierState = "miss";
      delete tweet.dataset.miladyShrinkifierDebug;
      applyMode(tweet);
      scheduleDelayedProcessVisibleTweets();
      return;
    }

    if (!avatar.currentSrc && !avatar.src) {
      setWhitelistState(tweet, false);
      tweet.dataset.miladyShrinkifierState = "miss";
      delete tweet.dataset.miladyShrinkifierDebug;
      applyMode(tweet);
      scheduleDelayedProcessVisibleTweets();
      return;
    }

    const normalizedUrl = normalizeProfileImageUrl(avatar.currentSrc || avatar.src);
    const isWhitelisted = author ? settings.whitelistHandles.includes(author.handle) : false;
    setWhitelistState(tweet, isWhitelisted);
    if (revealed.get(tweet) && revealed.get(tweet) !== normalizedUrl) {
      revealed.delete(tweet);
    }

    if (processed.get(tweet) === normalizedUrl && tweet.dataset.miladyShrinkifierState) {
      applyMode(tweet, normalizedUrl);
      return;
    }

    processed.set(tweet, normalizedUrl);

    tweet.dataset.miladyShrinkifierState = "miss";
    tweet.dataset.miladyShrinkifierDebug = "…";
    applyMode(tweet, normalizedUrl);
    incrementStat("tweetsScanned");
    const result = await detectAvatar(avatar, normalizedUrl);
    if (result.debugLabel) {
      tweet.dataset.miladyShrinkifierDebug = result.debugLabel;
    } else {
      delete tweet.dataset.miladyShrinkifierDebug;
    }
    recordCollectedAvatar({
      normalizedUrl,
      originalUrl: avatar.currentSrc || avatar.src,
      author,
      whitelisted: isWhitelisted,
      exampleTweetUrl: findTweetUrl(tweet),
      exampleNotificationUrl: null,
      sourceSurface: "tweet",
      result,
    });
    if (result.matched) {
      tweet.dataset.miladyShrinkifier = result.source ?? "match";
      tweet.dataset.miladyShrinkifierState = "match";
      if (!isWhitelisted) {
        incrementMatchStats(result);
      }
      if (author && !isWhitelisted) {
        recordMatchedAccount(author.handle, author.displayName);
      }
      applyMode(tweet, normalizedUrl);
      return;
    }

    revealed.delete(tweet);
    clearEffects(tweet);
    delete tweet.dataset.miladyShrinkifier;
    tweet.dataset.miladyShrinkifierState = "miss";
    if (result.debugLabel) {
      tweet.dataset.miladyShrinkifierDebug = result.debugLabel;
    }
    applyMode(tweet, normalizedUrl);
  } catch (error) {
    console.error("Milady post processing failed", error);
    setWhitelistState(tweet, false);
    clearEffects(tweet);
    delete tweet.dataset.miladyShrinkifier;
    tweet.dataset.miladyShrinkifierState = "miss";
    tweet.dataset.miladyShrinkifierDebug = "err";
    applyMode(tweet);
  }
}

async function processNotificationGroup(notification: HTMLElement): Promise<void> {
  const avatarEntries = collectNotificationAvatarEntries(notification);
  if (avatarEntries.length === 0) {
    return;
  }

  const signature = avatarEntries
    .map((entry) => `${entry.handle}:${entry.normalizedUrl}`)
    .sort()
    .join("|");
  if (processedNotifications.get(notification) === signature) {
    return;
  }
  processedNotifications.set(notification, signature);

  for (const entry of avatarEntries) {
    recordCollectedAvatar({
      normalizedUrl: entry.normalizedUrl,
      originalUrl: entry.originalUrl,
      author: {
        handle: entry.handle,
        displayName: null,
      },
      whitelisted: settings.whitelistHandles.includes(entry.handle),
      exampleTweetUrl: null,
      exampleNotificationUrl: window.location.href,
      sourceSurface: "notification-group",
    });
  }
}

async function detectAvatar(image: HTMLImageElement, normalizedUrl: string): Promise<DetectionResult> {
  const cached = cache.get(normalizedUrl);
  if (cached) {
    incrementStat("cacheHits");
    return cached;
  }

  const task = detectAvatarUncached(normalizedUrl);
  cache.set(normalizedUrl, task);
  return task;
}

async function detectAvatarUncached(normalizedUrl: string): Promise<DetectionResult> {
  incrementStat("avatarsChecked");
  try {
    const resolvedModel = await loadModelMetadata();
    const runtimeImage = await loadCorsImage(normalizedUrl);
    const variants = await Promise.all([
      computeBrowserImageFeatures(runtimeImage, resolvedModel.config, "center"),
      computeBrowserImageFeatures(runtimeImage, resolvedModel.config, "top"),
    ]);
    const score = await scoreWithOnnx(
      resolvedModel,
      variants,
      normalizedUrl,
    );
    return {
      matched: score >= resolvedModel.metadata.threshold,
      source: score >= resolvedModel.metadata.threshold ? "onnx" : null,
      score,
      tokenId: null,
      debugLabel: formatProbabilityDebugLabel(score, resolvedModel.metadata.threshold),
    };
  } catch (error) {
    console.error("Milady detection failed", error);
    incrementStat("errors");
    return {
      matched: false,
      source: null,
      score: null,
      tokenId: null,
      debugLabel: "err",
    };
  }
}

function findAvatar(tweet: HTMLElement): HTMLImageElement | null {
  return (
    tweet.querySelector<HTMLImageElement>('[data-testid="Tweet-User-Avatar"] img[src*="profile_images"]') ??
    tweet.querySelector<HTMLImageElement>('img[src*="profile_images"]')
  );
}

function findAuthor(tweet: HTMLElement): { handle: string; displayName: string | null } | null {
  const avatarLink = tweet.querySelector<HTMLAnchorElement>(
    '[data-testid="Tweet-User-Avatar"] a[href^="/"]',
  );
  const handle = normalizeHandle(avatarLink?.getAttribute("href"));
  if (!handle) {
    return null;
  }

  const userName = tweet.querySelector<HTMLElement>('[data-testid="User-Name"]');
  return {
    handle,
    displayName: userName ? extractDisplayName(userName) : null,
  };
}

function applyMode(tweet: HTMLElement, normalizedUrl?: string): void {
  clearVisualState(tweet);
  const isMatch = tweet.dataset.miladyShrinkifierState === "match";
  const isWhitelisted = tweet.dataset.miladyShrinkifierWhitelisted === "true";

  if (isWhitelisted && settings.mode !== "debug") {
    clearPlaceholder(tweet);
    tweet.style.display = "";
    return;
  }

  switch (settings.mode) {
    case "hide":
      if (!isMatch) {
        revealed.delete(tweet);
        clearPlaceholder(tweet);
        tweet.style.display = "";
        return;
      }
      if (normalizedUrl && revealed.get(tweet) === normalizedUrl) {
        clearPlaceholder(tweet);
        tweet.style.display = "";
        return;
      }
      applyHiddenState(tweet);
      return;
    case "fade":
      if (!isMatch) {
        clearPlaceholder(tweet);
        tweet.style.display = "";
        return;
      }
      clearPlaceholder(tweet);
      tweet.dataset.miladyShrinkifierEffect = "fade";
      tweet.style.display = "";
      return;
    case "debug":
      clearPlaceholder(tweet);
      applyDebugState(tweet);
      tweet.style.display = "";
      return;
    case "off":
    default:
      clearPlaceholder(tweet);
      tweet.style.display = "";
  }
}

function clearEffects(tweet: HTMLElement): void {
  clearVisualState(tweet);
  delete tweet.dataset.miladyShrinkifierDebug;
  clearPlaceholder(tweet);
  tweet.style.display = "";
}

function clearVisualState(tweet: HTMLElement): void {
  delete tweet.dataset.miladyShrinkifierEffect;
}

function applyDebugState(tweet: HTMLElement): void {
  if (tweet.dataset.miladyShrinkifierWhitelisted === "true") {
    tweet.dataset.miladyShrinkifierEffect = "debug-exempt";
    return;
  }

  if (tweet.dataset.miladyShrinkifierState === "match") {
    tweet.dataset.miladyShrinkifierEffect = "debug-match";
    return;
  }

  tweet.dataset.miladyShrinkifierEffect = "debug-miss";
}

function applyHiddenState(tweet: HTMLElement): void {
  let placeholder = placeholders.get(tweet);
  if (!placeholder) {
    placeholder = document.createElement("div");
    placeholder.className = "milady-shrinkifier-placeholder";
    const label = document.createElement("span");
    label.textContent = "Milady post hidden";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Show";
    button.addEventListener("click", () => {
      const normalizedUrl = processed.get(tweet);
      if (normalizedUrl) {
        revealed.set(tweet, normalizedUrl);
      }
      tweet.style.display = "";
      placeholder?.remove();
      placeholders.delete(tweet);
    });
    placeholder.append(label, button);
    placeholders.set(tweet, placeholder);
  }

  if (!placeholder.isConnected) {
    tweet.insertAdjacentElement("beforebegin", placeholder);
  }

  tweet.style.display = "none";
}

function clearPlaceholder(tweet: HTMLElement): void {
  const placeholder = placeholders.get(tweet);
  if (placeholder) {
    placeholder.remove();
    placeholders.delete(tweet);
  }
}

function setWhitelistState(tweet: HTMLElement, whitelisted: boolean): void {
  if (whitelisted) {
    tweet.dataset.miladyShrinkifierWhitelisted = "true";
    return;
  }

  delete tweet.dataset.miladyShrinkifierWhitelisted;
}

function injectStyles(): void {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    [data-milady-shrinkifier-effect="fade"] {
      opacity: 0.5;
    }

    [data-milady-shrinkifier-effect="debug-match"] {
      position: relative !important;
    }

    [data-milady-shrinkifier-effect="debug-miss"] {
      position: relative !important;
    }

    [data-milady-shrinkifier-effect="debug-exempt"] {
      position: relative !important;
    }

    [data-milady-shrinkifier-effect="debug-match"]::after,
    [data-milady-shrinkifier-effect="debug-miss"]::after,
    [data-milady-shrinkifier-effect="debug-exempt"]::after {
      content: "";
      position: absolute;
      inset: 0;
      border: 2px solid transparent;
      border-radius: 0 !important;
      pointer-events: none;
      z-index: 2147483647;
    }

    [data-milady-shrinkifier-effect="debug-match"]::before,
    [data-milady-shrinkifier-effect="debug-miss"]::before,
    [data-milady-shrinkifier-effect="debug-exempt"]::before {
      content: attr(data-milady-shrinkifier-debug);
      position: absolute;
      top: 6px;
      right: 6px;
      z-index: 2147483647;
      padding: 2px 6px;
      background: rgba(15, 20, 25, 0.92);
      color: rgb(255, 255, 255);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 11px;
      font-weight: 600;
      line-height: 1.4;
      pointer-events: none;
      border-radius: 0;
    }

    [data-milady-shrinkifier-effect="debug-match"]::after {
      border-color: rgba(231, 76, 60, 0.95);
    }

    [data-milady-shrinkifier-effect="debug-miss"]::after {
      border-color: rgba(46, 204, 113, 0.75);
    }

    [data-milady-shrinkifier-effect="debug-exempt"]::after {
      border-color: rgba(52, 152, 219, 0.85);
    }

    .milady-shrinkifier-placeholder {
      display: flex;
      align-items: center;
      gap: 12px;
      box-sizing: border-box;
      min-height: 52px;
      padding: 12px 16px;
      margin: 0;
      border-bottom: 1px solid rgb(239, 243, 244);
      background: rgb(255, 255, 255);
      color: rgb(83, 100, 113);
      font-family: TwitterChirp, -apple-system, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 15px;
      font-weight: 400;
      line-height: 20px;
    }

    .milady-shrinkifier-placeholder button {
      border: 0;
      padding: 0;
      background: transparent;
      color: rgb(29, 155, 240);
      font: inherit;
      cursor: pointer;
    }

    .milady-shrinkifier-placeholder button:hover {
      text-decoration: underline;
    }
  `;
  document.head.append(style);
}

function observeStorage(): void {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "sync" && (changes.mode || changes.whitelistHandles)) {
      const previousMode = settings.mode;
      const nextMode = changes.mode?.newValue;
      settings = {
        mode: isFilterMode(nextMode) ? nextMode : settings.mode,
        whitelistHandles: normalizeWhitelistHandles(
          changes.whitelistHandles?.newValue ?? settings.whitelistHandles,
        ),
      };
      if (changes.mode && settings.mode !== previousMode) {
        reapplyModeToVisibleTweets();
      }
      scheduleProcessVisibleTweets();
    }

    if (area === "local" && changes.stats) {
      stats = normalizeStats(changes.stats.newValue);
    }

    if (area === "local" && changes.matchedAccounts) {
      matchedAccounts = normalizeMatchedAccounts(changes.matchedAccounts.newValue);
    }

    if (area === "local" && changes.collectedAvatars) {
      collectedAvatars = normalizeCollectedAvatars(changes.collectedAvatars.newValue);
    }
  });
}

async function loadModelMetadata(): Promise<ResolvedModel> {
  if (!modelMetadataPromise) {
    modelMetadataPromise = resolveModel(CLASSIFIER_MODEL_METADATA_URL, CLASSIFIER_MODEL_URL);
  }
  return modelMetadataPromise;
}

async function resolveModel(
  metadataUrl: string,
  modelUrl: string,
): Promise<ResolvedModel> {
  const response = await fetch(chrome.runtime.getURL(metadataUrl));
  if (!response.ok) {
    throw new Error(`Failed to load model metadata: ${response.status}`);
  }

  const metadata = await response.json() as ModelMetadata;
  return {
    metadata,
    modelUrl: chrome.runtime.getURL(modelUrl),
    config: resolveRuntimeModelConfig(metadata),
  };
}

async function getWorker(resolvedModel: ResolvedModel): Promise<Worker> {
  if (workerPromise) {
    return workerPromise;
  }

  workerPromise = Promise.resolve().then(() => {
    const bootstrapUrl = URL.createObjectURL(
      new Blob([`importScripts(${JSON.stringify(chrome.runtime.getURL("worker.js"))});`], {
        type: "text/javascript",
      }),
    );
    const worker = new Worker(bootstrapUrl);
    URL.revokeObjectURL(bootstrapUrl);
    worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const pending = pendingWorker.get(event.data.id);
      if (!pending) {
        return;
      }
      pendingWorker.delete(event.data.id);
      if (event.data.error) {
        pending.reject(new Error(event.data.error));
        return;
      }
      if (typeof event.data.score !== "number") {
        pending.reject(new Error("Worker returned no score"));
        return;
      }
      pending.resolve(event.data.score);
    });
    worker.postMessage({
      modelUrl: resolvedModel.modelUrl,
      wasmPath: chrome.runtime.getURL("ort/"),
      positiveIndex: resolvedModel.config.positiveIndex,
    });
    return worker;
  });

  return workerPromise;
}

async function scoreWithOnnx(
  resolvedModel: ResolvedModel,
  variants: Array<{ modelTensor: Float32Array; modelShape: [1, number, number, number] }>,
  seed: string,
): Promise<number> {
  const worker = await getWorker(resolvedModel);
  const scores = await Promise.all(
    variants.map(
      ({ modelTensor, modelShape }, index) =>
        new Promise<number>((resolve, reject) => {
          const id = `${seed}:${index}:${crypto.randomUUID()}`;
          pendingWorker.set(id, { resolve, reject });
          const payload: WorkerRequest = {
            id,
            tensor: modelTensor,
            shape: modelShape,
          };
          worker.postMessage(payload, [modelTensor.buffer]);
        }),
    ),
  );
  return Math.max(...scores);
}

function isFilterMode(value: unknown): value is ExtensionSettings["mode"] {
  return value === "off" || value === "hide" || value === "fade" || value === "debug";
}

function incrementMatchStats(result: DetectionResult): void {
  incrementStat("postsMatched");
  if (result.source === "onnx") {
    incrementStat("modelMatches");
  }
  if (!stats) {
    return;
  }
  stats.lastMatchAt = new Date().toISOString();
  scheduleLocalStateWrite();
}

function incrementStat(key: keyof Omit<DetectionStats, "lastMatchAt">): void {
  if (!stats) {
    return;
  }
  stats[key] += 1;
  scheduleLocalStateWrite();
}

function recordMatchedAccount(handle: string, displayName: string | null): void {
  if (!matchedAccounts) {
    return;
  }

  const existing = matchedAccounts[handle];
  matchedAccounts[handle] = {
    handle,
    displayName: displayName ?? existing?.displayName ?? null,
    postsMatched: (existing?.postsMatched ?? 0) + 1,
    lastMatchedAt: new Date().toISOString(),
  };
  scheduleLocalStateWrite();
}

function recordCollectedAvatar(input: {
  normalizedUrl: string;
  originalUrl: string;
  author: { handle: string; displayName: string | null } | null;
  whitelisted: boolean;
  exampleTweetUrl: string | null;
  exampleNotificationUrl: string | null;
  sourceSurface: string;
  result?: DetectionResult;
}): void {
  if (!collectedAvatars) {
    return;
  }

  const existing = collectedAvatars[input.normalizedUrl];
  const now = new Date().toISOString();
  collectedAvatars[input.normalizedUrl] = {
    normalizedUrl: input.normalizedUrl,
    originalUrl: input.originalUrl || existing?.originalUrl || input.normalizedUrl,
    handles: mergeUniqueStrings(existing?.handles, input.author?.handle ?? null, true),
    displayNames: mergeUniqueStrings(existing?.displayNames, input.author?.displayName ?? null, false),
    sourceSurfaces: mergeUniqueStrings(existing?.sourceSurfaces, input.sourceSurface, false),
    seenCount: (existing?.seenCount ?? 0) + 1,
    firstSeenAt: existing?.firstSeenAt ?? now,
    lastSeenAt: now,
    exampleProfileUrl:
      existing?.exampleProfileUrl ?? (input.author ? toAbsoluteUrl(`/${input.author.handle}`) : null),
    exampleNotificationUrl: existing?.exampleNotificationUrl ?? input.exampleNotificationUrl,
    exampleTweetUrl: existing?.exampleTweetUrl ?? input.exampleTweetUrl,
    whitelisted: input.whitelisted || existing?.whitelisted === true,
  };
  scheduleLocalStateWrite();
}

function scheduleLocalStateWrite(): void {
  if (localStateWriteScheduled || !stats || !matchedAccounts || !collectedAvatars) {
    return;
  }
  localStateWriteScheduled = true;
  window.setTimeout(async () => {
    localStateWriteScheduled = false;
    if (!stats || !matchedAccounts || !collectedAvatars) {
      return;
    }
    await Promise.all([
      saveStats(stats),
      saveMatchedAccounts(matchedAccounts),
      saveCollectedAvatars(collectedAvatars),
    ]);
  }, 250);
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

function normalizeHandle(value: string | null | undefined): string {
  return (value ?? "").trim().replace(/^\/+/, "").replace(/^@+/, "").toLowerCase();
}

function formatProbabilityDebugLabel(score: number, threshold: number): string {
  return `p${score.toFixed(3)} t${threshold.toFixed(3)}`;
}

function findTweetUrl(tweet: HTMLElement): string | null {
  const link = tweet.querySelector<HTMLAnchorElement>('a[href*="/status/"]');
  return toAbsoluteUrl(link?.getAttribute("href"));
}

function collectNotificationAvatarEntries(notification: HTMLElement): Array<{
  handle: string;
  normalizedUrl: string;
  originalUrl: string;
}> {
  const results = new Map<string, { handle: string; normalizedUrl: string; originalUrl: string }>();

  for (const container of Array.from(notification.querySelectorAll<HTMLElement>('[data-testid^="UserAvatar-Container-"]'))) {
    const testId = container.dataset.testid ?? "";
    const handle = normalizeHandle(testId.replace(/^UserAvatar-Container-/, ""));
    const image = container.querySelector<HTMLImageElement>('img[src*="profile_images"]');
    const source = image?.currentSrc || image?.src;
    if (!handle || !source) {
      continue;
    }

    const normalizedUrl = normalizeProfileImageUrl(source);
    results.set(`${handle}:${normalizedUrl}`, {
      handle,
      normalizedUrl,
      originalUrl: source,
    });
  }

  return Array.from(results.values());
}

function toAbsoluteUrl(value: string | null | undefined): string | null {
  if (!value) {
    return null;
  }

  try {
    return new URL(value, window.location.origin).toString();
  } catch {
    return null;
  }
}

function extractDisplayName(userName: HTMLElement): string | null {
  for (const span of Array.from(userName.querySelectorAll("span"))) {
    const text = span.textContent?.trim();
    if (!text || text.startsWith("@") || text === "·") {
      continue;
    }
    return text;
  }

  return null;
}
