import "./styles.css";

import { QueryClient, QueryClientProvider, createMutation, createQuery, useQueryClient } from "@tanstack/solid-query";
import { createVirtualizer } from "@tanstack/solid-virtual";
import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { render } from "solid-js/web";

type ReviewLabel = "milady" | "not_milady" | "unclear";
type QueueName =
  | "unlabeled"
  | "human_vs_model"
  | "whitelisted"
  | "high_seen_count"
  | "notification_group"
  | "uncertain_unlabeled"
  | "high_score_unlabeled"
  | "high_score_false_positive";
type GridSource = "queue" | "labeled";
type LabeledGridFilter = "all" | ReviewLabel;

interface ReviewItem {
  sha256: string;
  label: ReviewLabel | null;
  localPath: string;
  byteSize: number | null;
  width: number | null;
  height: number | null;
  handles: string[];
  displayNames: string[];
  sourceSurfaces: string[];
  seenCount: number;
  whitelisted: boolean;
  maxModelScore: number | null;
  latestModelPredictedLabel: ReviewLabel | null;
  latestModelRunId: string | null;
  latestModelThreshold: number | null;
  latestModelDistanceToThreshold: number | null;
  disagreementFlags: string[];
  labeledAt: string | null;
  exampleProfileUrl: string | null;
  exampleNotificationUrl: string | null;
  exampleTweetUrl: string | null;
  lastSeenAt: string | null;
  imageUrlCount: number;
}

interface SummaryPayload {
  catalogPath: string;
  selectedRunId: string | null;
  availableRunIds: string[];
  totalImages: number;
  queueCounts: Record<QueueName, number>;
  labelCounts: Record<ReviewLabel, number>;
  unlabeled: number;
  canUndo: boolean;
}

interface QueuePayload {
  queue: QueueName;
  index: number;
  total: number;
  item: ReviewItem | null;
}

interface BatchPayload {
  queue: QueueName;
  total: number;
  offset: number;
  items: ReviewItem[];
}

interface ItemPayload {
  item: ReviewItem;
}

interface HistoryEntry {
  eventId: number;
  sha256: string;
  createdAt: string;
  newLabel: ReviewLabel;
  previousLabel: ReviewLabel | null;
  item: ReviewItem | null;
}

interface HistoryPayload {
  history: HistoryEntry[];
}

interface GridPayload {
  items: ReviewItem[];
  total: number;
}

interface BatchAssignment {
  item: ReviewItem;
  assignedLabel: ReviewLabel;
}

type GroupLabel = ReviewLabel | "unlabeled";

interface GridGroup {
  key: GroupLabel;
  title: string;
  items: ReviewItem[];
}

type VirtualGridRow =
  | { kind: "header"; key: string; title: string }
  | { kind: "items"; key: string; items: ReviewItem[] };

const queueLabels: Record<QueueName, string> = {
  unlabeled: "Unlabeled",
  human_vs_model: "Human vs model",
  whitelisted: "Whitelisted",
  high_seen_count: "High seen count",
  notification_group: "Notification group",
  uncertain_unlabeled: "Uncertain unlabeled",
  high_score_unlabeled: "High-score unlabeled",
  high_score_false_positive: "High-score false positives",
};

const labeledGridLabels: Record<LabeledGridFilter, string> = {
  all: "All",
  milady: "Milady",
  not_milady: "Not Milady",
  unclear: "Unclear",
};

const labelDisplay: Record<ReviewLabel, string> = {
  milady: "Milady",
  not_milady: "Not Milady",
  unclear: "Unclear",
};

const batchTileKeys = [7, 8, 9, 4, 5, 6, 1, 2, 3] as const;
const numpadIndexMap: Record<string, number> = {
  Numpad7: 0,
  Numpad8: 1,
  Numpad9: 2,
  Numpad4: 3,
  Numpad5: 4,
  Numpad6: 5,
  Numpad1: 6,
  Numpad2: 7,
  Numpad3: 8,
};
const batchLabelOrder: ReviewLabel[] = ["not_milady", "milady", "unclear"];
const gridOrder: GroupLabel[] = ["unlabeled", "milady", "not_milady", "unclear"];
const gridGroupLabels: Record<GroupLabel, string> = {
  unlabeled: "Unlabeled",
  milady: "Milady",
  not_milady: "Not Milady",
  unclear: "Unclear",
};
const preferredQueueOrder: QueueName[] = [
  "uncertain_unlabeled",
  "high_score_false_positive",
  "human_vs_model",
  "high_score_unlabeled",
  "unlabeled",
];
const queueGroups: Array<{ label: string; queues: QueueName[] }> = [
  {
    label: "Labeling",
    queues: ["uncertain_unlabeled", "high_score_false_positive", "high_score_unlabeled", "unlabeled"],
  },
  {
    label: "Audit",
    queues: ["human_vs_model", "whitelisted", "notification_group", "high_seen_count"],
  },
];
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
  },
});

async function fetchJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

async function postJson<T>(input: string, body?: unknown): Promise<T> {
  return fetchJson<T>(input, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function imageUrl(sha256: string): string {
  return `/api/image/${encodeURIComponent(sha256)}`;
}

function withRunId(path: string, runId: string | null): string {
  if (!runId) {
    return path;
  }
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}run_id=${encodeURIComponent(runId)}`;
}

function formatScore(value: number | null): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(3);
}

function scoreBadgeText(item: ReviewItem): string | null {
  if (item.maxModelScore == null || Number.isNaN(item.maxModelScore)) {
    return null;
  }
  return `p ${formatScore(item.maxModelScore)}`;
}

function shortLabel(label: ReviewLabel): string {
  if (label === "milady") return "M";
  if (label === "unclear") return "U";
  return "N";
}

function initialBatchLabel(item: ReviewItem): ReviewLabel {
  if (item.label) {
    return item.label;
  }
  if (item.latestModelPredictedLabel) {
    return item.latestModelPredictedLabel;
  }
  return "not_milady";
}

function renderStatusPills(item: ReviewItem) {
  const pills = [];
  if (item.latestModelPredictedLabel) {
    pills.push({
      text: `model ${item.latestModelPredictedLabel} ${formatScore(item.maxModelScore)}`,
      tone: item.latestModelPredictedLabel === "milady" ? "warn" : "good",
    });
  } else {
    pills.push({ text: "model unscored", tone: "" });
  }
  if (item.label) {
    pills.push({
      text: `human ${item.label}`,
      tone: item.label === "not_milady" ? "good" : "warn",
    });
  }
  for (const flag of item.disagreementFlags) {
    pills.push({ text: flag.replaceAll("_", " "), tone: "bad" });
  }
  return pills;
}

function metadataRows(item: ReviewItem): Array<{ label: string; value: string | { href: string; text: string } }> {
  return [
    { label: "sha256", value: item.sha256 },
    { label: "label", value: item.label ?? "unlabeled" },
    { label: "labeled at", value: item.labeledAt ?? "n/a" },
    { label: "handles", value: item.handles.join(", ") || "none" },
    { label: "display names", value: item.displayNames.join(", ") || "none" },
    { label: "seen count", value: String(item.seenCount) },
    { label: "source surfaces", value: item.sourceSurfaces.join(", ") || "none" },
    {
      label: "model",
      value: item.latestModelPredictedLabel
        ? `${item.latestModelPredictedLabel} (${formatScore(item.maxModelScore)})`
        : "unscored",
    },
    {
      label: "threshold",
      value:
        item.latestModelThreshold != null
          ? `${formatScore(item.latestModelThreshold)} (Δ ${formatScore(item.latestModelDistanceToThreshold)})`
          : "n/a",
    },
    { label: "model run", value: item.latestModelRunId ?? "n/a" },
    { label: "whitelisted", value: item.whitelisted ? "yes" : "no" },
    {
      label: "profile",
      value: item.exampleProfileUrl ? { href: item.exampleProfileUrl, text: item.exampleProfileUrl } : "n/a",
    },
    {
      label: "tweet",
      value: item.exampleTweetUrl ? { href: item.exampleTweetUrl, text: item.exampleTweetUrl } : "n/a",
    },
    {
      label: "notification",
      value: item.exampleNotificationUrl ? { href: item.exampleNotificationUrl, text: item.exampleNotificationUrl } : "n/a",
    },
  ];
}

function App() {
  const queryClient = useQueryClient();
  const [queue, setQueue] = createSignal<QueueName>("unlabeled");
  const [index, setIndex] = createSignal(0);
  const [selectedSha, setSelectedSha] = createSignal<string | null>(null);
  const [activeView, setActiveView] = createSignal<"individual" | "batch">("individual");
  const [gridSource, setGridSource] = createSignal<GridSource>("queue");
  const [gridFilter, setGridFilter] = createSignal<string>("unlabeled");
  const [selectedBatchIndex, setSelectedBatchIndex] = createSignal(0);
  const [batchAssignments, setBatchAssignments] = createSignal<BatchAssignment[]>([]);
  const [batchOffset, setBatchOffset] = createSignal(0);
  const [gridWidth, setGridWidth] = createSignal(0);
  const [selectedRunId, setSelectedRunId] = createSignal<string | null>(null);
  let gridScrollRef: HTMLDivElement | undefined;
  let gridResizeObserver: ResizeObserver | undefined;

  const summaryQuery = createQuery(() => ({
    queryKey: ["review", "summary", selectedRunId()],
    queryFn: () => fetchJson<SummaryPayload>(withRunId("/api/summary", selectedRunId())),
  }));

  createEffect(() => {
    const summary = summaryQuery.data;
    if (!summary) {
      return;
    }
    if (selectedRunId() !== summary.selectedRunId) {
      setSelectedRunId(summary.selectedRunId);
    }

    const preferredQueue = preferredQueueOrder.find((candidate) => (summary.queueCounts[candidate] ?? 0) > 0) ?? "unlabeled";

    if (!(queue() in summary.queueCounts)) {
      setQueue(preferredQueue);
      setIndex(0);
    } else if (queue() === "unlabeled" && preferredQueue !== "unlabeled" && (summary.queueCounts.unlabeled ?? 0) === 0) {
      setQueue(preferredQueue);
      setIndex(0);
    }

    if (gridSource() === "queue") {
      if (!(gridFilter() in summary.queueCounts)) {
        setGridFilter(queue());
      } else if (gridFilter() === "unlabeled" && preferredQueue !== "unlabeled" && (summary.queueCounts.unlabeled ?? 0) === 0) {
        setGridFilter(preferredQueue);
      }
    } else if (!(gridFilter() in labeledGridLabels)) {
      setGridFilter("all");
    }
  });

  const selectedItemQuery = createQuery(() => ({
    queryKey: ["review", "item", selectedRunId(), selectedSha()],
    enabled: selectedSha() !== null,
    queryFn: () => fetchJson<ItemPayload>(withRunId(`/api/item/${encodeURIComponent(selectedSha() ?? "")}`, selectedRunId())),
  }));

  const queueQuery = createQuery(() => ({
    queryKey: ["review", "queue", selectedRunId(), queue(), index()],
    enabled: selectedSha() === null,
    queryFn: () => fetchJson<QueuePayload>(withRunId(`/api/queue?queue=${encodeURIComponent(queue())}&index=${index()}`, selectedRunId())),
  }));

  createEffect(() => {
    const payload = queueQuery.data;
    if (selectedSha() !== null || !payload) {
      return;
    }
    if (payload.index !== index()) {
      setIndex(payload.index);
    }
  });

  const currentItem = createMemo<ReviewItem | null>(() => {
    if (selectedSha()) {
      return selectedItemQuery.data?.item ?? null;
    }
    return queueQuery.data?.item ?? null;
  });

  const currentHeading = createMemo(() => {
    if (selectedSha()) {
      return `selected ${selectedSha()!.slice(0, 8)}`;
    }
    const payload = queueQuery.data;
    if (!payload?.item) {
      return "Queue empty";
    }
    return `${queueLabels[payload.queue]} ${payload.index + 1}/${payload.total}`;
  });

  const historyQuery = createQuery(() => ({
    queryKey: ["review", "history", selectedRunId()],
    queryFn: () => fetchJson<HistoryPayload>(withRunId("/api/history", selectedRunId())),
  }));

  const batchQuery = createQuery(() => ({
    queryKey: ["review", "batch", selectedRunId(), queue(), batchOffset()],
    enabled: activeView() === "batch",
    queryFn: () =>
      fetchJson<BatchPayload>(
        withRunId(`/api/batch?queue=${encodeURIComponent(queue())}&limit=9&offset=${batchOffset()}`, selectedRunId()),
      ),
  }));

  createEffect(() => {
    const payload = batchQuery.data;
    if (!payload) {
      return;
    }
    if (payload.offset !== batchOffset()) {
      setBatchOffset(payload.offset);
    }
    setSelectedBatchIndex(0);
    setBatchAssignments(
      payload.items.map((item) => ({
        item,
        assignedLabel: initialBatchLabel(item),
      })),
    );
  });

  const effectiveGridFilter = createMemo(() => {
    if (gridSource() === "queue") {
      const value = gridFilter();
      return value in queueLabels ? value : queue();
    }
    const value = gridFilter();
    return value in labeledGridLabels ? value : "all";
  });

  const gridQuery = createQuery(() => ({
    queryKey: ["review", "grid", selectedRunId(), gridSource(), effectiveGridFilter()],
    queryFn: () =>
      gridSource() === "queue"
        ? fetchJson<GridPayload>(withRunId(`/api/queue-grid?queue=${encodeURIComponent(effectiveGridFilter())}`, selectedRunId()))
        : fetchJson<GridPayload>(withRunId(`/api/labeled-grid?filter_name=${encodeURIComponent(effectiveGridFilter())}`, selectedRunId())),
  }));

  const groupedGridItems = createMemo(() => {
    const items = gridQuery.data?.items ?? [];
    return gridOrder
      .map((label) => {
        const groupedItems = items.filter((item) => (item.label ?? "unlabeled") === label);
        return {
          key: label,
          title: `${gridGroupLabels[label]} (${groupedItems.length})`,
          items: groupedItems,
        } satisfies GridGroup;
      })
      .filter((group) => group.items.length > 0);
  });

  const gridColumns = createMemo(() => {
    const width = gridWidth();
    if (width <= 0) {
      return 1;
    }
    return Math.max(1, Math.floor((width + 10) / 106));
  });

  const gridItemRowHeight = createMemo(() => {
    const width = gridWidth();
    if (width <= 0) {
      return 128;
    }
    const columns = gridColumns();
    const thumbWidth = (width - Math.max(0, columns - 1) * 10) / columns;
    return Math.ceil(thumbWidth + 32);
  });

  const virtualGridRows = createMemo<VirtualGridRow[]>(() => {
    const columns = gridColumns();
    const rows: VirtualGridRow[] = [];
    for (const group of groupedGridItems()) {
      rows.push({ kind: "header", key: `${group.key}-header`, title: group.title });
      for (let offset = 0; offset < group.items.length; offset += columns) {
        rows.push({
          kind: "items",
          key: `${group.key}-${offset}`,
          items: group.items.slice(offset, offset + columns),
        });
      }
    }
    return rows;
  });

  const gridVirtualizer = createVirtualizer<HTMLDivElement, HTMLDivElement>({
    get count() {
      return virtualGridRows().length;
    },
    getScrollElement: () => gridScrollRef ?? null,
    estimateSize: (index) => {
      const row = virtualGridRows()[index];
      return row?.kind === "header" ? 28 : gridItemRowHeight();
    },
    overscan: 8,
  });

  createEffect(() => {
    virtualGridRows();
    gridItemRowHeight();
    gridVirtualizer.measure();
  });

  async function invalidateAll() {
    await queryClient.invalidateQueries({ queryKey: ["review"] });
  }

  const labelMutation = createMutation(() => ({
    mutationFn: (payload: { sha256: string; label: ReviewLabel }) =>
      postJson<{ ok: true; changed: boolean }>("/api/label", payload),
  }));

  const batchMutation = createMutation(() => ({
    mutationFn: (payload: { items: Array<{ sha256: string; label: ReviewLabel }> }) =>
      postJson<{ ok: true; batchId: string; count: number }>("/api/batch-label", payload),
  }));

  const undoMutation = createMutation(() => ({
    mutationFn: () => postJson<{ undoneSha256: string | null }>("/api/undo"),
  }));

  async function handleLabel(label: ReviewLabel) {
    const item = currentItem();
    if (!item) {
      return;
    }
    const wasQueueItem = selectedSha() === null;
    const payload = await labelMutation.mutateAsync({ sha256: item.sha256, label });
    setSelectedSha(null);
    if (wasQueueItem && payload.changed) {
      setIndex((value) => value + 1);
    }
    await invalidateAll();
  }

  async function handleSkip() {
    setSelectedSha(null);
    setIndex((value) => value + 1);
  }

  function cycleBatchLabel(tileIndex: number) {
    setBatchAssignments((current) => {
      const next = current.slice();
      const entry = next[tileIndex];
      if (!entry) {
        return current;
      }
      const currentIndex = batchLabelOrder.indexOf(entry.assignedLabel);
      next[tileIndex] = {
        ...entry,
        assignedLabel: batchLabelOrder[(currentIndex + 1) % batchLabelOrder.length],
      };
      return next;
    });
  }

  async function handleCommitBatch() {
    const items = batchAssignments();
    if (items.length === 0) {
      return;
    }
    const payload = await batchMutation.mutateAsync({
      items: items.map((entry) => ({ sha256: entry.item.sha256, label: entry.assignedLabel })),
    });
    if (payload.count === 0) {
      setBatchOffset((value) => value + items.length);
      await batchQuery.refetch();
      return;
    }
    setSelectedSha(null);
    await invalidateAll();
  }

  function moveBatchSelection(delta: number) {
    setSelectedBatchIndex((value) => {
      const total = batchAssignments().length;
      if (total === 0) {
        return 0;
      }
      return Math.max(0, Math.min(total - 1, value + delta));
    });
  }

  async function handleUndo() {
    const wasQueueItem = selectedSha() === null;
    const payload = await undoMutation.mutateAsync();
    setSelectedSha(payload.undoneSha256 ?? null);
    if (wasQueueItem) {
      setIndex((value) => Math.max(0, value - 1));
    }
    await invalidateAll();
  }

  function selectGridItem(sha256: string) {
    setSelectedSha(sha256);
    setActiveView("individual");
  }

  function bindGridScrollRef(element: HTMLDivElement) {
    gridScrollRef = element;
    if (!gridResizeObserver) {
      return;
    }
    setGridWidth(element.clientWidth);
    gridResizeObserver.disconnect();
    gridResizeObserver.observe(element);
    queueMicrotask(() => gridVirtualizer.measure());
  }

  onMount(() => {
    gridResizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      setGridWidth(entry.contentRect.width);
      queueMicrotask(() => gridVirtualizer.measure());
    });

    if (gridScrollRef) {
      setGridWidth(gridScrollRef.clientWidth);
      gridResizeObserver.observe(gridScrollRef);
    }

    const handleKeyDown = async (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) {
        return;
      }

      if (event.key.toLowerCase() === "z") {
        event.preventDefault();
        await handleUndo();
        return;
      }

      if (activeView() === "individual") {
        if (event.key === "1") {
          event.preventDefault();
          await handleLabel("milady");
        }
        if (event.key === "2") {
          event.preventDefault();
          await handleLabel("not_milady");
        }
        if (event.key === "3") {
          event.preventDefault();
          await handleLabel("unclear");
        }
        if (event.key.toLowerCase() === "x") {
          event.preventDefault();
          await handleSkip();
        }
        return;
      }

      const numpadIndex = numpadIndexMap[event.code];
      if (numpadIndex != null) {
        event.preventDefault();
        setSelectedBatchIndex(numpadIndex);
        cycleBatchLabel(numpadIndex);
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        moveBatchSelection(-1);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        moveBatchSelection(1);
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        moveBatchSelection(-3);
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        moveBatchSelection(3);
        return;
      }
      if (event.key === "Enter" || event.code === "NumpadEnter") {
        event.preventDefault();
        await handleCommitBatch();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    onCleanup(() => {
      window.removeEventListener("keydown", handleKeyDown);
      gridResizeObserver?.disconnect();
    });
  });

  return (
    <div class="app">
      <div class="layout">
        <aside class="panel sidebar-panel">
          <h2>Queues</h2>
          <label>
            <span>Run</span>
            <select
              value={selectedRunId() ?? ""}
              onInput={(event) => {
                setSelectedRunId(event.currentTarget.value || null);
                setSelectedSha(null);
                setIndex(0);
                setBatchOffset(0);
              }}
            >
              <Show when={summaryQuery.data} fallback={<option>Loading…</option>}>
                {(summary) => (
                  <For each={summary().availableRunIds}>
                    {(runId) => <option value={runId}>{runId}</option>}
                  </For>
                )}
              </Show>
            </select>
          </label>
          <label>
            <span>Queue</span>
            <select
              value={queue()}
              onInput={(event) => {
                setQueue(event.currentTarget.value as QueueName);
                setSelectedSha(null);
                setIndex(0);
                setBatchOffset(0);
                if (gridSource() === "queue") {
                  setGridFilter(event.currentTarget.value);
                }
              }}
            >
              <Show when={summaryQuery.data} fallback={<option>Loading…</option>}>
                {(summary) => (
                  <For each={queueGroups}>
                    {(group) => (
                      <optgroup label={group.label}>
                        <For each={group.queues}>
                          {(queueName) => (
                            <option value={queueName}>{`${queueLabels[queueName]} (${summary().queueCounts[queueName] ?? 0})`}</option>
                          )}
                        </For>
                      </optgroup>
                    )}
                  </For>
                )}
              </Show>
            </select>
          </label>
          <p class="summary-copy">
            <Show when={summaryQuery.data} fallback="Loading summary…">
              {(summary) => `${summary().totalImages} images, ${summary().unlabeled} unlabeled, run ${summary().selectedRunId ?? "unscored"}`}
            </Show>
          </p>
          <div class="actions">
            <button type="button" disabled={!summaryQuery.data?.canUndo || undoMutation.isPending} onClick={() => void handleUndo()}>
              Undo last label
            </button>
          </div>
          <div class="hint">Hotkeys: 1=milady, 2=not_milady, 3=unclear, x=skip, z=undo</div>
        </aside>

        <section class="panel workspace-panel">
          <div class="workspace-header">
            <div class="tab-bar">
              <button type="button" class="tab-button" data-active={String(activeView() === "individual")} onClick={() => setActiveView("individual")}>
                Individual
              </button>
              <button type="button" class="tab-button" data-active={String(activeView() === "batch")} onClick={() => setActiveView("batch")}>
                Batch
              </button>
            </div>
          </div>

          <div class="view" hidden={activeView() !== "individual"}>
            <div class="individual-layout">
              <section class="panel">
                <h2>{currentHeading()}</h2>
                <Show when={currentItem()} fallback={<p class="loading-copy">Queue empty.</p>}>
                  {(item) => (
                    <>
                      <div class="status-strip">
                        <For each={renderStatusPills(item())}>
                          {(pill) => (
                            <span class="pill" data-tone={pill.tone}>
                              {pill.text}
                            </span>
                          )}
                        </For>
                      </div>
                      <img class="preview-image" src={imageUrl(item().sha256)} alt={item().sha256} />
                      <div class="actions">
                        <button type="button" disabled={labelMutation.isPending} onClick={() => void handleLabel("milady")}>
                          1 Milady
                        </button>
                        <button type="button" disabled={labelMutation.isPending} onClick={() => void handleLabel("not_milady")}>
                          2 Not Milady
                        </button>
                        <button type="button" disabled={labelMutation.isPending} onClick={() => void handleLabel("unclear")}>
                          3 Unclear
                        </button>
                        <button type="button" onClick={() => void handleSkip()}>
                          Skip
                        </button>
                      </div>
                    </>
                  )}
                </Show>
              </section>

              <div class="individual-stack">
                <section class="panel">
                  <h2>Metadata</h2>
                  <Show when={currentItem()} fallback={<p class="empty-copy">Select an image to inspect it.</p>}>
                    {(item) => (
                      <dl class="metadata-grid">
                        <For each={metadataRows(item())}>
                          {(row) => (
                            <>
                              <dt>{row.label}</dt>
                              <dd>
                                {typeof row.value === "string" ? (
                                  row.value
                                ) : (
                                  <a href={row.value.href} target="_blank" rel="noreferrer">
                                    {row.value.text}
                                  </a>
                                )}
                              </dd>
                            </>
                          )}
                        </For>
                      </dl>
                    )}
                  </Show>
                </section>

                <section class="panel">
                  <div class="history-header">
                    <h2>Recent labels</h2>
                  </div>
                  <Show when={(historyQuery.data?.history.length ?? 0) > 0} fallback={<p class="empty-copy">No recent labels yet.</p>}>
                    <div class="history-grid">
                      <For each={historyQuery.data?.history ?? []}>
                        {(entry) => (
                          <Show when={entry.item}>
                            <button
                              type="button"
                              class="thumb-button"
                              data-selected={String(selectedSha() === entry.sha256)}
                              onClick={() => selectGridItem(entry.sha256)}
                            >
                              <img src={imageUrl(entry.sha256)} alt={entry.sha256} />
                              <span>{labelDisplay[entry.newLabel]}</span>
                            </button>
                          </Show>
                        )}
                      </For>
                    </div>
                  </Show>
                </section>
              </div>
            </div>
          </div>

          <div class="view" hidden={activeView() !== "batch"}>
            <section class="panel">
              <div class="grid-toolbar">
                <h2>Batch mode</h2>
                <div class="batch-actions">
                  <button type="button" onClick={() => void batchQuery.refetch()}>
                    Load batch
                  </button>
                  <button type="button" disabled={batchOffset() === 0} onClick={() => setBatchOffset((value) => Math.max(0, value - 9))}>
                    Prev batch
                  </button>
                  <button type="button" disabled={batchAssignments().length === 0} onClick={() => setBatchOffset((value) => value + 9)}>
                    Next batch
                  </button>
                  <button type="button" disabled={batchAssignments().length === 0 || batchMutation.isPending} onClick={() => void handleCommitBatch()}>
                    Enter: commit batch
                  </button>
                </div>
              </div>
              <div class="hint">Numpad cycles labels, arrow keys move selection, Enter commits or advances.</div>
              <div class="batch-panel">
                <Show when={batchAssignments().length > 0} fallback={<p class="empty-copy">No items in this queue.</p>}>
                  <div class="batch-grid">
                    <For each={batchAssignments()}>
                      {(entry, itemIndex) => (
                        <button
                          type="button"
                          class="batch-tile"
                          data-selected={String(itemIndex() === selectedBatchIndex())}
                          data-label={entry.assignedLabel}
                          onClick={() => {
                            setSelectedBatchIndex(itemIndex());
                            cycleBatchLabel(itemIndex());
                          }}
                        >
                          <img src={imageUrl(entry.item.sha256)} alt={entry.item.sha256} />
                          <div class="batch-caption">
                            <span>{batchTileKeys[itemIndex()]}</span>
                            <span class="batch-badge">{shortLabel(entry.assignedLabel)}</span>
                          </div>
                        </button>
                      )}
                    </For>
                  </div>
                </Show>
              </div>
            </section>
          </div>
        </section>

        <section class="panel full-width">
          <div class="grid-toolbar">
            <h2>Browse grid</h2>
            <div class="toolbar-inline">
              <label>
                <span>Source</span>
                <select
                  value={gridSource()}
                  onInput={(event) => {
                    const nextSource = event.currentTarget.value as GridSource;
                    setGridSource(nextSource);
                    setGridFilter(nextSource === "queue" ? queue() : "all");
                  }}
                >
                  <option value="queue">Current queue</option>
                  <option value="labeled">Labeled set</option>
                </select>
              </label>
              <label>
                <span>Filter</span>
                <select value={gridFilter()} onInput={(event) => setGridFilter(event.currentTarget.value)}>
                  <Show when={gridSource() === "queue"} fallback={
                    <For each={Object.entries(labeledGridLabels) as Array<[LabeledGridFilter, string]>}>
                      {([value, label]) => <option value={value}>{label}</option>}
                    </For>
                  }>
                    <Show when={summaryQuery.data}>
                      {(summary) => (
                        <For each={queueGroups}>
                          {(group) => (
                            <optgroup label={group.label}>
                              <For each={group.queues}>
                                {(queueName) => (
                                  <option value={queueName}>{`${queueLabels[queueName]} (${summary().queueCounts[queueName] ?? 0})`}</option>
                                )}
                              </For>
                            </optgroup>
                          )}
                        </For>
                      )}
                    </Show>
                  </Show>
                </select>
              </label>
            </div>
          </div>

          <Show when={(gridQuery.data?.items.length ?? 0) > 0} fallback={<p class="empty-copy">No images for this view.</p>}>
            <div
              ref={bindGridScrollRef}
              class="grid-scroll"
            >
              <div class="grid-virtual-space" style={{ height: `${gridVirtualizer.getTotalSize()}px` }}>
                <For each={gridVirtualizer.getVirtualItems()}>
                  {(virtualItem) => {
                    return (
                      <div
                        ref={(element) => gridVirtualizer.measureElement(element)}
                        class="grid-virtual-row"
                        style={{ transform: `translateY(${virtualItem.start}px)` }}
                      >
                        <Show when={virtualGridRows()[virtualItem.index]} keyed>
                          {(row) =>
                            row.kind === "header" ? (
                              <div class="grid-group-header">{row.title}</div>
                            ) : (
                              <div class="grid-row" style={{ "grid-template-columns": `repeat(${gridColumns()}, minmax(0, 1fr))` }}>
                                <For each={row.items}>
                                  {(item) => (
                                    <button
                                      type="button"
                                      class="thumb-button"
                                      data-selected={String(selectedSha() === item.sha256)}
                                      onClick={() => selectGridItem(item.sha256)}
                                    >
                                      <div class="thumb-image-frame">
                                        <img src={imageUrl(item.sha256)} alt={item.sha256} />
                                        <Show when={scoreBadgeText(item)}>
                                          {(scoreText) => <span class="thumb-score-badge">{scoreText()}</span>}
                                        </Show>
                                      </div>
                                      <span class="thumb-label">{item.label ? labelDisplay[item.label] : "unlabeled"}</span>
                                    </button>
                                  )}
                                </For>
                              </div>
                            )
                          }
                        </Show>
                      </div>
                    );
                  }}
                </For>
              </div>
            </div>
          </Show>
        </section>
      </div>
    </div>
  );
}

render(
  () => (
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  ),
  document.getElementById("root")!,
);
