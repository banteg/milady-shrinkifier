import "./styles.css";

import { QueryClient, QueryClientProvider, createMutation, createQuery, useQueryClient } from "@tanstack/solid-query";
import { createVirtualizer } from "@tanstack/solid-virtual";
import { For, Show, createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import { render } from "solid-js/web";

type ReviewLabel = "milady" | "not_milady" | "unclear";
type QueueName =
  | "needs_review"
  | "model_disagreements"
  | "exempted"
  | "high_impact"
  | "notifications"
  | "boundary_review"
  | "model_backlog"
  | "hard_negatives";
type GridSource = "queue" | "labeled";
type LabeledGridFilter = "all" | ReviewLabel;

interface ReviewItem {
  sha256: string;
  label: ReviewLabel | null;
  label_source: string | null;
  local_path: string;
  handles: string[];
  display_names: string[];
  source_surfaces: string[];
  seen_count: number;
  whitelisted: boolean;
  max_model_score: number | null;
  latest_model_predicted_label: ReviewLabel | null;
  latest_model_run_id: string | null;
  latest_model_threshold: number | null;
  latest_model_distance_to_threshold: number | null;
  disagreement_flags: string[];
  labeled_at: string | null;
  example_profile_url: string | null;
  example_notification_url: string | null;
  example_tweet_url: string | null;
  last_seen_at: string | null;
}

interface SummaryPayload {
  selected_run_id: string | null;
  available_run_ids: string[];
  total_images: number;
  queue_counts: Record<QueueName, number>;
  needs_review: number;
  can_undo: boolean;
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
  event_id: number;
  sha256: string;
  created_at: string;
  new_label: ReviewLabel;
  previous_label: ReviewLabel | null;
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
  needs_review: "Needs review",
  model_disagreements: "Model disagreements",
  exempted: "Exempted",
  high_impact: "High-impact",
  notifications: "Notifications",
  boundary_review: "Boundary review",
  model_backlog: "Model backlog",
  hard_negatives: "Hard negatives",
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
  unlabeled: "Unreviewed",
  milady: "Milady",
  not_milady: "Not Milady",
  unclear: "Unclear",
};
const preferredQueueOrder: QueueName[] = [
  "hard_negatives",
  "model_disagreements",
  "boundary_review",
  "notifications",
  "high_impact",
  "needs_review",
  "model_backlog",
  "exempted",
];
const queueGroups: Array<{ label: string; queues: QueueName[] }> = [
  {
    label: "Priority Review",
    queues: ["hard_negatives", "model_disagreements", "boundary_review", "notifications", "high_impact"],
  },
  {
    label: "Backlog",
    queues: ["needs_review", "model_backlog"],
  },
  {
    label: "Audit",
    queues: ["exempted"],
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
  if (item.max_model_score == null || Number.isNaN(item.max_model_score)) {
    return null;
  }
  return `p ${formatScore(item.max_model_score)}`;
}

function scoreBarPercent(value: number | null): number | null {
  if (value == null || Number.isNaN(value)) {
    return null;
  }
  return Math.max(0, Math.min(100, value * 100));
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
  if (item.latest_model_predicted_label) {
    return item.latest_model_predicted_label;
  }
  return "not_milady";
}

function renderStatusPills(item: ReviewItem) {
  const pills = [];
  if (item.latest_model_predicted_label) {
    pills.push({
      text: `model ${item.latest_model_predicted_label} ${formatScore(item.max_model_score)}`,
      tone: item.latest_model_predicted_label === "milady" ? "warn" : "good",
    });
  } else {
    pills.push({ text: "model unscored", tone: "" });
  }
  if (item.label) {
    pills.push({
      text: `${item.label_source ?? "n/a"} ${item.label}`,
      tone: item.label === "not_milady" ? "good" : "warn",
    });
  }
  for (const flag of item.disagreement_flags) {
    pills.push({ text: flag.replaceAll("_", " "), tone: "bad" });
  }
  return pills;
}

function metadataRows(item: ReviewItem): Array<{ label: string; value: string | { href: string; text: string } }> {
  return [
    { label: "sha256", value: item.sha256 },
    { label: "label", value: item.label ?? "unlabeled" },
    { label: "label source", value: item.label_source ?? "n/a" },
    { label: "labeled at", value: item.labeled_at ?? "n/a" },
    { label: "handles", value: item.handles.join(", ") || "none" },
    { label: "display names", value: item.display_names.join(", ") || "none" },
    { label: "seen count", value: String(item.seen_count) },
    { label: "source surfaces", value: item.source_surfaces.join(", ") || "none" },
    {
      label: "model",
      value: item.latest_model_predicted_label
        ? `${item.latest_model_predicted_label} (${formatScore(item.max_model_score)})`
        : "unscored",
    },
    {
      label: "threshold",
      value:
        item.latest_model_threshold != null
          ? `${formatScore(item.latest_model_threshold)} (Δ ${formatScore(item.latest_model_distance_to_threshold)})`
          : "n/a",
    },
    { label: "model run", value: item.latest_model_run_id ?? "n/a" },
    { label: "whitelisted", value: item.whitelisted ? "yes" : "no" },
    {
      label: "profile",
      value: item.example_profile_url ? { href: item.example_profile_url, text: item.example_profile_url } : "n/a",
    },
    {
      label: "tweet",
      value: item.example_tweet_url ? { href: item.example_tweet_url, text: item.example_tweet_url } : "n/a",
    },
    {
      label: "notification",
      value: item.example_notification_url ? { href: item.example_notification_url, text: item.example_notification_url } : "n/a",
    },
  ];
}

function App() {
  const queryClient = useQueryClient();
  const [queue, setQueue] = createSignal<QueueName>("needs_review");
  const [index, setIndex] = createSignal(0);
  const [selectedSha, setSelectedSha] = createSignal<string | null>(null);
  const [activeView, setActiveView] = createSignal<"individual" | "batch">("individual");
  const [gridSource, setGridSource] = createSignal<GridSource>("queue");
  const [gridFilter, setGridFilter] = createSignal<string>("needs_review");
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
    if (selectedRunId() !== summary.selected_run_id) {
      setSelectedRunId(summary.selected_run_id);
    }

    const preferredQueue = preferredQueueOrder.find((candidate) => (summary.queue_counts[candidate] ?? 0) > 0) ?? "needs_review";

    if (!(queue() in summary.queue_counts)) {
      setQueue(preferredQueue);
      setIndex(0);
    } else if (queue() === "needs_review" && preferredQueue !== "needs_review" && (summary.queue_counts.needs_review ?? 0) === 0) {
      setQueue(preferredQueue);
      setIndex(0);
    }

    if (gridSource() === "queue") {
      if (!(gridFilter() in summary.queue_counts)) {
        setGridFilter(queue());
      } else if (gridFilter() === "needs_review" && preferredQueue !== "needs_review" && (summary.queue_counts.needs_review ?? 0) === 0) {
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

  const activeQueueCount = createMemo(() => {
    if (activeView() === "batch") {
      return batchQuery.data?.total ?? summaryQuery.data?.queue_counts[queue()] ?? 0;
    }
    if (selectedSha() === null) {
      return queueQuery.data?.total ?? summaryQuery.data?.queue_counts[queue()] ?? 0;
    }
    return summaryQuery.data?.queue_counts[queue()] ?? 0;
  });

  const summaryCopy = createMemo(() => {
    const summary = summaryQuery.data;
    if (!summary) {
      return "Loading summary…";
    }
    const queueCount = activeQueueCount();
    if (queue() === "needs_review") {
      return `${summary.total_images} images, ${queueCount} need review, run ${summary.selected_run_id ?? "unscored"}`;
    }
    return `${summary.total_images} images, ${queueCount} in ${queueLabels[queue()].toLowerCase()}, ${summary.needs_review} need review overall, run ${summary.selected_run_id ?? "unscored"}`;
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

  async function refetchVisibleQueries() {
    await Promise.all([
      summaryQuery.refetch(),
      historyQuery.refetch(),
      gridQuery.refetch(),
      activeView() === "batch" ? batchQuery.refetch() : queueQuery.refetch(),
      selectedSha() !== null ? selectedItemQuery.refetch() : Promise.resolve(),
    ]);
  }

  const labelMutation = createMutation(() => ({
    mutationFn: (payload: { sha256: string; label: ReviewLabel }) =>
      postJson<{ ok: true; changed: boolean }>("/api/label", payload),
  }));

  const batchMutation = createMutation(() => ({
    mutationFn: (payload: { items: Array<{ sha256: string; label: ReviewLabel }> }) =>
      postJson<{ ok: true; batch_id: string; count: number }>("/api/batch-label", payload),
  }));

  const undoMutation = createMutation(() => ({
    mutationFn: () => postJson<{ undone_sha256: string | null }>("/api/undo"),
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
    await refetchVisibleQueries();
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
    await refetchVisibleQueries();
  }

  function moveBatchOffset(delta: number) {
    setBatchOffset((value) => Math.max(0, value + delta));
  }

  async function handleUndo() {
    const wasQueueItem = selectedSha() === null;
    const payload = await undoMutation.mutateAsync();
    setSelectedSha(payload.undone_sha256 ?? null);
    if (wasQueueItem) {
      setIndex((value) => Math.max(0, value - 1));
    }
    await invalidateAll();
    await refetchVisibleQueries();
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
        moveBatchOffset(-9);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        moveBatchOffset(9);
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
                  <For each={summary().available_run_ids}>
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
                            <option value={queueName}>{`${queueLabels[queueName]} (${summary().queue_counts[queueName] ?? 0})`}</option>
                          )}
                        </For>
                      </optgroup>
                    )}
                  </For>
                )}
              </Show>
            </select>
          </label>
          <p class="summary-copy">{summaryCopy()}</p>
          <div class="actions">
            <button type="button" disabled={!summaryQuery.data?.can_undo || undoMutation.isPending} onClick={() => void handleUndo()}>
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
                              <span>{labelDisplay[entry.new_label]}</span>
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
                  <button type="button" disabled={batchOffset() === 0} onClick={() => moveBatchOffset(-9)}>
                    Prev batch
                  </button>
                  <button type="button" disabled={batchAssignments().length === 0} onClick={() => moveBatchOffset(9)}>
                    Next batch
                  </button>
                  <button type="button" disabled={batchAssignments().length === 0 || batchMutation.isPending} onClick={() => void handleCommitBatch()}>
                    Enter: commit batch
                  </button>
                </div>
              </div>
              <div class="hint">Numpad cycles labels, left/right moves batches, Enter commits or advances. Batch labels are stored as manual.</div>
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
                          <Show when={scoreBarPercent(entry.item.max_model_score) != null}>
                            <Show
                              when={scoreBarPercent(entry.item.latest_model_threshold) != null}
                              fallback={
                                <div
                                  class="score-bar"
                                  data-predicted-label={entry.item.latest_model_predicted_label ?? "unscored"}
                                  aria-label={`p ${formatScore(entry.item.max_model_score)} t ${formatScore(entry.item.latest_model_threshold)}`}
                                  style={`--score-percent: ${scoreBarPercent(entry.item.max_model_score)}%;`}
                                >
                                  <div class="score-bar-fill" />
                                </div>
                              }
                            >
                              <div
                                class="score-bar"
                                data-predicted-label={entry.item.latest_model_predicted_label ?? "unscored"}
                                aria-label={`p ${formatScore(entry.item.max_model_score)} t ${formatScore(entry.item.latest_model_threshold)}`}
                                style={`--score-percent: ${scoreBarPercent(entry.item.max_model_score)}%; --threshold-percent: ${scoreBarPercent(entry.item.latest_model_threshold)}%;`}
                              >
                                <div class="score-bar-fill" />
                                <div class="score-bar-threshold" />
                              </div>
                            </Show>
                          </Show>
                          <div class="batch-caption">
                            <span class="batch-caption-left">{batchTileKeys[itemIndex()]}</span>
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
                                  <option value={queueName}>{`${queueLabels[queueName]} (${summary().queue_counts[queueName] ?? 0})`}</option>
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
                                      <span class="thumb-label">{item.label ? labelDisplay[item.label] : "needs review"}</span>
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
