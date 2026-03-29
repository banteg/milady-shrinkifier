from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from .download_collection_samples import COLLECTIONS as NFT_COLLECTIONS, COLLECTION_LABEL_SOURCE
from .mobilenet_common import DatasetEntry, SPLIT_SEED, dataset_entries_to_jsonl
from .pipeline_common import (
    COLLECTION_MANIFEST_PATH,
    SPLIT_MANIFEST_PATH,
    SPLIT_ROOT,
    connect_db,
    connect_offline_cache_db,
    get_file_fingerprint,
    now_iso,
    read_json_file,
    resolve_repo_path,
    sha256_bytes,
    write_json_file,
)

COLLECTIONS_BY_SLUG = {collection.slug: collection for collection in NFT_COLLECTIONS}
SOURCE_PRIORITY = {
    "export": 0,
    "collection": 1,
}
LABEL_TIER_PRIORITY = {
    "gold": 0,
    "trusted": 1,
}
TRUSTED_COLLECTION_WEIGHT = 0.5
DEFAULT_MODEL_LABEL_WEIGHT = 1.0
GOLD_LABEL_SOURCE = "manual"
MODEL_LABEL_SOURCE = "model"
TRUSTED_LABEL_SOURCES = {MODEL_LABEL_SOURCE}
PERCEPTUAL_HASH_HAMMING_THRESHOLD = 4
COLLECTION_HOLDOUT_VAL_COUNT = 64
COLLECTION_HOLDOUT_TEST_COUNT = 64


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    path: Path
    label: str
    source: str
    raw_sha: str
    pixel_digest: str
    perceptual_hash: str
    label_source: str
    label_tier: str
    sample_weight: float
    blind_eval_eligible: bool
    exported_sha: str | None = None


@dataclass(slots=True)
class GroupRecord:
    group_id: str
    label: str
    split: str
    canonical: SampleRecord
    members: list[SampleRecord]
    blind_eval_eligible: bool


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


@dataclass(slots=True)
class BKTreeNode:
    value: int
    indices: list[int]
    children: dict[int, "BKTreeNode"]


class HammingBKTree:
    def __init__(self) -> None:
        self.root: BKTreeNode | None = None

    def insert(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKTreeNode(value=value, indices=[index], children={})
            return

        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKTreeNode(value=value, indices=[index], children={})
                return
            node = child

    def search(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []

        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= max_distance:
                matches.extend(node.indices)
            lower = max(1, distance - max_distance)
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)
        return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize grouped train/val/test JSONL files for MobileNetV3-Small training.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--model-label-weight", type=float, default=DEFAULT_MODEL_LABEL_WEIGHT)
    parser.add_argument("--reset-splits", action="store_true", help="Recompute all split assignments from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = connect_db()
    cache_connection = connect_offline_cache_db()
    try:
        print("[build-dataset] collecting samples", flush=True)
        samples = build_sample_records(connection, cache_connection, args.model_label_weight)
        print(f"[build-dataset] collected {len(samples)} samples", flush=True)
        print("[build-dataset] grouping duplicates", flush=True)
        groups = build_group_records(samples)
        print(f"[build-dataset] grouped into {len(groups)} image families", flush=True)
        print("[build-dataset] assigning splits", flush=True)
        collection_holdout_assignments = assign_collection_holdout_groups(groups)
        assignments, manifest_mode = assign_group_splits(groups, args, SPLIT_MANIFEST_PATH)
        print(f"[build-dataset] split mode={manifest_mode}", flush=True)

        dataset_entries: list[DatasetEntry] = []
        exported_split_updates: dict[str, str] = {}
        manifest_groups: list[dict[str, object]] = []

        for group in groups:
            split = assignments[group.group_id]
            dataset_entries.append(
                DatasetEntry(
                    sample_id=group.canonical.sample_id,
                    path=group.canonical.path,
                    label=group.label,
                    source=group.canonical.source,
                    split=split,
                    label_source=group.canonical.label_source,
                    label_tier=group.canonical.label_tier,
                    sample_weight=group.canonical.sample_weight,
                )
            )
            for member in group.members:
                if member.exported_sha is not None:
                    exported_split_updates[member.exported_sha] = split
            manifest_groups.append(
                {
                    "groupId": group.group_id,
                    "label": group.label,
                    "split": split,
                    "canonical": {
                        "id": group.canonical.sample_id,
                        "path": str(group.canonical.path),
                        "source": group.canonical.source,
                        "labelSource": group.canonical.label_source,
                        "labelTier": group.canonical.label_tier,
                        "sampleWeight": group.canonical.sample_weight,
                        "blindEvalEligible": group.blind_eval_eligible,
                        "rawSha": group.canonical.raw_sha,
                        "pixelDigest": group.canonical.pixel_digest,
                        "perceptualHash": group.canonical.perceptual_hash,
                    },
                    "members": [
                        {
                            "id": member.sample_id,
                            "path": str(member.path),
                            "source": member.source,
                            "labelSource": member.label_source,
                            "labelTier": member.label_tier,
                            "sampleWeight": member.sample_weight,
                            "blindEvalEligible": member.blind_eval_eligible,
                            "rawSha": member.raw_sha,
                            "pixelDigest": member.pixel_digest,
                            "perceptualHash": member.perceptual_hash,
                            "exportedSha": member.exported_sha,
                        }
                        for member in sorted(group.members, key=lambda item: item.sample_id)
                    ],
                }
            )

        for exported_sha, split in exported_split_updates.items():
            connection.execute(
                "UPDATE images SET split = ?, updated_at = CURRENT_TIMESTAMP WHERE sha256 = ?",
                (split, exported_sha),
            )
        connection.commit()
        cache_connection.commit()

        by_split = {
            split_name: [entry for entry in dataset_entries if entry.split == split_name]
            for split_name in ("train", "val", "test")
        }

        SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
        for split_name, entries in by_split.items():
            dataset_entries_to_jsonl(entries, SPLIT_ROOT / f"{split_name}.jsonl")

        summary = {
            "manifestMode": manifest_mode,
            "sampleCount": len(samples),
            "groupCount": len(groups),
            "dedupedSampleCount": len(dataset_entries),
            "duplicatesRemoved": len(samples) - len(dataset_entries),
            "blindEvalEligibleGroups": sum(1 for group in groups if group.blind_eval_eligible),
            "trainOnlyGroups": sum(1 for group in groups if not group.blind_eval_eligible),
            "collectionBlindHoldoutGroups": len(collection_holdout_assignments),
            "splits": {
                split_name: {
                    "total": len(entries),
                    "milady": sum(1 for entry in entries if entry.label == "milady"),
                    "not_milady": sum(1 for entry in entries if entry.label == "not_milady"),
                    "gold": sum(1 for entry in entries if entry.label_tier == "gold"),
                    "trusted": sum(1 for entry in entries if entry.label_tier == "trusted"),
                }
                for split_name, entries in by_split.items()
            },
        }
        write_json_file(
            SPLIT_MANIFEST_PATH,
            {
                "version": 2,
                "generatedAt": now_iso(),
                "mode": manifest_mode,
                "evaluationPolicy": {
                    "blindEvalIncludesCollectionHoldoutPositives": True,
                    "goldLabelSource": GOLD_LABEL_SOURCE,
                    "trustedLabelSources": sorted(TRUSTED_LABEL_SOURCES),
                    "trustedCollectionWeight": TRUSTED_COLLECTION_WEIGHT,
                    "modelLabelWeight": args.model_label_weight,
                    "collectionBlindHoldoutValCount": COLLECTION_HOLDOUT_VAL_COUNT,
                    "collectionBlindHoldoutTestCount": COLLECTION_HOLDOUT_TEST_COUNT,
                },
                "ratios": {
                    "train": args.train_ratio,
                    "val": args.val_ratio,
                    "test": args.test_ratio,
                },
                "groups": manifest_groups,
            },
        )
        (SPLIT_ROOT / "summary.json").write_text(json.dumps(summary["splits"], indent=2, sort_keys=True))
        print("[build-dataset] wrote split manifest and jsonl files", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        cache_connection.close()
        connection.close()


def build_sample_records(connection, cache_connection, model_label_weight: float) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    processed = 0

    for collection, token_id, path in load_collection_rows():
        fingerprint = get_file_fingerprint(cache_connection, path, 128)
        if not fingerprint.readable:
            continue
        samples.append(
            SampleRecord(
                sample_id=f"collection:{collection.slug}:{token_id}",
                path=path,
                label="milady",
                source=collection.slug,
                raw_sha=fingerprint.raw_sha,
                pixel_digest=fingerprint.pixel_digest,
                perceptual_hash=fingerprint.perceptual_hash,
                label_source=COLLECTION_LABEL_SOURCE,
                label_tier="trusted",
                sample_weight=TRUSTED_COLLECTION_WEIGHT,
                blind_eval_eligible=False,
            )
        )
        processed = maybe_flush_fingerprint_cache(cache_connection, processed + 1)

    exported_rows = connection.execute(
        """
        SELECT sha256, local_path, label
             , label_source
        FROM images
        WHERE label IN ('milady', 'not_milady')
          AND local_path IS NOT NULL
          AND label_source IN ('manual', 'model')
        ORDER BY sha256 ASC
        """
    ).fetchall()
    for row in exported_rows:
        path = resolve_repo_path(str(row["local_path"]))
        if not path.exists():
            raise SystemExit(f"Missing exported avatar file: {path}")
        label_source = str(row["label_source"])
        fingerprint = get_file_fingerprint(cache_connection, path, 128)
        if not fingerprint.readable:
            raise SystemExit(f"Unreadable exported avatar file: {path}")
        label_tier = label_tier_for_export_label_source(label_source)
        samples.append(
            SampleRecord(
                sample_id=f"export:{row['sha256']}",
                path=path,
                label=str(row["label"]),
                source="export",
                raw_sha=str(row["sha256"]),
                pixel_digest=fingerprint.pixel_digest,
                perceptual_hash=fingerprint.perceptual_hash,
                label_source=label_source,
                label_tier=label_tier,
                sample_weight=sample_weight_for_export_label_source(label_source, model_label_weight),
                blind_eval_eligible=label_tier == "gold",
                exported_sha=str(row["sha256"]),
            )
        )
        processed = maybe_flush_fingerprint_cache(cache_connection, processed + 1)

    cache_connection.commit()
    return samples


def load_collection_rows() -> list[tuple[object, int, Path]]:
    if not COLLECTION_MANIFEST_PATH.exists():
        raise SystemExit("Missing collection manifest. Run `uv run milady download-collections` first.")

    manifest = read_json_file(COLLECTION_MANIFEST_PATH)
    raw_collections = manifest["collections"]

    rows: list[tuple[object, int, Path]] = []
    for collection in raw_collections:
        slug = str(collection["slug"])
        samples = collection["samples"]
        spec = COLLECTIONS_BY_SLUG.get(slug)
        if spec is None:
            raise SystemExit(f"Unknown collection in manifest: {slug}")
        for sample in samples:
            token_id = int(sample["tokenId"])
            local_path = str(sample["localPath"])
            path = resolve_repo_path(local_path)
            if not path.exists():
                raise SystemExit(f"Missing collection sample file: {path}")
            rows.append((spec, token_id, path))
    return rows


def build_group_records(samples: list[SampleRecord]) -> list[GroupRecord]:
    union_find = UnionFind(len(samples))
    raw_sha_to_index: dict[str, int] = {}
    pixel_digest_to_index: dict[str, int] = {}
    perceptual_hash_tree = HammingBKTree()

    for index, sample in enumerate(samples):
        previous = raw_sha_to_index.get(sample.raw_sha)
        if previous is not None:
            union_find.union(index, previous)
        else:
            raw_sha_to_index[sample.raw_sha] = index

        previous = pixel_digest_to_index.get(sample.pixel_digest)
        if previous is not None:
            union_find.union(index, previous)
        else:
            pixel_digest_to_index[sample.pixel_digest] = index

        if sample.perceptual_hash:
            perceptual_hash_value = int(sample.perceptual_hash, 16)
            for previous_index in perceptual_hash_tree.search(
                perceptual_hash_value,
                PERCEPTUAL_HASH_HAMMING_THRESHOLD,
            ):
                union_find.union(index, previous_index)
            perceptual_hash_tree.insert(perceptual_hash_value, index)

    buckets: dict[int, list[SampleRecord]] = {}
    for index, sample in enumerate(samples):
        buckets.setdefault(union_find.find(index), []).append(sample)

    groups: list[GroupRecord] = []
    for members in buckets.values():
        labels = {member.label for member in members}
        if len(labels) != 1:
            conflicts = ", ".join(sorted(f"{member.sample_id}:{member.label}" for member in members))
            raise SystemExit(f"Conflicting labels within duplicate group: {conflicts}")
        label = next(iter(labels))
        canonical = min(members, key=sample_sort_key)
        group_id = compute_group_id(members)
        blind_eval_eligible = all(member.blind_eval_eligible for member in members)
        groups.append(
            GroupRecord(
                group_id=group_id,
                label=label,
                split="",
                canonical=canonical,
                members=members,
                blind_eval_eligible=blind_eval_eligible,
            )
        )

    return sorted(groups, key=lambda group: group.canonical.sample_id)


def assign_group_splits(groups: list[GroupRecord], args: argparse.Namespace, manifest_path: Path) -> tuple[dict[str, str], str]:
    forced_eval_assignments = assign_collection_holdout_groups(groups)
    train_only_assignments = {
        group.group_id: "train"
        for group in groups
        if not group.blind_eval_eligible and group.group_id not in forced_eval_assignments
    }
    blind_eval_groups = [
        group
        for group in groups
        if group.blind_eval_eligible and group.group_id not in forced_eval_assignments
    ]

    if args.reset_splits or not manifest_path.exists():
        assignments = forced_eval_assignments | train_only_assignments | initial_group_assignments(
            blind_eval_groups,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        return assignments, "fresh"

    manifest = read_json_file(manifest_path)
    raw_groups = manifest.get("groups")
    if not isinstance(raw_groups, list):
        assignments = train_only_assignments | initial_group_assignments(
            blind_eval_groups,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        return assignments, "fresh"

    assignments = {
        str(group["groupId"]): str(group["split"])
        for group in raw_groups
        if isinstance(group, dict) and group.get("split") in {"train", "val", "test"}
    }
    assignments.update(forced_eval_assignments)
    for group_id, split in train_only_assignments.items():
        assignments[group_id] = split

    new_groups = [
        group
        for group in blind_eval_groups
        if group.group_id not in assignments
    ]
    if not new_groups:
        return {group.group_id: assignments[group.group_id] for group in groups}, "reused"

    appended = assign_train_val_only(new_groups, args.train_ratio, args.val_ratio)
    assignments.update(appended)
    return {group.group_id: assignments[group.group_id] for group in groups}, "appended"


def initial_group_assignments(groups: list[GroupRecord], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, str]:
    if not groups:
        return {}
    group_ids = [group.group_id for group in groups]
    labels = [1 if group.label == "milady" else 0 for group in groups]
    test_indices, remaining_indices = stratified_group_partition(group_ids, labels, test_ratio, SPLIT_SEED)
    remaining_group_ids = [group_ids[index] for index in remaining_indices]
    remaining_labels = [labels[index] for index in remaining_indices]
    relative_val_ratio = val_ratio / max(1e-9, train_ratio + val_ratio)
    val_indices_within_remaining, train_indices_within_remaining = stratified_group_partition(
        remaining_group_ids,
        remaining_labels,
        relative_val_ratio,
        SPLIT_SEED + 1,
    )

    assignments = {group_ids[index]: "test" for index in test_indices}
    for index in val_indices_within_remaining:
        assignments[remaining_group_ids[index]] = "val"
    for index in train_indices_within_remaining:
        assignments[remaining_group_ids[index]] = "train"
    return assignments


def assign_train_val_only(groups: list[GroupRecord], train_ratio: float, val_ratio: float) -> dict[str, str]:
    if not groups:
        return {}
    group_ids = [group.group_id for group in groups]
    labels = [1 if group.label == "milady" else 0 for group in groups]
    relative_val_ratio = val_ratio / max(1e-9, train_ratio + val_ratio)
    val_indices, train_indices = stratified_group_partition(group_ids, labels, relative_val_ratio, SPLIT_SEED + 2)
    assignments = {group_ids[index]: "val" for index in val_indices}
    for index in train_indices:
        assignments[group_ids[index]] = "train"
    return assignments


def assign_collection_holdout_groups(groups: list[GroupRecord]) -> dict[str, str]:
    eligible_groups = [
        group
        for group in groups
        if group.label == "milady" and all(member.label_source == "collection_corpus" for member in group.members)
    ]
    ranked_groups = sorted(
        eligible_groups,
        key=lambda group: sha256_bytes(f"collection-holdout:{group.group_id}".encode("utf-8")),
    )
    test_groups = ranked_groups[:COLLECTION_HOLDOUT_TEST_COUNT]
    val_groups = ranked_groups[
        COLLECTION_HOLDOUT_TEST_COUNT : COLLECTION_HOLDOUT_TEST_COUNT + COLLECTION_HOLDOUT_VAL_COUNT
    ]
    assignments = {group.group_id: "test" for group in test_groups}
    assignments.update({group.group_id: "val" for group in val_groups})
    return assignments


def stratified_group_partition(group_ids: list[str], labels: list[int], holdout_ratio: float, random_state: int) -> tuple[list[int], list[int]]:
    total = len(group_ids)
    if total == 0:
        return [], []
    if holdout_ratio <= 0:
        return [], list(range(total))

    target_splits = max(2, round(1 / holdout_ratio))
    class_counts = {
        label: labels.count(label)
        for label in set(labels)
    }
    min_class_count = min(class_counts.values()) if class_counts else 1
    n_splits = min(target_splits, min_class_count, total)
    if n_splits < 2:
        holdout_count = max(1, round(total * holdout_ratio))
        holdout = list(range(holdout_count))
        remaining = list(range(holdout_count, total))
        return holdout, remaining

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split = next(splitter.split([[0]] * total, labels, groups=group_ids))
    remaining_indices, holdout_indices = split
    return list(holdout_indices), list(remaining_indices)


def compute_group_id(members: list[SampleRecord]) -> str:
    keys = sorted({f"sha:{member.raw_sha}" for member in members} | {f"pix:{member.pixel_digest}" for member in members})
    return sha256_bytes("|".join(keys).encode("utf-8"))


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def label_tier_for_export_label_source(label_source: str) -> str:
    if label_source == GOLD_LABEL_SOURCE:
        return "gold"
    if label_source in TRUSTED_LABEL_SOURCES:
        return "trusted"
    raise SystemExit(f"Unsupported exported label source for dataset build: {label_source}")


def sample_weight_for_export_label_source(label_source: str, model_label_weight: float) -> float:
    if label_source in TRUSTED_LABEL_SOURCES:
        return model_label_weight
    return 1.0


def sample_sort_key(sample: SampleRecord) -> tuple[int, str]:
    label_tier_priority = LABEL_TIER_PRIORITY.get(sample.label_tier, len(LABEL_TIER_PRIORITY))
    if sample.source == "export":
        priority = SOURCE_PRIORITY["export"]
    else:
        priority = SOURCE_PRIORITY["collection"]
    return label_tier_priority, priority, sample.sample_id


def maybe_flush_fingerprint_cache(connection, processed: int, flush_every: int = 250) -> int:
    if processed % flush_every == 0:
        connection.commit()
        print(f"[build-dataset] fingerprint cache persisted for {processed} samples", flush=True)
    return processed
if __name__ == "__main__":
    main()
