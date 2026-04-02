from __future__ import annotations

import argparse
import html
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx
import msgspec

from .image_files import guess_extension
from .paths import COLLECTION_MANIFEST_PATH, COLLECTION_ROOT
from .wire import CollectionFailure, CollectionManifest, CollectionManifestCollection, CollectionSample, decode_json, dump_json

IPFS_GATEWAYS = (
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
    "https://nftstorage.link/ipfs/",
)
DEFAULT_TIMEOUT = 20.0
BATCH_SIZE = 200


class CollectionSpec(msgspec.Struct, frozen=True, kw_only=True):
    slug: str
    name: str
    total_supply: int
    target_count: int
    contract: str | None = None
    token_id_start: int = 1
    image_url_templates: tuple[str, ...] = ()
    metadata_url_template: str | None = None
    asset_page_template: str | None = None


COLLECTIONS: tuple[CollectionSpec, ...] = (
    CollectionSpec(
        slug="milady-maker",
        name="Milady Maker",
        total_supply=10_000,
        target_count=10_000,
        token_id_start=0,
        image_url_templates=(
            "https://www.miladymaker.net/milady/{token_id}.png",
            "https://miladymaker.net/milady/{token_id}.png",
        ),
    ),
    CollectionSpec(
        slug="remilio",
        name="Redacted Remilio Babies",
        total_supply=10_000,
        target_count=2_000,
        contract="0xD3D9ddd0CF0A5F0BFB8f7fcEAe075DF687eAEBaB",
        metadata_url_template="https://remilio.org/remilio/json/{token_id}",
    ),
    CollectionSpec(
        slug="pixelady",
        name="Pixelady Maker",
        total_supply=10_000,
        target_count=1_000,
        contract="0x8Fc0D90f2C45a5e7f94904075c952e0943CFCCfd",
        metadata_url_template="ipfs://bafybeigd7557iwardhnwg5kbmg2s7tmuxqkstjeoixu7wunooiywbb3jqq/{token_id}",
    ),
)

class DownloadResult(msgspec.Struct, kw_only=True):
    token_id: int
    success: bool
    local_path: str | None = None
    image_url: str | None = None
    metadata_url: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download configured NFT collection samples into the local collection cache.")
    parser.add_argument(
        "--collection",
        dest="collections",
        action="append",
        choices=[collection.slug for collection in COLLECTIONS],
        help="Limit to one or more configured collections. Defaults to all configured collections.",
    )
    parser.add_argument("--concurrency", type=int, default=12, help="Concurrent downloads per batch.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [collection for collection in COLLECTIONS if not args.collections or collection.slug in args.collections]
    manifest_collections: list[CollectionManifestCollection] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=args.timeout,
        headers={"User-Agent": "milady-shrinkifier/collections"},
    ) as client:
        for collection in selected:
            token_ids = sample_token_ids(collection)
            collection_root = COLLECTION_ROOT / collection.slug
            collection_root.mkdir(parents=True, exist_ok=True)
            existing_files = index_existing_files(collection_root)
            results: list[DownloadResult] = []
            for offset in range(0, len(token_ids), BATCH_SIZE):
                batch = token_ids[offset : offset + BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    future_map = {
                        executor.submit(download_token, client, collection, collection_root, existing_files, token_id, args.force): token_id
                        for token_id in batch
                    }
                    for future in as_completed(future_map):
                        results.append(future.result())

            successful_results = sorted((result for result in results if result.success), key=lambda result: result.token_id)
            failed_results = sorted((result for result in results if not result.success), key=lambda result: result.token_id)

            manifest_collections.append(
                CollectionManifestCollection(
                    slug=collection.slug,
                    name=collection.name,
                    contract=collection.contract,
                    total_supply=collection.total_supply,
                    target_count=collection.target_count,
                    sample_count=len(token_ids),
                    downloaded_count=len(successful_results),
                    failed_count=len(failed_results),
                    samples=[
                        CollectionSample(
                            token_id=result.token_id,
                            local_path=str(result.local_path),
                            image_url=result.image_url,
                            metadata_url=result.metadata_url,
                        )
                        for result in successful_results
                    ],
                    failures=[
                        CollectionFailure(
                            token_id=result.token_id,
                            error=str(result.error),
                        )
                        for result in failed_results
                    ],
                )
            )
            print(
                f"{collection.slug}: downloaded {len(successful_results)}/{len(token_ids)} "
                f"(target {min(collection.target_count, collection.total_supply)}, failed {len(failed_results)})"
            )

    manifest_payload = CollectionManifest(
        version=1,
        generated_at=datetime.now(UTC).isoformat(),
        collections=manifest_collections,
    )
    dump_json(COLLECTION_MANIFEST_PATH, manifest_payload)
    print(f"Wrote collection manifest to {COLLECTION_MANIFEST_PATH}")


def sample_token_ids(collection: CollectionSpec) -> list[int]:
    sample_count = min(collection.target_count, collection.total_supply)
    token_ids = list(range(collection.token_id_start, collection.token_id_start + collection.total_supply))
    if sample_count >= len(token_ids):
        return token_ids
    rng = random.Random(f"{collection.slug}:{collection.total_supply}:{sample_count}:v1")
    return sorted(rng.sample(token_ids, sample_count))


def download_token(
    client: httpx.Client,
    collection: CollectionSpec,
    collection_root: Path,
    existing_files: dict[int, Path],
    token_id: int,
    force: bool,
) -> DownloadResult:
    existing = existing_files.get(token_id)
    if existing is not None and not force:
        return DownloadResult(
            token_id=token_id,
            success=True,
            local_path=str(existing),
        )

    try:
        metadata_url, image_urls = resolve_urls(client, collection, token_id)
        last_error: Exception | None = None
        for image_url in image_urls:
            try:
                response = client.get(image_url)
                response.raise_for_status()
                payload = response.content
                extension = guess_extension(response.headers.get("content-type"), image_url)
                destination = collection_root / f"{token_id}{extension}"
                destination.write_bytes(payload)
                return DownloadResult(
                    token_id=token_id,
                    success=True,
                    local_path=str(destination),
                    image_url=image_url,
                    metadata_url=metadata_url,
                )
            except Exception as error:  # noqa: BLE001
                last_error = error
        raise ValueError(f"Failed image fetch for {collection.slug} #{token_id}: {last_error}")
    except Exception as error:  # noqa: BLE001
        return DownloadResult(token_id=token_id, success=False, error=str(error))


def index_existing_files(collection_root: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(collection_root.glob("*.*")):
        if not path.is_file():
            continue
        try:
            token_id = int(path.stem)
        except ValueError:
            continue
        indexed[token_id] = path
    return indexed


def resolve_urls(client: httpx.Client, collection: CollectionSpec, token_id: int) -> tuple[str | None, list[str]]:
    if collection.image_url_templates:
        return None, [template.format(token_id=token_id) for template in collection.image_url_templates]

    if collection.metadata_url_template:
        metadata_url_template = collection.metadata_url_template.format(token_id=token_id)
        metadata_url, payload = fetch_json_with_fallback(client, metadata_url_template)
        image_url = extract_image_url(payload)
        if image_url is None:
            raise ValueError(f"No image URL in metadata for {collection.slug} #{token_id}")
        _, image_url = fetch_head_candidate(client, image_url)
        return metadata_url, [image_url]

    if collection.asset_page_template:
        asset_url = collection.asset_page_template.format(token_id=token_id)
        response = client.get(asset_url)
        response.raise_for_status()
        image_url = extract_opensea_image_url(response.text, collection.contract)
        if image_url is None:
            raise ValueError(f"No OpenSea cached image URL found for {collection.slug} #{token_id}")
        return asset_url, [image_url]

    raise ValueError(f"No source configured for {collection.slug}")


def normalize_urls(url: str) -> list[str]:
    if not url.startswith("ipfs://"):
        return [url]
    cid_path = url.removeprefix("ipfs://")
    return [f"{gateway}{cid_path}" for gateway in IPFS_GATEWAYS]


def extract_image_url(payload: dict[str, str]) -> str:
    image_url = payload.get("image")
    if image_url:
        return image_url
    raise ValueError("metadata payload is missing image")


def fetch_json_with_fallback(client: httpx.Client, url: str) -> tuple[str, dict[str, str]]:
    errors: list[str] = []
    for candidate in normalize_urls(url):
        try:
            response = client.get(candidate)
            response.raise_for_status()
            return candidate, decode_json(response.content, dict[str, str])
        except Exception as error:  # noqa: BLE001
            errors.append(f"{candidate}: {error}")
    raise ValueError("; ".join(errors))


def fetch_head_candidate(client: httpx.Client, url: str) -> tuple[str, str]:
    errors: list[str] = []
    for candidate in normalize_urls(url):
        try:
            response = client.head(candidate)
            response.raise_for_status()
            return candidate, candidate
        except Exception as error:  # noqa: BLE001
            errors.append(f"{candidate}: {error}")

    for candidate in normalize_urls(url):
        try:
            response = client.get(candidate, headers={"Range": "bytes=0-0"})
            response.raise_for_status()
            return candidate, candidate
        except Exception as error:  # noqa: BLE001
            errors.append(f"{candidate}: {error}")

    raise ValueError("; ".join(errors))


def extract_opensea_image_url(page_html: str, contract: str | None) -> str | None:
    if contract is None:
        return None
    normalized_contract = contract.lower()
    pattern = re.compile(rf"https://i2c\.seadn\.io/ethereum/{re.escape(normalized_contract)}/[^\"' ]+")
    match = pattern.search(page_html)
    if match is None:
        return None
    return html.unescape(match.group(0)).replace("\\", "")


if __name__ == "__main__":
    main()
