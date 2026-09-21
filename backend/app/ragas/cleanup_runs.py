"""Delete old Langfuse experiment runs by removing their traces.

Workaround for v4 events_only mode where the dataset run management API
is unavailable. In v4, experiment records are derived from trace events,
so deleting the traces removes the run.
"""

import httpx
from langfuse import Langfuse

from app.config import settings

BASE = settings.langfuse_base_url.rstrip("/")
AUTH = (settings.langfuse_public_key, settings.langfuse_secret_key)
DATASET = "ragwise_ragas_testset"

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)


def normalize_dt(dt_value) -> str:
    """Normalize any datetime to strict ISO 8601 UTC with Z suffix.

    '2026-09-18 17:32:01.206000+00:00' -> '2026-09-18T17:32:01.206Z'
    """
    s = str(dt_value)
    # Space -> T
    s = s.replace(" ", "T", 1)
    # +00:00 / +0000 -> Z
    s = s.replace("+00:00", "Z").replace("+0000", "Z")
    # Truncate microseconds (6 digits) to milliseconds (3 digits)
    if "." in s:
        before, after = s.split(".", 1)
        s = f"{before}.{after[:3]}Z"
    return s


# === 1. Get dataset ID + createdAt ===
print(f"Fetching dataset '{DATASET}'...")

dataset_id = None
created_at = None

# Try SDK first
try:
    ds = langfuse.get_dataset(DATASET)
    dataset_id = getattr(ds, "id", None)
    created_at = getattr(ds, "created_at", None) or getattr(ds, "createdAt", None)
    print(f"  id={dataset_id}  createdAt={created_at}")
except Exception:
    pass

# Fallback: REST API
if not dataset_id:
    r = httpx.get(f"{BASE}/api/public/datasets/{DATASET}", auth=AUTH, timeout=10)
    if r.status_code == 200:
        ds = r.json()
        dataset_id = ds.get("id")
        created_at = ds.get("createdAt") or ds.get("created_at")
        print(f"  (REST) id={dataset_id}  createdAt={created_at}")
    else:
        print(f"  REST failed: {r.status_code}")

if not dataset_id:
    raise SystemExit("Could not get dataset ID.")

# Normalize the datetime for the API
from_start = normalize_dt(created_at) if created_at else "2000-01-01T00:00:00.000Z"
print(f"  fromStartTime (normalized): {from_start}")

# === 2. List experiments for this dataset ===
print(f"\nListing experiments...")

r = httpx.get(
    f"{BASE}/api/public/experiments",
    auth=AUTH,
    params={
        "fromStartTime": from_start,
        "datasetId": dataset_id,
        "limit": 100,
    },
    timeout=30,
)

if r.status_code != 200:
    print(f"  Failed: {r.status_code} {r.text[:300]}")
    raise SystemExit

data = r.json()
experiments = data.get("data", []) if isinstance(data, dict) else data

print(f"\nFound {len(experiments)} experiment(s):")
for i, exp in enumerate(experiments):
    name = exp.get("name", "?")
    eid = exp.get("id", "?")
    print(f"  [{i}] {name}  (id={eid})")

if not experiments:
    raise SystemExit("Nothing to delete.")

# === 3. Confirm ===
if (
    input(f"\nDelete all {len(experiments)} experiment(s)? (y/N): ").strip().lower()
    != "y"
):
    raise SystemExit("Aborted.")

# === 4. For each experiment: collect trace IDs, then delete traces ===
print()
for exp in experiments:
    exp_id = exp.get("id")
    exp_name = exp.get("name", "?")

    if not exp_id:
        print(f"  ✗ {exp_name}: no experiment ID")
        continue

    # --- 4a. Collect all trace IDs (paginated, 50 per page) ---
    trace_ids: list[str] = []
    cursor = None

    while True:
        params: dict = {
            "fromStartTime": from_start,
            "experimentId": exp_id,
        }
        if cursor:
            params["cursor"] = cursor

        r = httpx.get(
            f"{BASE}/api/public/experiment-items",
            auth=AUTH,
            params=params,
            timeout=30,
        )
        if r.status_code != 200:
            print(
                f"  ✗ {exp_name}: failed to fetch items ({r.status_code} {r.text[:200]})"
            )
            break

        data = r.json()
        items = data.get("data", []) if isinstance(data, dict) else data

        for item in items:
            tid = item.get("traceId") or item.get("trace_id") or item.get("id")
            if tid:
                trace_ids.append(tid)

        # Pagination cursor
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        cursor = meta.get("cursor") or meta.get("nextCursor")
        if not cursor or not items:
            break

    if not trace_ids:
        print(f"  ✗ {exp_name}: no traces found")
        continue

    # --- 4b. Delete traces in batches of 100 ---
    deleted = 0
    for i in range(0, len(trace_ids), 100):
        batch = trace_ids[i : i + 100]
        r = httpx.request(
            "DELETE",
            f"{BASE}/api/public/traces",
            auth=AUTH,
            json={"traceIds": batch},
            timeout=30,
        )
        if r.status_code in (200, 204):
            deleted += len(batch)
        else:
            print(
                f"  ✗ {exp_name}: batch {i // 100} failed ({r.status_code} {r.text[:100]})"
            )

    print(f"  ✓ {exp_name}: deleted {deleted}/{len(trace_ids)} traces")

langfuse.flush()
print("\n✅ Done. Runs should disappear from the UI within a few minutes.")
print("   Note: GET /api/public/datasets/{name} may still show stale run")
print("   names in its 'runs' field — the experiments API reflects reality.")
