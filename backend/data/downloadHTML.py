"""
Download arXiv HTML pages listed in arxiv_rag_metadata.jsonl.

Input:
    ../arxiv_rag_metadata.jsonl

Output:
    ../arxiv_rag_papers/html/<arxiv_id>.html

Failed downloads:
    ../arxiv_rag_papers/html_failed.jsonl

The script is resumable:
- Existing HTML files are skipped.
- Failed papers are recorded.
- A network/API failure does not stop the whole run.
"""

import json
import time
from pathlib import Path

import requests

METADATA_FILE = Path("arxiv_rag_metadata.jsonl")
OUTPUT_DIR = Path("arxiv_rag_papers/html")
FAILED_FILE = Path("arxiv_rag_papers/html_failed.jsonl")

SLEEP_BETWEEN_REQUESTS = 3
TIMEOUT = 30

BASE_URL = "https://arxiv.org/html"


def load_papers():
    """Yield papers from the metadata JSONL."""
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            yield json.loads(line)


def download_html(arxiv_id: str, session: requests.Session):
    """Download one arXiv HTML page."""

    output_path = OUTPUT_DIR / f"{arxiv_id}.html"

    if output_path.exists():
        return "skipped"

    url = f"{BASE_URL}/{arxiv_id}"

    try:
        response = session.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "arxiv-rag-research/1.0"},
        )

        if response.status_code == 404:
            return "not_available"

        response.raise_for_status()

        output_path.write_text(
            response.text,
            encoding="utf-8",
        )

        return "downloaded"

    except requests.RequestException as exc:
        print(f"  ERROR {arxiv_id}: {exc}")
        return f"error: {exc}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    downloaded = 0
    skipped = 0
    unavailable = 0
    failed = 0

    with FAILED_FILE.open("a", encoding="utf-8") as failed_out:
        for index, paper in enumerate(load_papers(), start=1):
            arxiv_id = paper["id"]

            print(f"[{index}] {arxiv_id}", end=" ")

            status = download_html(arxiv_id, session)

            if status == "downloaded":
                downloaded += 1
                print("-> downloaded")

            elif status == "skipped":
                skipped += 1
                print("-> skipped")

            elif status == "not_available":
                unavailable += 1
                print("-> HTML unavailable")

                failed_out.write(
                    json.dumps(
                        {
                            "id": arxiv_id,
                            "status": "html_unavailable",
                        }
                    )
                    + "\n"
                )

            else:
                failed += 1
                print("-> failed")

                failed_out.write(
                    json.dumps(
                        {
                            "id": arxiv_id,
                            "status": status,
                        }
                    )
                    + "\n"
                )

            failed_out.flush()

            # Don't sleep after the final request isn't important,
            # so sleeping here is acceptable and keeps the implementation simple.
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("\nDone.")
    print(f"Downloaded:      {downloaded}")
    print(f"Skipped:         {skipped}")
    print(f"HTML unavailable:{unavailable}")
    print(f"Failed:          {failed}")


if __name__ == "__main__":
    main()
