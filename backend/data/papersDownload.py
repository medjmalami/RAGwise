"""
Download arXiv papers (metadata + optionally PDFs) matching a search query.

Respects arXiv's API usage guidance:
- max ~1 request per 3 seconds to the query API
- paginates in modest batches instead of requesting thousands at once
- resumable: already-downloaded PDFs are skipped on re-run

Usage:
    python download_arxiv_papers.py            # test run, 50 papers, metadata + PDFs
    Then edit main() at the bottom to remove the limit for the full 3060.
"""

import json
import os
import re
import time

import feedparser
import requests

BASE_URL = "http://export.arxiv.org/api/query"
OUTPUT_DIR = "arxiv_rag_papers"
METADATA_FILE = "arxiv_rag_metadata.jsonl"
BATCH_SIZE = 100  # papers per API request
SLEEP_BETWEEN_REQUESTS = 3  # seconds, per arXiv API terms of use
SLEEP_BETWEEN_DOWNLOADS = 1  # seconds, be polite to the PDF mirror

categories = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.MA)"
keywords = '(ti:"retrieval augmented" OR ti:"RAG" OR ti:"retrieval-augmented")'
query = f"{categories} AND {keywords}"


def fetch_all_metadata(query, batch_size=BATCH_SIZE):
    """Page through the arXiv API and yield each entry's metadata."""
    start = 0
    total = None
    while True:
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        if total is None:
            total = int(feed.feed.opensearch_totalresults)
            print(f"Total results reported by API: {total}")

        if not feed.entries:
            break

        for entry in feed.entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            pdf_url = next(
                (link.href for link in entry.links if link.type == "application/pdf"),
                None,
            )
            yield {
                "id": arxiv_id,
                "title": entry.title.strip().replace("\n", " "),
                "authors": [a.name for a in entry.authors],
                "summary": entry.summary.strip().replace("\n", " "),
                "published": entry.published,
                "updated": entry.updated,
                "pdf_url": pdf_url,
                "primary_category": entry.arxiv_primary_category["term"],
            }

        start += batch_size
        print(f"Fetched {min(start, total)}/{total}")

        if start >= total:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)


def safe_filename(arxiv_id):
    return re.sub(r"[^\w.\-]", "_", arxiv_id) + ".pdf"


def download_pdf(pdf_url, dest_path, session):
    if os.path.exists(dest_path):
        return "skipped (already exists)"
    try:
        r = session.get(pdf_url, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return "downloaded"
    except Exception as e:
        print(f"  ! failed: {pdf_url} ({e})")
        return "failed"


def main(download_pdfs=True, limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()

    with open(METADATA_FILE, "w", encoding="utf-8") as meta_out:
        count = 0
        for paper in fetch_all_metadata(query):
            meta_out.write(json.dumps(paper, ensure_ascii=False) + "\n")
            count += 1

            if download_pdfs and paper["pdf_url"]:
                dest = os.path.join(OUTPUT_DIR, safe_filename(paper["id"]))
                status = download_pdf(paper["pdf_url"], dest, session)
                print(f"[{count}] {paper['id']} - {status}")
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)
            else:
                print(f"[{count}] {paper['id']} - metadata only")

            if limit and count >= limit:
                break

    print(f"\nDone. Metadata saved to {METADATA_FILE}, PDFs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    # Test first with a small limit, then rerun with limit=None for the full 3060.
    # Set download_pdfs=False if you only want metadata/abstracts (much faster, no storage cost).
    main(download_pdfs=True, limit=None)
