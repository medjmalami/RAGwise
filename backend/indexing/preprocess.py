#!/usr/bin/env python3
"""
preprocess_arxiv_html.py

Cleans arXiv HTML and inlines images as base64.
- Strips arXiv website chrome (headers, footers, nav).
- Wraps the <article> in a valid HTML document so Docling parses it correctly.
- Replaces relative <img src> with base64 data URIs.
"""

import base64
import mimetypes
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = SCRIPT_DIR / ".." / "arxiv_rag_papers" / "html"
ASSETS_DIR = SCRIPT_DIR / ".." / "arxiv_rag_papers" / "assets"
OUTPUT_DIR = SCRIPT_DIR / ".." / "arxiv_rag_papers" / "preprocessed_html"

SKIP_IMAGE_FILENAMES = {
    "arxiv-logo-primary-light.svg",
    "schmidt-sciences.png",
    "simons-foundation.png",
    "simons-foundation-international.png",
    "smileybones-small.svg",
}

_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".ico": "image/x-icon",
    ".avif": "image/avif",
}


def _mime_for(ext: str) -> str:
    ext = ext.lower()
    if ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    guessed = mimetypes.guess_type(f"a{ext}")
    return guessed[0] if guessed and guessed[0] else "application/octet-stream"


def _find_local_image(assets_dir: Path, src: str) -> Optional[Path]:
    if not src:
        return None
    src = src.split("?", 1)[0].split("#", 1)[0]
    basename = Path(src).name
    if not basename or basename in SKIP_IMAGE_FILENAMES:
        return None

    candidate = assets_dir / basename
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate

    matches = list(assets_dir.glob(f"**/{basename}"))
    for m in matches:
        if m.is_file() and m.stat().st_size > 0:
            return m
    return None


def _process_one(html_path: Path) -> tuple[str, str, str]:
    arxiv_id = html_path.stem
    out_path = OUTPUT_DIR / f"{arxiv_id}.html"

    if out_path.exists():
        return arxiv_id, "skipped", ""

    try:
        html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # 1. Extract only the paper content
        article = soup.find("article", class_="ltx_document")
        if not article:
            article = soup.find("main") or soup

        # 2. Inline images
        assets_dir = ASSETS_DIR / arxiv_id
        for img in article.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src or src.lower().startswith("data:"):
                continue
            if src.startswith(("http://", "https://", "//")):
                continue

            local = _find_local_image(assets_dir, src)
            if local:
                data = local.read_bytes()
                mime = _mime_for(local.suffix)
                b64 = base64.b64encode(data).decode("ascii")
                img["src"] = f"data:{mime};base64,{b64}"
            else:
                img.decompose()

        # 3. Wrap in a full valid HTML document!
        clean_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{arxiv_id}</title></head>
<body>
{str(article)}
</body>
</html>"""

        out_path.write_text(clean_html, encoding="utf-8")
        return arxiv_id, "success", ""

    except Exception as e:
        return arxiv_id, "error", str(e)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_files = sorted(INPUT_DIR.glob("*.html"))

    print(f"Preprocessing {len(html_files)} HTML files...")

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_process_one, p): p for p in html_files}

        success = skipped = errors = 0
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Preprocessing"
        ):
            _, status, err = fut.result()
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"Error: {err}")

    print(f"\nDone. Success: {success}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
