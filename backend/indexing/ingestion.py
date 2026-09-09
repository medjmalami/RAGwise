import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

METADATA_FILE = Path("arxiv_rag_metadata.jsonl")

HTML_DIR = Path("arxiv_rag_papers/html")
ASSETS_DIR = Path("arxiv_rag_papers/assets")

FAILED_FILE = Path("arxiv_rag_papers/html_failed.jsonl")
IMAGE_FAILED_FILE = Path("arxiv_rag_papers/image_failed.jsonl")

BASE_URL = "https://arxiv.org/html"

SLEEP_BETWEEN_PAPERS = 3
SLEEP_BETWEEN_IMAGES = 0.5

TIMEOUT = 30

USER_AGENT = "arxiv-rag-research/1.0"


# ============================================================
# Images to ignore
# ============================================================

SKIP_IMAGE_FILENAMES = {
    "arxiv-logo-primary-light.svg",
    "schmidt-sciences.png",
    "simons-foundation.png",
    "simons-foundation-international.png",
    "smileybones-small.svg",
}


# ============================================================
# Files
# ============================================================


def load_papers():
    """Yield papers from the metadata JSONL."""

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            yield json.loads(line)


# ============================================================
# Image helpers
# ============================================================


def is_data_uri(src: str) -> bool:
    """Return True if src is a data URI."""

    return src.lower().startswith("data:")


def extension_from_mime(mime_type: str) -> str:
    """Convert MIME type to a file extension."""

    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/x-icon": ".ico",
        "image/vnd.microsoft.icon": ".ico",
        "image/avif": ".avif",
    }

    mime_type = mime_type.lower().strip()

    if mime_type in mapping:
        return mapping[mime_type]

    extension = mimetypes.guess_extension(mime_type)

    if extension:
        return extension

    return ".bin"


def filename_from_url(url: str, index: int) -> str:
    """Get a filename from a normal URL."""

    path = urlparse(url).path

    filename = Path(path).name

    if not filename:
        filename = f"image_{index:04d}"

    return filename


def make_unique_path(path: Path) -> Path:
    """Avoid filename collisions."""

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix

    counter = 1

    while True:
        candidate = path.parent / f"{stem}_{counter}{suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


# ============================================================
# HTML image extraction
# ============================================================


def get_image_sources(html: str, paper_url: str):
    """
    Extract relevant remote image sources from the HTML.

    Ignored:
        - Base64/data URI images
        - Known arXiv/non-paper assets

    The HTML is never modified.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Prefer <main> to avoid arXiv website chrome.
    main = soup.find("main")

    if main is not None:
        image_tags = main.find_all("img")
    else:
        image_tags = soup.find_all("img")

    sources = []

    seen_urls = set()

    for img in image_tags:
        src = img.get("src")

        if not src:
            continue

        src = src.strip()

        if not src:
            continue

        # ----------------------------------------------------
        # Ignore Base64/data URI images.
        # ----------------------------------------------------

        if is_data_uri(src):
            continue

        # ----------------------------------------------------
        # Resolve relative URL.
        # ----------------------------------------------------

        absolute_url = urljoin(
            paper_url,
            src,
        )

        parsed = urlparse(absolute_url)

        if parsed.scheme not in ("http", "https"):
            continue

        # ----------------------------------------------------
        # Determine filename.
        # ----------------------------------------------------

        filename = Path(parsed.path).name

        if not filename:
            continue

        # ----------------------------------------------------
        # Ignore known non-paper assets.
        # ----------------------------------------------------

        if filename in SKIP_IMAGE_FILENAMES:
            continue

        # ----------------------------------------------------
        # Prevent duplicate URLs.
        # ----------------------------------------------------

        if absolute_url in seen_urls:
            continue

        seen_urls.add(absolute_url)

        sources.append(
            {
                "src": src,
                "url": absolute_url,
                "filename": filename,
            }
        )

    return sources


# ============================================================
# Remote images
# ============================================================


def download_remote_image(
    url: str,
    output_dir: Path,
    index: int,
    session: requests.Session,
):
    """
    Download one HTTP/HTTPS image.

    Returns:
        output_path, status

    status:
        "downloaded"
        "skipped"
    """

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    filename = filename_from_url(
        url,
        index,
    )

    output_path = output_dir / filename

    # --------------------------------------------------------
    # Existing image.
    # --------------------------------------------------------

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, "skipped"

    # --------------------------------------------------------
    # Download.
    # --------------------------------------------------------

    response = session.get(
        url,
        timeout=TIMEOUT,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # If URL has no extension, use Content-Type.
    # --------------------------------------------------------

    if not output_path.suffix:
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()

        extension = extension_from_mime(content_type)

        output_path = output_dir / f"image_{index:04d}{extension}"

    output_path = make_unique_path(output_path)

    output_path.write_bytes(response.content)

    return output_path, "downloaded"


# ============================================================
# Paper image processing
# ============================================================


def process_images_for_paper(
    arxiv_id: str,
    html: str,
    session: requests.Session,
):
    """
    Download all relevant paper images.

    Ignored:
        - Base64/data URI images
        - known arXiv/non-paper assets

    Existing images are skipped.

    The HTML is NEVER modified.
    """

    paper_url = f"{BASE_URL}/{arxiv_id}"

    image_sources = get_image_sources(
        html=html,
        paper_url=paper_url,
    )

    assets_dir = ASSETS_DIR / arxiv_id

    assets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(image_sources)

    downloaded = 0
    skipped = 0
    failed = 0

    for index, image in enumerate(
        image_sources,
        start=1,
    ):
        url = image["url"]

        try:
            output_path, status = download_remote_image(
                url=url,
                output_dir=assets_dir,
                index=index,
                session=session,
            )

            if status == "downloaded":
                downloaded += 1

                print(f"      -> image downloaded: {output_path.name}")

            else:
                skipped += 1

                print(f"      -> image exists: {output_path.name}")

        except Exception as exc:
            failed += 1

            print(f"      IMAGE ERROR: {url}")

            print(f"      {type(exc).__name__}: {exc}")

        time.sleep(SLEEP_BETWEEN_IMAGES)

    return {
        "total": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


# ============================================================
# Completeness
# ============================================================


def paper_images_are_complete(
    arxiv_id: str,
    html: str,
):
    """
    Determine whether every relevant remote image already
    exists locally.

    Base64 images are ignored.

    Known non-paper images are ignored.

    Returns:
        True  -> relevant images exist
        False -> at least one relevant image is missing
    """

    paper_url = f"{BASE_URL}/{arxiv_id}"

    image_sources = get_image_sources(
        html=html,
        paper_url=paper_url,
    )

    # --------------------------------------------------------
    # No relevant images.
    #
    # IMPORTANT:
    # This is NOT the same thing as "images already exist".
    # --------------------------------------------------------

    if not image_sources:
        return None

    assets_dir = ASSETS_DIR / arxiv_id

    if not assets_dir.exists():
        return False

    # --------------------------------------------------------
    # Every relevant image must exist.
    # --------------------------------------------------------

    for index, image in enumerate(
        image_sources,
        start=1,
    ):
        url = image["url"]

        filename = filename_from_url(
            url,
            index,
        )

        path = assets_dir / filename

        # Normal filename exists.
        if path.exists() and path.stat().st_size > 0:
            continue

        # URL had no extension, so downloader may have used
        # image_XXXX.<extension>.
        if not Path(filename).suffix:
            possible = list(assets_dir.glob(f"image_{index:04d}.*"))

            if any(p.is_file() and p.stat().st_size > 0 for p in possible):
                continue

        # Missing image.
        return False

    return True


# ============================================================
# Paper processing
# ============================================================


def process_paper(
    arxiv_id: str,
    session: requests.Session,
):
    """
    Process one paper.

    Existing HTML is NEVER modified.

    HTML is downloaded only if it doesn't exist.

    Images are processed independently.
    """

    html_path = HTML_DIR / f"{arxiv_id}.html"

    # --------------------------------------------------------
    # Download HTML if necessary.
    # --------------------------------------------------------

    if not html_path.exists():
        url = f"{BASE_URL}/{arxiv_id}"

        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
                headers={
                    "User-Agent": USER_AGENT,
                },
            )

            if response.status_code == 404:
                return {"status": "html_unavailable"}

            response.raise_for_status()

            html_path.write_text(
                response.text,
                encoding="utf-8",
            )

            print("  -> HTML downloaded")

        except requests.RequestException as exc:
            print(f"  ERROR HTML {arxiv_id}: {exc}")

            return {
                "status": "html_failed",
                "error": str(exc),
            }

    # --------------------------------------------------------
    # Read existing HTML.
    #
    # NEVER write it back.
    # --------------------------------------------------------

    html = html_path.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # Determine image state.
    # --------------------------------------------------------

    image_state = paper_images_are_complete(
        arxiv_id=arxiv_id,
        html=html,
    )

    # --------------------------------------------------------
    # No relevant images.
    # --------------------------------------------------------

    if image_state is None:
        return {"status": "no_images"}

    # --------------------------------------------------------
    # All relevant images already exist.
    # --------------------------------------------------------

    if image_state is True:
        return {"status": "skipped"}

    # --------------------------------------------------------
    # At least one image is missing.
    # --------------------------------------------------------

    stats = process_images_for_paper(
        arxiv_id=arxiv_id,
        html=html,
        session=session,
    )

    # --------------------------------------------------------
    # Verify again after downloading.
    # --------------------------------------------------------

    image_state = paper_images_are_complete(
        arxiv_id=arxiv_id,
        html=html,
    )

    if image_state is not True:
        return {
            "status": "images_failed",
            "image_stats": stats,
        }

    return {
        "status": "complete",
        "image_stats": stats,
    }


# ============================================================
# Main
# ============================================================


def main():

    HTML_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ASSETS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    complete = 0
    skipped = 0
    no_images = 0
    unavailable = 0
    failed = 0
    images_failed = 0

    with (
        FAILED_FILE.open(
            "a",
            encoding="utf-8",
        ) as failed_out,
        IMAGE_FAILED_FILE.open(
            "a",
            encoding="utf-8",
        ) as image_failed_out,
    ):
        for index, paper in enumerate(
            load_papers(),
            start=1,
        ):
            arxiv_id = paper["id"]

            print(f"\n[{index}] {arxiv_id}")

            result = process_paper(
                arxiv_id=arxiv_id,
                session=session,
            )

            status = result["status"]

            # ------------------------------------------------
            # Complete
            # ------------------------------------------------

            if status == "complete":
                complete += 1

                stats = result["image_stats"]

                print(
                    f"  -> complete "
                    f"(total={stats['total']}, "
                    f"downloaded={stats['downloaded']}, "
                    f"existing={stats['skipped']})"
                )

            # ------------------------------------------------
            # Already complete
            # ------------------------------------------------

            elif status == "skipped":
                skipped += 1

                print("  -> skipped (HTML + images already exist)")

            # ------------------------------------------------
            # HTML exists but contains no relevant images
            # ------------------------------------------------

            elif status == "no_images":
                no_images += 1

                print("  -> no relevant images to download")

            # ------------------------------------------------
            # HTML unavailable
            # ------------------------------------------------

            elif status == "html_unavailable":
                unavailable += 1

                print("  -> HTML unavailable")

                failed_out.write(
                    json.dumps(
                        {
                            "id": arxiv_id,
                            "status": "html_unavailable",
                        }
                    )
                    + "\n"
                )

            # ------------------------------------------------
            # HTML failed
            # ------------------------------------------------

            elif status == "html_failed":
                failed += 1

                print("  -> HTML failed")

                failed_out.write(
                    json.dumps(
                        {
                            "id": arxiv_id,
                            "status": "html_failed",
                            "error": result.get("error"),
                        }
                    )
                    + "\n"
                )

            # ------------------------------------------------
            # Images failed
            # ------------------------------------------------

            elif status == "images_failed":
                images_failed += 1

                stats = result["image_stats"]

                print(f"  -> images incomplete (failed={stats['failed']})")

                image_failed_out.write(
                    json.dumps(
                        {
                            "id": arxiv_id,
                            "status": "images_failed",
                            "image_stats": stats,
                        }
                    )
                    + "\n"
                )

            # ------------------------------------------------
            # Flush logs after every paper.
            # ------------------------------------------------

            failed_out.flush()
            image_failed_out.flush()

            time.sleep(SLEEP_BETWEEN_PAPERS)

    print("\nDone.")

    print(f"Complete:         {complete}")

    print(f"Skipped:          {skipped}")

    print(f"No images:        {no_images}")

    print(f"HTML unavailable: {unavailable}")

    print(f"HTML failed:      {failed}")

    print(f"Images failed:    {images_failed}")


if __name__ == "__main__":
    main()
