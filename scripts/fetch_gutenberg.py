#!/usr/bin/env python3
"""Download the top-N most popular English books from Project Gutenberg.

Pure stdlib — no pip install needed. Idempotent: already-downloaded books are
skipped. Strips Gutenberg header/footer boilerplate. Saves one `{id}.txt` per
book into `testdata/gutenberg/`.

Usage:
    python3 scripts/fetch_gutenberg.py [count]

Default count: 1000.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_COUNT = 1000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "testdata" / "gutenberg"
CATALOG_URL = "https://gutendex.com/books?languages=en&sort=popular&page={page}"
USER_AGENT = "corpust-fetch/0.1 (https://github.com/MartyJRE/corpust)"
POLITE_DELAY_SEC = 0.25
TIMEOUT_SEC = 30
BOILERPLATE_START = re.compile(
    r"\*\*\*\s*START OF (?:THIS |THE )?PROJECT GUTENBERG[^*]*\*\*\*",
    re.IGNORECASE,
)
BOILERPLATE_END = re.compile(
    r"\*\*\*\s*END OF (?:THIS |THE )?PROJECT GUTENBERG",
    re.IGNORECASE,
)


def fetch(url: str, timeout: int = TIMEOUT_SEC, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    assert last_err is not None
    raise last_err


def catalog_page(page: int) -> dict:
    return json.loads(fetch(CATALOG_URL.format(page=page), timeout=60, retries=5))


def plain_text_url(formats: dict[str, str]) -> str | None:
    # Prefer explicit UTF-8 plain text; fall back to any text/plain.
    utf8 = next(
        (url for mime, url in formats.items() if mime.startswith("text/plain") and "utf-8" in mime),
        None,
    )
    if utf8 and not utf8.endswith(".zip"):
        return utf8
    any_text = next(
        (url for mime, url in formats.items() if mime.startswith("text/plain")),
        None,
    )
    if any_text and not any_text.endswith(".zip"):
        return any_text
    return None


def strip_boilerplate(text: str) -> str:
    start = BOILERPLATE_START.search(text)
    if start:
        text = text[start.end():]
    end = BOILERPLATE_END.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()


def main(count: int) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    already = {p.stem for p in OUTPUT_DIR.glob("*.txt")}
    downloaded = len(already)
    if downloaded >= count:
        print(f"Already have {downloaded} books in {OUTPUT_DIR}, nothing to do.")
        return 0

    page = 1
    while downloaded < count:
        try:
            data = catalog_page(page)
        except Exception as e:
            print(f"catalog page {page} failed: {e}", file=sys.stderr, flush=True)
            page += 1
            continue

        for book in data["results"]:
            if downloaded >= count:
                break
            book_id = str(book["id"])
            out_path = OUTPUT_DIR / f"{book_id}.txt"
            if out_path.exists():
                continue

            url = plain_text_url(book.get("formats", {}))
            if not url:
                continue

            try:
                raw = fetch(url)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  skip {book_id}: {e}", file=sys.stderr)
                continue

            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  skip {book_id}: decode error {e}", file=sys.stderr)
                continue

            text = strip_boilerplate(text)
            if len(text) < 500:
                # Suspiciously tiny — likely a dead link or placeholder page.
                print(f"  skip {book_id}: too short after boilerplate strip")
                continue

            out_path.write_text(text, encoding="utf-8")
            downloaded += 1
            title = book.get("title", "").splitlines()[0][:60]
            print(f"  [{downloaded:>4}/{count}] #{book_id}  {title}")
            time.sleep(POLITE_DELAY_SEC)

        if not data.get("next"):
            print("  catalog exhausted", file=sys.stderr)
            break
        page += 1

    print(f"done — {downloaded} books in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT
    sys.exit(main(count))
