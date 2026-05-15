#!/usr/bin/env python3
"""
download_images.py
------------------
Downloads correctly-matched, free-licence images for every item in CATALOGUE.

Sources (tried in order):
  1. Pixabay API   – CC0, no attribution required.
                    Requires a FREE key → https://pixabay.com/api/docs/
                    Takes ~30 seconds to get one after registration.
                    Set PIXABAY_API_KEY below or via env var.

  2. Openverse API – CC-licensed images from the WordPress/Creative-Commons
                    open catalogue. No key needed for basic searches.
                    https://openverse.org

Usage:
    pip install requests
    PIXABAY_API_KEY=your_key python download_images.py

    # Without a key: Openverse is used for every item automatically.
    python download_images.py
"""

import os
import sys
import time
import requests

# ── Your Pixabay API key (optional but recommended) ───────────────────────────
# Get one free at https://pixabay.com/api/docs/
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")

# ── Settings ──────────────────────────────────────────────────────────────────
TIMEOUT = 20          # seconds per HTTP request
DELAY   = 0.5         # polite pause between requests

# ── Catalogue ─────────────────────────────────────────────────────────────────
CATALOGUE = [
    ("item_001", "data/cat_1.jpg",    "ginger tabby cat sitting on a sofa",   "cat"),
    ("item_002", "data/cat_2.jpg",    "black cat with green eyes",            "cat"),
    ("item_003", "data/dog_1.jpg",    "golden retriever in a park",           "dog"),
    ("item_004", "data/dog_2.jpg",    "small white poodle wearing a sweater", "dog"),
    ("item_005", "data/car_1.jpg",    "red vintage sports car",               "car"),
    ("item_006", "data/car_2.jpg",    "black SUV on a city street",           "car"),
    ("item_007", "data/bike_1.jpg",   "mountain bike with knobby tyres",      "bike"),
    ("item_008", "data/bike_2.jpg",   "road bike in racing colours",          "bike"),
    ("item_009", "data/tshirt_1.jpg", "white cotton t-shirt on hanger",       "tshirt"),
    ("item_010", "data/tshirt_2.jpg", "black graphic band t-shirt",           "tshirt"),
    ("item_011", "data/jeans_1.jpg",  "blue denim straight leg jeans",        "jeans"),
    ("item_012", "data/jeans_2.jpg",  "ripped skinny light wash jeans",       "jeans"),
]

# Fine-tuned search queries per image (beats using the raw caption)
QUERIES = {
    "data/cat_1.jpg":    ("ginger tabby cat",        "ginger tabby cat"),
    "data/cat_2.jpg":    ("black cat",               "black cat"),
    "data/dog_1.jpg":    ("golden retriever",        "golden retriever dog"),
    "data/dog_2.jpg":    ("white poodle",            "white poodle dog"),
    "data/car_1.jpg":    ("vintage red sports car",  "vintage red sports car"),
    "data/car_2.jpg":    ("black SUV car",           "black SUV automobile"),
    "data/bike_1.jpg":   ("mountain bike",           "mountain bike trail"),
    "data/bike_2.jpg":   ("road racing bicycle",     "road bike cycling"),
    "data/tshirt_1.jpg": ("white t-shirt",           "white t-shirt clothing"),
    "data/tshirt_2.jpg": ("black graphic t-shirt",   "black band t-shirt"),
    "data/jeans_1.jpg":  ("blue denim jeans",        "blue denim jeans"),
    "data/jeans_2.jpg":  ("ripped skinny jeans",     "ripped jeans fashion"),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CatalogueDownloader/2.0 (educational use)"})


# ── Download helper ───────────────────────────────────────────────────────────

def save_image(url: str, dest: str, source_label: str) -> bool:
    """Fetch *url* and write bytes to *dest*. Returns True on success."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "image" not in ct and "octet-stream" not in ct:
            print(f"    ✗ [{source_label}] unexpected content-type: {ct!r}")
            return False
        if len(r.content) < 3_000:
            print(f"    ✗ [{source_label}] suspiciously small ({len(r.content)} bytes) — skipped")
            return False
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"    ✓ {dest}  ({len(r.content) // 1024} KB)  [{source_label}]")
        return True
    except requests.RequestException as exc:
        print(f"    ✗ [{source_label}] {exc}")
        return False


# ── Source 1: Pixabay ─────────────────────────────────────────────────────────

def from_pixabay(query: str, dest: str) -> bool:
    if not PIXABAY_API_KEY:
        return False
    params = {
        "key":        PIXABAY_API_KEY,
        "q":          query,
        "image_type": "photo",
        "safesearch": "true",
        "order":      "popular",
        "per_page":   5,
        "min_width":  400,
    }
    try:
        r = SESSION.get("https://pixabay.com/api/", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            print(f"    – Pixabay: 0 results for {query!r}")
            return False
        url = hits[0]["webformatURL"]
        return save_image(url, dest, f"Pixabay CC0 · {query!r}")
    except Exception as exc:
        print(f"    ✗ Pixabay API error: {exc}")
        return False


# ── Source 2: Openverse (no key needed) ───────────────────────────────────────

def from_openverse(query: str, dest: str) -> bool:
    """
    Openverse is the Creative Commons / WordPress image search.
    No API key required for up to ~100 req/day per IP.
    Filters to CC0 + CC-BY + CC-BY-SA (all freely reusable).
    """
    params = {
        "q":            query,
        "license_type": "commercial",      # CC0, CC-BY, CC-BY-SA
        "media_type":   "image",
        "page_size":    5,
        "mature":       "false",
    }
    try:
        r = SESSION.get(
            "https://api.openverse.org/v1/images/",
            params=params,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            print(f"    – Openverse: 0 results for {query!r}")
            return False
        # Pick the first result that has a direct image URL
        for hit in results:
            url = hit.get("url", "")
            if url and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                licence = hit.get("license", "CC")
                creator = hit.get("creator", "unknown")
                label = f"Openverse · {licence.upper()} · {creator}"
                if save_image(url, dest, label):
                    print(f"    ℹ  Attribution: '{hit.get('title','untitled')}' by {creator} ({licence.upper()})")
                    return True
        # Fallback: try the first result's url regardless of extension
        hit = results[0]
        url = hit.get("url", "")
        if url:
            licence  = hit.get("license", "CC")
            creator  = hit.get("creator", "unknown")
            label    = f"Openverse · {licence.upper()} · {creator}"
            if save_image(url, dest, label):
                print(f"    ℹ  Attribution: '{hit.get('title','untitled')}' by {creator} ({licence.upper()})")
                return True
        print(f"    – Openverse: no usable URL in results for {query!r}")
        return False
    except Exception as exc:
        print(f"    ✗ Openverse error: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 66)
    print("Catalogue image downloader  v3")
    print()
    if PIXABAY_API_KEY:
        masked = "*" * max(0, len(PIXABAY_API_KEY) - 4) + PIXABAY_API_KEY[-4:]
        print(f"  [1] Pixabay API  : key {masked}  (CC0)")
    else:
        print("  [1] Pixabay API  : NO KEY SET — will be skipped")
        print("      → get a free key in ~30 s: https://pixabay.com/api/docs/")
    print("  [2] Openverse    : active (no key needed, CC-licensed images)")
    print()
    print("  Set key with:  PIXABAY_API_KEY=xxxx python download_images.py")
    print("=" * 66)

    counts = {"pixabay": 0, "openverse": 0, "skipped": 0, "failed": 0}

    for item_id, path, caption, _cat in CATALOGUE:
        print(f"\n[{item_id}] {caption}")

        if os.path.exists(path) and os.path.getsize(path) > 3_000:
            print("    – already exists, skipping")
            counts["skipped"] += 1
            continue

        pq, oq = QUERIES.get(path, (caption, caption))

        # 1 — Pixabay
        if from_pixabay(pq, path):
            counts["pixabay"] += 1
            time.sleep(DELAY)
            continue

        # 2 — Openverse
        if not PIXABAY_API_KEY:
            pass   # already obvious
        else:
            print("    ↩  trying Openverse …")
        if from_openverse(oq, path):
            counts["openverse"] += 1
        else:
            print(f"    ✗ all sources exhausted for {path}")
            counts["failed"] += 1

        time.sleep(DELAY)

    total = counts["pixabay"] + counts["openverse"]
    print("\n" + "=" * 66)
    print(f"Downloaded : {total}  "
          f"(Pixabay: {counts['pixabay']}, Openverse: {counts['openverse']})")
    print(f"Skipped    : {counts['skipped']}")
    print(f"Failed     : {counts['failed']}")

    if counts["failed"]:
        print()
        print("Tip: obtain a free Pixabay key and re-run:")
        print("  PIXABAY_API_KEY=<key> python download_images.py")
    else:
        print()
        print("All images saved to ./data/  ✓")
    print("=" * 66)
    return counts["failed"]


if __name__ == "__main__":
    sys.exit(main())