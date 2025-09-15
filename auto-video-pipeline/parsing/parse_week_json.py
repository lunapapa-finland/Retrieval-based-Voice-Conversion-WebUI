#!/usr/bin/env python3
"""
Parse weekly LLM JSON -> scripts & images using requests + BeautifulSoup only.

- Skips already-processed JSONs via LOG_FILE (basename match).
- Writes scripts to OUTPUT_BASE/<stem>/sections/NN_slug.txt (UTF-8).
- Image handling per section.image_url:
    1) Try direct download: if response Content-Type startswith image/
    2) Else, if ENABLE_HTML_SCRAPE=true, fetch page HTML and extract a real image URL via:
         <meta property="og:image">, <meta name="twitter:image">, or <img class="tv-snapshot-image">
       then download that image.
    3) If all fails, write a .link file with the original URL (keeps pipeline unblocked).

Run from repo root (same as tts_piper):
  python ./parsing/parse_week_json.py            # process all new JSONs
  python ./parsing/parse_week_json.py 2025Week38.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

# ---------------- .env loader ----------------
def _clean_val(v: str) -> str:
    v = v.split("#", 1)[0].strip()
    if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
        v = v[1:-1].strip()
    return v

def load_env(env_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = _clean_val(v)
    return cfg

def cfg_get(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)

# ---------------- deps ----------------
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# ---------------- utils ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\- ]+", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "untitled"

def is_image_content_type(ct: Optional[str]) -> bool:
    return bool(ct and ct.lower().startswith("image/"))

def guess_ext_from_content_type(ct: str) -> str:
    if not ct:
        return ".bin"
    ct = ct.lower().split(";", 1)[0].strip()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
    }
    return mapping.get(ct, ".bin")

def write_link(url: str, dest_base: Path) -> Path:
    out = dest_base.with_suffix(".link")
    out.write_text(url.strip() + "\n", encoding="utf-8")
    return out

# ---------------- HTTP helpers ----------------
UA_DEFAULT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

def session_from_cfg(ua: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ua or UA_DEFAULT,
        "Accept": "*/*",
        "Referer": "https://www.tradingview.com/",
    })
    return s

def fetch_with_retries(sess: requests.Session, url: str, timeout: int, retries: int, stream: bool = False):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = sess.get(url, timeout=timeout, allow_redirects=True, stream=stream)
            return r
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    if last_exc:
        raise last_exc

# ---------------- HTML → image resolution (from your extract_outline.py approach) ----------------
def resolve_snapshot_from_html(html_text: str, base_url: str = "") -> Optional[str]:
    if not HAVE_BS4:
        return None
    # Prefer lxml if available; fall back to html.parser
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")

    # 1) <meta property="og:image" content="...">
    tag = soup.find("meta", {"property": "og:image"})
    if tag and tag.get("content"):
        return urljoin(base_url, tag["content"])

    # 2) <meta name="twitter:image" content="...">
    tag = soup.find("meta", {"name": "twitter:image"})
    if tag and tag.get("content"):
        return urljoin(base_url, tag["content"])

    # 3) TradingView snapshot <img class="tv-snapshot-image" src="...">
    img = soup.find("img", {"class": re.compile(r"tv-snapshot-image")})
    if img and img.get("src"):
        return urljoin(base_url, img["src"])

    # 4) Fallback: first <img>
    img = soup.find("img")
    if img and img.get("src"):
        return urljoin(base_url, img["src"])

    return None

def download_bytes_to(dest_base: Path, content: bytes, content_type: str) -> Path:
    ext = guess_ext_from_content_type(content_type or "image/png")
    out_path = dest_base.with_suffix(ext)
    out_path.write_bytes(content)
    return out_path

def try_direct_image(sess: requests.Session, url: str, timeout: int, retries: int, dest_base: Path) -> Optional[Path]:
    r = fetch_with_retries(sess, url, timeout=timeout, retries=retries, stream=True)
    ct = r.headers.get("Content-Type", "")
    if r.status_code == 200 and is_image_content_type(ct):
        out_path = dest_base.with_suffix(guess_ext_from_content_type(ct))
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return out_path
    return None

def try_html_then_image(sess: requests.Session, url: str, timeout: int, retries: int, dest_base: Path) -> Optional[Path]:
    """Fetch HTML page, resolve a snapshot image URL, then download it."""
    r = fetch_with_retries(sess, url, timeout=timeout, retries=retries, stream=False)
    ct = (r.headers.get("Content-Type") or "").lower()
    text_like = ("text/html" in ct) or ("application/xhtml" in ct) or ("text/plain" in ct)
    if not text_like and not r.text:
        # If it's not clearly HTML and there's no text, nothing to parse
        return None

    snap_url = resolve_snapshot_from_html(r.text, base_url=url)
    if not snap_url:
        return None

    # Download the resolved snapshot (should be image/*)
    r2 = fetch_with_retries(sess, snap_url, timeout=timeout, retries=retries, stream=True)
    ct2 = r2.headers.get("Content-Type", "")
    if r2.status_code == 200 and is_image_content_type(ct2):
        out_path = dest_base.with_suffix(guess_ext_from_content_type(ct2))
        with open(out_path, "wb") as f:
            for chunk in r2.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return out_path
    return None

# ---------------- core ----------------
def process_json_file(json_path: Path, cfg: Dict[str, str], repo_root: Path) -> bool:
    """Returns True if processed; False on failure."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERR] cannot parse JSON: {json_path.name}: {e}")
        return False

    if not isinstance(data, dict) or "sections" not in data or not isinstance(data["sections"], list):
        print(f"[ERR] bad shape (need object with 'sections' array): {json_path.name}")
        return False

    # Paths from config
    out_base_rel = cfg_get(cfg, "OUTPUT_BASE", "data/step1_scripts")
    sections_sub = cfg_get(cfg, "SECTIONS_SUBDIR", "sections")
    images_sub = cfg_get(cfg, "IMAGES_SUBDIR", "image")

    timeout = int(cfg_get(cfg, "TIMEOUT_SEC", "20"))
    retries = int(cfg_get(cfg, "RETRIES", "2"))
    ua = cfg_get(cfg, "USER_AGENT", UA_DEFAULT)
    enable_html = cfg_get(cfg, "ENABLE_HTML_SCRAPE", "true").lower() == "true"

    # Output root for this JSON
    stem = json_path.stem  # e.g., 2025Week38
    out_root = (repo_root / out_base_rel / stem).resolve()
    sections_dir = out_root / sections_sub
    images_dir = out_root / images_sub
    ensure_dir(sections_dir)
    ensure_dir(images_dir)

    # Shared session
    if not HAVE_REQUESTS:
        print("[ERR] requests not installed.")
        return False
    sess = session_from_cfg(ua)

    count_written = 0
    for idx, sec in enumerate(data["sections"], start=1):
        if not isinstance(sec, dict):
            print(f"[WARN] skip non-object section at index {idx}")
            continue

        title = (sec.get("title") or "").strip()
        slug = (sec.get("slug") or "").strip() or slugify(title or f"section-{idx}")
        script = sec.get("script", "")
        image_url = (sec.get("image_url") or "").strip() or None

        # 1) Script
        script_name = f"{idx:02d}_{slug}.txt"
        script_path = sections_dir / script_name
        with script_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        count_written += 1

        # 2) Image (direct → html-scrape fallback → .link)
        if image_url:
            base = images_dir / f"{idx:02d}_{slug}"

            # Try direct image
            saved = None
            try:
                saved = try_direct_image(sess, image_url, timeout, retries, base)
            except Exception as e:
                # continue to HTML scrape
                saved = None

            # If not a direct image, try HTML resolve
            if not saved and enable_html and HAVE_BS4:
                try:
                    saved = try_html_then_image(sess, image_url, timeout, retries, base)
                except Exception as e:
                    saved = None

            # If still nothing, write .link
            if not saved:
                write_link(image_url, base)

    print(f"[OK] {json_path.name}: wrote {count_written} script file(s) to {out_root.relative_to(repo_root)}")
    return True

def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    env_path = repo_root / "config" / "parsing.env"
    cfg = load_env(env_path)

    # Input dirs
    json_dir = repo_root / cfg_get(cfg, "JSON_INPUT_DIR", "data/step0_json")
    alt_dir = cfg_get(cfg, "JSON_INPUT_DIR_ALT", "").strip()
    json_dir_alt = (repo_root / alt_dir) if alt_dir else None

    # Log
    log_rel = cfg_get(cfg, "LOG_FILE", "data/step1_scripts/parsed.log")
    log_path = (repo_root / log_rel).resolve()
    processed = set()
    if log_path.exists():
        processed = {s.strip() for s in log_path.read_text(encoding="utf-8").splitlines() if s.strip()}

    # Gather targets
    args = sys.argv[1:]
    targets: List[Path] = []
    if args:
        for name in args:
            p = Path(name)
            if not p.is_absolute():
                cand = json_dir / p.name
                if not cand.exists() and json_dir_alt and (json_dir_alt / p.name).exists():
                    cand = json_dir_alt / p.name
                p = cand
            targets.append(p)
    else:
        if json_dir.exists():
            targets.extend(sorted(json_dir.glob("*.json")))
        if json_dir_alt and json_dir_alt.exists():
            targets.extend(sorted(json_dir_alt.glob("*.json")))

    if not targets:
        print("[INFO] No input JSON files found.")
        return 0

    # De-dup, skip already processed (by basename)
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in targets:
        if not p.exists():
            print(f"[WARN] missing: {p}")
            continue
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        if key in processed:
            print(f"[skip] already parsed: {key}")
            continue
        uniq.append(p)

    if not uniq:
        print("[INFO] Nothing to do (all inputs already parsed).")
        return 0

    any_ok = False
    for p in uniq:
        ok = process_json_file(p, cfg, repo_root)
        if ok:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(p.name + "\n")
            any_ok = True

    if not any_ok:
        print("[INFO] No files processed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
