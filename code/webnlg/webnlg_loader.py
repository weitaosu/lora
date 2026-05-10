"""Download and parse WebNLG (English) from the official GitLab corpus.

Avoids the Hugging Face `datasets` script loader (which is removed in
datasets>=4.0 and triggers Windows MAX_PATH errors when extracting).

Two versions supported:
- 'v3.0_en' (default): release_v3.0/en/, larger test set (3,934 entries)
- 'v2.1':              release_v2.1/xml/, the WebNLG 2017 challenge data the
                       LoRA paper used (1,600 test entries / 4,222 lex)

Usage:
    from webnlg_loader import load_webnlg
    train, dev, test = load_webnlg(data_dir="../../data/webnlg/raw", version="v2.1")

Each split is a list of dicts:
    {"src": "subj : pred : obj | ...", "refs": [str, ...], "category": "Airport"}
"""
from __future__ import annotations
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

WEBNLG_URL = (
    "https://gitlab.com/shimorina/webnlg-dataset/-/archive/"
    "587fa698bec705efbefe72a235a6019c2b9b8b6c/"
    "webnlg-dataset-587fa698bec705efbefe72a235a6019c2b9b8b6c.zip"
)
ARCHIVE_NAME = "webnlg-dataset-587fa698bec705efbefe72a235a6019c2b9b8b6c"

# Map a public version label to (zip-internal prefix, on-disk subfolder).
_VERSIONS = {
    "v3.0_en": ("release_v3.0/en/",  "v3.0_en"),
    "v2.1":    ("release_v2.1/xml/", "v2.1"),
}


def _long(p: Path) -> str:
    """Wrap a Windows path with the \\?\\ prefix so it bypasses MAX_PATH."""
    s = str(p.resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _extract(zip_path: Path, out_root: Path, version: str) -> Path:
    """Extract `release_<version>/.../{train,dev,test}/**/*.xml` from the archive."""
    if version not in _VERSIONS:
        raise ValueError(f"Unknown version {version!r}. Pick one of {list(_VERSIONS)}.")
    inner_prefix, subfolder = _VERSIONS[version]
    target_root = out_root / subfolder
    sentinel = target_root / ".extracted_ok"
    if sentinel.exists():
        return target_root
    target_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{ARCHIVE_NAME}/{inner_prefix}"
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and m.endswith(".xml")]
        if not members:
            raise RuntimeError(f"No XML files found under {prefix} in archive")
        for m in members:
            rel = m[len(prefix):]
            out_path = target_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(_long(out_path), "wb") as dst:
                dst.write(src.read())
    sentinel.write_text("ok")
    print(f"Extracted {len(members)} XML files -> {target_root}")
    return target_root


def _parse_xml_file(path: Path) -> list[dict]:
    rows = []
    with open(_long(path), "rb") as f:
        tree = ET.parse(f)
    for entry in tree.getroot().iter("entry"):
        # Modified triples are the cleaned-up version used by the challenge.
        mts = entry.find("modifiedtripleset")
        if mts is None:
            continue
        triples = []
        for mt in mts.findall("mtriple"):
            t = (mt.text or "").strip()
            if t:
                triples.append(t)
        if not triples:
            continue
        src = " | ".join(t.replace(" | ", " : ") for t in triples)
        refs = []
        for lex in entry.findall("lex"):
            t = (lex.text or "").strip()
            if t:
                refs.append(t)
        if not refs:
            continue
        rows.append({
            "src": src,
            "refs": refs,
            "category": entry.attrib.get("category", ""),
        })
    return rows


def _load_split(split_dir: Path) -> list[dict]:
    rows = []
    if not split_dir.exists():
        return rows
    for xml_path in sorted(split_dir.rglob("*.xml")):
        rows.extend(_parse_xml_file(xml_path))
    return rows


def load_webnlg(
    data_dir: str | Path = "data/webnlg/raw",
    version: str = "v3.0_en",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (train, dev, test) lists of {'src', 'refs', 'category'} dicts.

    Downloads + extracts on first call; reuses cached files afterwards.

    version:
        'v3.0_en' (default) -- release_v3.0/en/, larger eval set
        'v2.1'              -- release_v2.1/xml/, the WebNLG 2017 challenge
                               data used by the LoRA paper

    For training, expand each row into one example per reference text yourself
    (e.g. [{'src': r['src'], 'tgt': t} for r in train for t in r['refs']]).
    """
    data_dir = Path(data_dir)
    zip_path = data_dir / "webnlg-v3.zip"
    _download(WEBNLG_URL, zip_path)
    root = _extract(zip_path, data_dir, version)
    train = _load_split(root / "train")
    dev   = _load_split(root / "dev")
    test  = _load_split(root / "test")
    return train, dev, test


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../data/webnlg/raw")
    ver = sys.argv[2] if len(sys.argv) > 2 else "v3.0_en"
    tr, dv, te = load_webnlg(out, version=ver)
    print(f"version={ver}  train={len(tr)}  dev={len(dv)}  test={len(te)}")
    print("Sample:", tr[0])
