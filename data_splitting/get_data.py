"""Get the WISDM Actitracker v2.0 data into data/raw/wisdm/ (never committed to git).

    # on fep the archive is already in the project folder
    python data_splitting/get_data.py --archive ~/AITDM_Project/WISDM_at_latest.tar.gz

    # or download it from the WISDM Lab first (Fordham University)
    python data_splitting/get_data.py --download

    python data_splitting/get_data.py --list      # what is installed

Extracts only what the pipeline uses (flattened into data/raw/wisdm/):
WISDM_at_v2.0_raw.txt (labelled accelerometer data), WISDM_at_v2.0_demographics.txt,
readme.txt and the *_about.txt files. Next: data_splitting/prepare_wisdm.py.
Please cite the WISDM papers listed in readme.txt when you use the data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "raw" / "wisdm"
URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_at_latest.tar.gz"
WANTED = ("WISDM_at_v2.0_raw.txt", "WISDM_at_v2.0_demographics.txt", "readme.txt")


def wanted(name: str) -> bool:
    base = Path(name).name
    return base in WANTED or base.endswith("_about.txt")


def install(archive: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    found = []
    with tarfile.open(archive) as tar:
        for member in tar:
            # regular files only, written under their base name (no paths from the archive)
            if member.isfile() and wanted(member.name) and "__MACOSX" not in member.name:
                src = tar.extractfile(member)
                with open(out / Path(member.name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found.append(Path(member.name).name)
    if "WISDM_at_v2.0_raw.txt" not in found:
        sys.exit(f"ERROR: no WISDM_at_v2.0_raw.txt in {archive}")
    print(f"Installed {', '.join(sorted(found))} in {out}")
    print("Next: python data_splitting/prepare_wisdm.py --input "
          f"{(out / 'WISDM_at_v2.0_raw.txt').relative_to(REPO_ROOT)} --out data/wisdm")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", type=Path, help="WISDM_at_latest.tar.gz you already have")
    src.add_argument("--download", action="store_true", help="download the archive first")
    src.add_argument("--list", action="store_true", help="show what is installed")
    p.add_argument("--url", default=URL, help="download address")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    if args.list:
        files = sorted(args.out.glob("*")) if args.out.exists() else []
        print("\n".join(f"  {f.name:<34} {f.stat().st_size / 1e6:8.1f} MB" for f in files) or "  nothing installed")
        return
    if args.archive:
        archive = args.archive.expanduser()
        if not archive.exists():
            sys.exit(f"ERROR: {archive} does not exist")
        install(archive, args.out)
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "WISDM_at_latest.tar.gz"
        print(f"Downloading {args.url} ...")
        urllib.request.urlretrieve(args.url, archive)
        install(archive, args.out)


if __name__ == "__main__":
    main()
