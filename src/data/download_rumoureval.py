from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

from src.common.io import ensure_dir


DEFAULT_URL = "https://ndownloader.figshare.com/files/16188500"
DEFAULT_MD5 = "85b37dc70106c2b9f584e63164fcbb92"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out_dir: Path, expected_md5: Optional[str] = DEFAULT_MD5) -> Path:
    ensure_dir(out_dir)
    output = out_dir / "rumoureval2019.tar.bz2"
    if output.exists():
        if expected_md5 and md5sum(output) != expected_md5:
            raise ValueError(f"Existing file has unexpected md5: {output}")
        return output

    with urllib.request.urlopen(url) as response, output.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    if expected_md5 and md5sum(output) != expected_md5:
        raise ValueError(f"Downloaded file has unexpected md5: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official RumourEval 2019 data.")
    parser.add_argument("--out-dir", default="data/raw/rumoureval2019")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    path = download(args.url, Path(args.out_dir), None if args.skip_md5 else DEFAULT_MD5)
    print(path)


if __name__ == "__main__":
    main()
