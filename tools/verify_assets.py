"""Verify companion assets against the source release's SHA256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "assets/checkpoints.json"
    records = json.loads(manifest.read_text())
    for name, record in records.items():
        path = args.root / record["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Extract the companion model-assets ZIP here first: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"SHA256 differs for {name}")
        print(f"PASS {name}")


if __name__ == "__main__":
    main()
