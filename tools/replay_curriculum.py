"""Replay the retained six-stage lineage on one exposed GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", default="data/train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError("Use a fresh curriculum output directory.")
    stages = [
        "01_scratch",
        "02_processing",
        "03_noise",
        "04_bilinear",
        "05_strong_noise",
        "06_final",
    ]
    parent = None
    for stage in stages[: args.stop_after]:
        output = args.output.resolve() / stage
        command = [
            sys.executable,
            "-m",
            "ai_attribution.train",
            "--config",
            str(root / f"configs/{stage}.json"),
            "--output",
            str(output),
            "--images",
            str(Path(args.images).resolve()),
            "--device",
            args.device,
            "--manifest",
            str(root / "configs/manifest.csv"),
        ]
        if parent is not None:
            command += ["--parent", str(parent)]
        subprocess.run(command, check=True, cwd=root)
        parent = output / "final_model.pt"
    print(f"Retained checkpoint: {parent}")


if __name__ == "__main__":
    main()
