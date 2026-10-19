"""Run paired final branches on three dedicated GPUs and select on labels."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=3, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 3:
        raise ValueError("Supply three different dedicated GPU IDs.")
    if args.output.exists():
        raise FileExistsError("Use a fresh ablation output directory.")
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    output.mkdir(parents=True)
    children, streams, runs = [], [], []
    try:
        for dose, gpu in zip((25, 50, 75), args.gpus):
            run = output / f"post_{dose}"
            runs.append(run)
            stream = (output / f"post_{dose}.log").open("w", encoding="utf-8")
            streams.append(stream)
            command = [
                sys.executable,
                "-m",
                "ai_attribution.train",
                "--config",
                str(root / f"configs/ablation_post_{dose}.json"),
                "--parent",
                str(args.parent.resolve()),
                "--images",
                str(args.images.resolve()),
                "--manifest",
                str(root / "configs/manifest.csv"),
                "--output",
                str(run),
                "--device",
                "cuda:0",
            ]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            children.append(
                subprocess.Popen(
                    command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT
                )
            )
            print(f"Started post={dose}% on GPU {gpu}; log {stream.name}", flush=True)
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(f"A branch failed; check logs. Exit codes: {codes}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_attribution.select",
                "--runs",
                *map(str, runs),
                "--output",
                str(output / "selection.json"),
            ],
            cwd=root,
            check=True,
        )
        print(f"Comparison complete: {output / 'selection.json'}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
