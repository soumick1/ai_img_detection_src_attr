"""Evaluate a saved model on the clean holdout or the complete stress bank."""

import argparse

import torch
from torch.utils.data import DataLoader

from .checkpoints import load_model
from .manifest import read_manifest
from .utils import device_from_name, seed_everything, seed_worker, write_json
from .validation import CONDITIONS, ValidationImages, evaluate, quality_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", default="data/train")
    parser.add_argument("--manifest", default="configs/manifest.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=("all",) + CONDITIONS, default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seed_everything()
    model, _ = load_model(args.checkpoint)
    device = device_from_name(args.device)
    model.to(device)
    rows = [r for r in read_manifest(args.manifest) if r["split"] == "validation"]
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    dataset = ValidationImages(args.images, rows, conditions)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(7454),
    )
    metrics = evaluate(model, loader, rows, device)
    write_json(
        args.output,
        {"metrics": metrics, "gate": quality_gate(metrics) if args.condition == "all" else None},
    )
    for metric in metrics:
        print(f"{metric['condition']}: {metric['accuracy']:.4%}")


if __name__ == "__main__":
    main()
