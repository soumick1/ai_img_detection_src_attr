"""Replay one documented curriculum stage from scratch or its own parent."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoints import load_model, save_model
from .data import TrainingImages
from .engine import train_epoch
from .manifest import file_sha256, read_manifest
from .model import build_model
from .samplers import AnchorSampler, CurriculumSampler, NoiseSampler, ProcessingSampler
from .utils import device_from_name, learning_rate, seed_everything, seed_worker, write_json
from .validation import ValidationImages, evaluate, quality_gate

MANIFEST_SHA = "99d8b47c5ec2f26e9d962ec869804792f53f827f1ae91fda1975d2e33adcb18a"


def run(config_path, images, manifest, output, parent=None, device_name="auto", workers=None):
    config_path, output = Path(config_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh run directory.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if file_sha256(manifest) != MANIFEST_SHA:
        raise ValueError("Use the supplied, fixed original split manifest.")
    rows = read_manifest(manifest)
    training_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if len(training_rows) != 12000 or len(validation_rows) != 3000:
        raise ValueError("Expected the original 12,000/3,000 split.")
    if [sum(r["label"] == c for r in training_rows) for c in range(3)] != [4000] * 3:
        raise ValueError("Original training class counts changed.")
    processing_path = config_path.parent.parent / config["processing_config"]
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    if config["parent_epochs"]:
        if parent is None:
            raise ValueError("This continuation requires its own preceding-stage checkpoint.")
        model, checkpoint = load_model(parent)
        if checkpoint["metadata"]["training_epochs"] != config["parent_epochs"]:
            raise ValueError("Parent training-epoch count differs from the selected curriculum.")
    else:
        if parent is not None:
            raise ValueError("The scratch stage cannot load pretrained weights.")
        model = build_model()
    device = device_from_name(device_name)
    model.to(device)
    dataset = TrainingImages(images, training_rows, config, processing)
    if config["kind"] == "anchor":
        sampler = AnchorSampler(training_rows, 64, config["seed"])
    elif config["kind"] == "broad":
        sampler = ProcessingSampler(len(training_rows), 64, config["seed"])
    elif config["kind"] == "noise":
        sampler = NoiseSampler(len(training_rows), 64, config["seed"])
    elif config["kind"] == "curriculum":
        sampler = CurriculumSampler(training_rows, config["bilinear_quota"], 64, config["seed"])
    else:
        raise ValueError("Unknown curriculum stage kind.")
    count_workers = config["workers"] if workers is None else workers
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=count_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    history, started = [], time.perf_counter()
    counters = {
        k: int(v) for k, v in model.state_dict().items() if k.endswith("num_batches_tracked")
    }
    for epoch in range(config["epochs"]):
        augmentation_epoch = config["augmentation_epoch_start"] + epoch
        sampler.set_epoch(augmentation_epoch)
        generator.manual_seed(config["seed"] + 10000 + augmentation_epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, epoch)
        loss, accuracy, steps = train_epoch(model, loader, optimizer, scaler, device, config)
        if steps != 188 or any(
            int(model.state_dict()[k]) != value + 188 * (epoch + 1) for k, value in counters.items()
        ):
            raise RuntimeError("Physical batch or BatchNorm update count changed.")
        history.append(
            {
                "epoch": epoch + 1,
                "augmentation_epoch": augmentation_epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps": steps,
            }
        )
        write_json(output / "history.json", history)
        print(f"Epoch {epoch + 1}/{config['epochs']} | train {accuracy:.4%}", flush=True)
    metadata = {
        "training_epochs": config["parent_epochs"] + config["epochs"],
        "config": config,
        "manifest_sha256": MANIFEST_SHA,
        "parent_sha256": file_sha256(parent) if parent else None,
        "initialization": "own_provided_data_scratch_lineage",
        "external_pretraining": False,
        "training_images": 12000,
        "validation_images": 3000,
    }
    saved = output / "final_model.pt"
    save_model(saved, model, metadata)
    # Evaluate the serialized state, not a different in-memory training endpoint.
    model, _ = load_model(saved)
    model.to(device)
    validation = ValidationImages(images, validation_rows)
    valid_loader = DataLoader(
        validation,
        batch_size=32,
        shuffle=False,
        num_workers=config["validation_workers"] if workers is None else workers,
        generator=torch.Generator().manual_seed(config["seed"]),
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, valid_loader, validation_rows, device, config["amp"])
    result = {
        "metrics": metrics,
        "gate": quality_gate(metrics),
        "checkpoint_sha256": file_sha256(saved),
        "completed_epochs": config["epochs"],
        "training_epochs": metadata["training_epochs"],
        "parent_sha256": metadata["parent_sha256"],
        "manifest_sha256": MANIFEST_SHA,
        "config": config,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "hidden_training_images": 0,
        "exact_resume_supported": False,
    }
    write_json(output / "evaluation.json", result)
    print(json.dumps(result["gate"]), flush=True)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", default="data/train")
    parser.add_argument("--manifest", default="configs/manifest.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    run(
        args.config, args.images, args.manifest, args.output, args.parent, args.device, args.workers
    )


if __name__ == "__main__":
    main()
