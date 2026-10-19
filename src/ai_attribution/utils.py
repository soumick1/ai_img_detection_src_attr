"""Determinism, serialization and small shared utilities."""

import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


def seed_everything(seed=7454):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def learning_rate(config, epoch):
    schedule = config.get("schedule_epochs", config["epochs"])
    warmup = min(config["warmup_epochs"], max(0, schedule - 1))
    if epoch < warmup:
        return config["learning_rate"] * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, schedule - warmup - 1)
    return config["min_learning_rate"] + 0.5 * (
        config["learning_rate"] - config["min_learning_rate"]
    ) * (1 + math.cos(math.pi * progress))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def device_from_name(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for CPU inference.")
    return device
