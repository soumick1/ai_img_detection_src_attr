"""Safe model-only checkpoints; no historical wrapper imports are required."""

from pathlib import Path

import torch

from .manifest import file_sha256
from .model import PARAMETER_COUNT, build_model

FORMAT = "rgb_npr_source_attribution_v1"


def load_model(path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None and file_sha256(path) != expected_sha:
        raise ValueError("Checkpoint SHA256 differs from the supplied release record.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError(
            "Use the clean release checkpoint or convert the trusted original checkpoint first."
        )
    model = build_model()
    state, reference = checkpoint["model"], model.state_dict()
    if set(state) != set(reference) or any(
        not torch.is_tensor(value)
        or value.shape != reference[name].shape
        or value.dtype != reference[name].dtype
        or not bool(torch.isfinite(value).all())
        for name, value in state.items()
    ):
        raise ValueError("Checkpoint tensor keys, shapes, dtypes or finite values are invalid.")
    if checkpoint["parameters"] != PARAMETER_COUNT:
        raise ValueError("Unexpected model parameter count.")
    model.load_state_dict(state, strict=True)
    return model.eval(), checkpoint


def save_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "format": FORMAT,
            "parameters": PARAMETER_COUNT,
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def convert(original, output, expected_sha):
    """Repackage metadata while preserving every original learned tensor and buffer."""
    original, output = Path(original), Path(output)
    if file_sha256(original) != expected_sha:
        raise ValueError("Trusted original checkpoint SHA256 differs.")
    native = torch.load(original, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(native["model"], strict=True)
    save_model(
        output,
        model,
        {
            "original_checkpoint_sha256": expected_sha,
            "training_epochs": native["total_own_training_epochs"],
            "training_images": 12000,
            "validation_images": 3000,
            "seed": 7454,
            "initialization": "own_provided_data_scratch_lineage",
            "architecture": "RGB + NPR / ResNet18 / mean-std pooling",
            "external_pretraining": False,
        },
    )
    converted = torch.load(output, map_location="cpu", weights_only=True)
    if any(not torch.equal(v, converted["model"][k]) for k, v in native["model"].items()):
        raise RuntimeError("Checkpoint conversion changed a tensor.")
    return file_sha256(output)
