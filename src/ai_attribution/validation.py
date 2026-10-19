"""The recorded 27-view bank, evaluated separately rather than averaged."""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import gaussian, jpeg, normalize

KERNELS = ("lanczos", "bicubic", "bilinear")
CORNERS = ((0, 0), (4, 0), (0, 4), (4, 4))
CONDITIONS = (
    ("clean", "q90_420")
    + tuple(
        f"crop508_{kernel}_q{quality}_left2_top2" for kernel in KERNELS for quality in (85, 90, 95)
    )
    + tuple(f"crop508_lanczos_q90_left{x}_top{y}" for x, y in CORNERS)
    + tuple(
        f"noise_{position}_sigma{sigma}" for position in ("pre", "post") for sigma in (1, 2, 4, 8)
    )
    + ("brightness_0.98", "brightness_1.02", "contrast_0.95", "contrast_1.05")
)
PRIMARY = (
    tuple(f"crop508_{kernel}_q90_left2_top2" for kernel in KERNELS)
    + tuple(f"crop508_lanczos_q90_left{x}_top{y}" for x, y in CORNERS)
    + (
        "noise_pre_sigma1",
        "noise_pre_sigma2",
        "noise_post_sigma1",
        "noise_post_sigma2",
        "noise_pre_sigma4",
        "noise_post_sigma4",
    )
)


def anchor(image):
    return jpeg(image.crop((2, 2, 510, 510)).resize((512, 512), Image.Resampling.LANCZOS))


def render_validation(image, filename, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown validation condition: {condition}")
    if condition == "clean":
        return image.copy()
    if condition == "q90_420":
        return jpeg(image)
    if condition.startswith("crop508_"):
        _, kernel, quality, left, top = condition.split("_")
        x, y = int(left[4:]), int(top[3:])
        return jpeg(
            image.crop((x, y, x + 508, y + 508)).resize(
                (512, 512), getattr(Image.Resampling, kernel.upper())
            ),
            int(quality[1:]),
        )
    if condition.startswith("noise_"):
        _, position, severity = condition.split("_")
        material = "labeled_perturbation_v53_745453\0" + filename + "\0" + condition
        seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")
        if position == "pre":
            return anchor(gaussian(image, int(severity[5:]), seed))
        return gaussian(anchor(image), int(severity[5:]), seed)
    values = np.array(image, dtype=np.float64, copy=True)
    factor = float(condition.split("_")[1])
    if condition.startswith("brightness"):
        values *= factor
    else:
        values = 127.5 + (values - 127.5) * factor
    return anchor(Image.fromarray(np.clip(np.rint(values), 0, 255).astype(np.uint8)))


class ValidationImages(Dataset):
    def __init__(self, directory, rows, conditions=CONDITIONS):
        if not rows or any(row["split"] != "validation" for row in rows):
            raise ValueError("Evaluation requires the original labeled holdout.")
        self.directory, self.rows, self.conditions = (
            Path(directory).resolve(),
            rows,
            tuple(conditions),
        )
        if any(name not in CONDITIONS for name in conditions):
            raise ValueError("Undeclared evaluation condition.")

    def __len__(self):
        return len(self.rows) * len(self.conditions)

    def __getitem__(self, index):
        condition_index, row_index = divmod(index, len(self.rows))
        row = self.rows[row_index]
        path = self.directory / row["filename"]
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError("Image path escapes its dataset directory.")
        with Image.open(path) as source:
            image = source.convert("RGB")
            if hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]:
                raise ValueError("Original holdout pixels changed.")
            view = render_validation(image, row["filename"], self.conditions[condition_index])
            return normalize(view), row["label"], row_index, condition_index


@torch.inference_mode()
def evaluate(model, loader, rows, device, amp=True):
    model.eval()
    conditions = loader.dataset.conditions
    matrices = torch.zeros(len(conditions), 3, 3, dtype=torch.long)
    losses = [[] for _ in conditions]
    predictions = [[] for _ in conditions]
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cursor = 0
    for images, labels, indices, condition_ids in loader:
        if (condition_ids * len(rows) + indices).tolist() != list(
            range(cursor, cursor + len(labels))
        ):
            raise RuntimeError("Evaluation requires condition-major, unshuffled row order.")
        if labels.tolist() != [rows[i]["label"] for i in indices.tolist()]:
            raise RuntimeError("Original filename/label pairing changed.")
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(images.to(device, non_blocking=True))
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Non-finite evaluation logits.")
        chosen = logits.argmax(1).cpu()
        ce = (
            torch.nn.functional.cross_entropy(logits.float(), labels.to(device), reduction="none")
            .cpu()
            .double()
        )
        for condition in condition_ids.unique().tolist():
            mask = condition_ids == condition
            matrices[condition] += torch.bincount(
                labels[mask] * 3 + chosen[mask], minlength=9
            ).reshape(3, 3)
            losses[condition].extend(ce[mask].tolist())
            predictions[condition].extend(chosen[mask].tolist())
        cursor += len(labels)
    if cursor != len(conditions) * len(rows):
        raise RuntimeError("Incomplete condition coverage.")
    if any(not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError("Evaluation modified model parameters or buffers.")
    metrics = []
    for index, name in enumerate(conditions):
        matrix = matrices[index]
        if int(matrix.sum()) != len(rows):
            raise RuntimeError("Incomplete holdout evaluation.")
        correct = int(matrix.diag().sum())
        metrics.append(
            {
                "condition": name,
                "images": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "loss": math.fsum(losses[index]) / len(rows),
                "confusion_matrix": matrix.tolist(),
                "class_recall": (matrix.diag().double() / matrix.sum(1)).tolist(),
                "prediction_counts": matrix.sum(0).tolist(),
                "error_filenames": [
                    r["filename"] for r, p in zip(rows, predictions[index]) if r["label"] != p
                ],
            }
        )
    return metrics


def selection_score(metrics):
    selected = [m for m in metrics if m["condition"] in PRIMARY]
    if len(selected) != len(PRIMARY):
        raise ValueError("Selection requires all 13 primary conditions.")
    return (
        min(m["correct"] for m in selected),
        sum(m["correct"] for m in selected),
        -math.fsum(m["loss"] for m in selected),
    )


def quality_gate(metrics):
    selection_score(metrics)
    primary = [m for m in metrics if m["condition"] in PRIMARY]
    accuracy = min(m["accuracy"] for m in primary)
    recall = min(r for m in primary for r in m["class_recall"])
    return {
        "passed": accuracy >= 0.98 and recall >= 0.97,
        "minimum_primary_accuracy": accuracy,
        "minimum_primary_class_recall": recall,
    }
