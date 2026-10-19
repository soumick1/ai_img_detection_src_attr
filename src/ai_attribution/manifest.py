"""Dataset bookkeeping independent of PyTorch."""

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_splits(rows, seed=7454, validation_fraction=0.2):
    """Keep exact decoded-pixel duplicates together, stratified by label."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    by_class = defaultdict(list)
    for digest, group in sorted(groups.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(
                "Identical training pixels have conflicting labels; review the training data."
            )
        by_class[next(iter(labels))].append(group)
    if set(by_class) != {0, 1, 2}:
        raise ValueError("Training data must contain classes 0, 1, and 2.")
    rng = random.Random(seed)
    result = []
    for label in (0, 1, 2):
        class_groups = by_class[label]
        if len(class_groups) < 2:
            raise ValueError(f"Class {label} needs at least two distinct images for a holdout.")
        rng.shuffle(class_groups)
        target = max(1, round(sum(map(len, class_groups)) * validation_fraction))
        valid_count = 0
        for index, group in enumerate(class_groups):
            # Leave at least one distinct image in training for every class.
            use_validation = valid_count < target and index < len(class_groups) - 1
            split = "validation" if use_validation else "train"
            if use_validation:
                valid_count += len(group)
            result.extend({**row, "split": split} for row in group)
    return sorted(result, key=lambda row: row["filename"])


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "label", "split", "pixel_sha256"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Manifest is empty or has duplicate filenames.")
    digest_splits = defaultdict(set)
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1, 2) or row["split"] not in ("train", "validation"):
            raise ValueError("Manifest has invalid labels or split names.")
        if (
            Path(row["filename"]).name != row["filename"]
            or "/" in row["filename"]
            or "\\" in row["filename"]
        ):
            raise ValueError("Manifest filenames must not contain paths.")
        digest_splits[row["pixel_sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in digest_splits.values()):
        raise ValueError("Exact training duplicates cross the validation boundary.")
    counts = Counter((row["split"], row["label"]) for row in rows)
    if any(counts[(split, label)] == 0 for split in ("train", "validation") for label in (0, 1, 2)):
        raise ValueError("Every split must contain all classes.")
    return rows
