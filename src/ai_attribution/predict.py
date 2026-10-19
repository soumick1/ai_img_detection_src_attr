"""Predict unchanged whole images and write the official CodaBench JSON ZIP."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoints import load_model
from .data import PredictionImages
from .manifest import file_sha256
from .utils import device_from_name, seed_everything, seed_worker, write_json


def read_names(path):
    names = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if (
        not names
        or len(names) != len(set(names))
        or any(Path(n).name != n or "/" in n or "\\" in n for n in names)
    ):
        raise ValueError("Expected unique plain image filenames.")
    return names


def canonical_json(predictions):
    return (json.dumps(predictions, indent=2) + "\n").encode("utf-8")


@torch.inference_mode()
def predict(model, directory, names, device, batch_size=32, workers=4, amp=True):
    model.to(device).eval()
    loader = DataLoader(
        PredictionImages(directory, names),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(7454 + 2000000),
    )
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    output = {}
    for images, filenames in loader:
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(images.to(device, non_blocking=True))
        if logits.shape != (len(filenames), 3) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Unexpected or non-finite classifier output.")
        output.update(zip(filenames, map(int, logits.argmax(1).cpu().tolist())))
    if list(output) != names:
        raise RuntimeError("Prediction filename coverage changed.")
    if any(not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError("Inference modified model state.")
    return output


def validate_mapping(predictions, names):
    if set(predictions) != set(names) or any(
        type(v) is not int or v not in (0, 1, 2) for v in predictions.values()
    ):
        raise ValueError("Prediction keys or integer class labels violate the official format.")


def write_submission(output, predictions, names):
    output = Path(output)
    validate_mapping(predictions, names)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(
            "Choose a new output name; existing predictions are never overwritten."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(predictions)
    output.with_suffix(".json").write_bytes(payload)
    # Fixed ZIP metadata makes this clean release's container deterministic.
    info = zipfile.ZipInfo("predictions.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--filenames", default="assets/test_filenames.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-reverse", action="store_true")
    parser.add_argument(
        "--reference-json",
        help="Require exact filename-keyed agreement with a recorded prediction JSON.",
    )
    args = parser.parse_args()
    names = read_names(args.filenames)
    seed_everything()
    model, checkpoint = load_model(args.checkpoint, args.expected_sha)
    device = device_from_name(args.device)
    predictions = predict(model, args.images, names, device, args.batch_size, args.workers)
    if args.verify_reverse:
        reverse = predict(
            model, args.images, list(reversed(names)), device, max(1, args.batch_size // 2), 0
        )
        if reverse != predictions:
            raise RuntimeError("Reverse-order, half-batch replay changed predictions.")
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
        if reference != predictions:
            raise RuntimeError("Predictions differ from the recorded final submission.")
    digest = write_submission(args.output, predictions, names)
    write_json(
        Path(args.output).with_suffix(".meta.json"),
        {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "prediction_json_sha256": digest,
            "predictions": len(names),
            "prediction_counts": dict(Counter(predictions.values())),
            "model_count": 1,
            "views_per_image": 1,
            "preprocessing": "RGB512, ToTensor, mean/std .5",
            "model_state_unchanged": True,
            "reverse_replay_verified": args.verify_reverse,
            "reference_verified": bool(args.reference_json),
            "training_updates": 0,
            "test_time_augmentation": False,
            "uploaded": False,
        },
    )
    print(f"Valid submission: {len(names)} predictions; JSON SHA256 {digest}")


if __name__ == "__main__":
    main()
