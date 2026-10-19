"""One-view supervised optimization with sample-weighted tail batches."""

import math

import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    count, correct, total_loss, steps = 0, 0, 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=config["amp"] and device.type == "cuda",
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(
                logits, labels, label_smoothing=config["label_smoothing"]
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    if count != len(loader.dataset) or steps != math.ceil(count / config["batch_size"]):
        raise RuntimeError("An epoch did not visit every original training row exactly once.")
    return total_loss / count, correct / count, steps
