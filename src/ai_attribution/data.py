"""Labeled curriculum data and unchanged-image inference preprocessing."""

import hashlib
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .processing import augment, augmentation_parameters
from .samplers import _subseed


def normalize(image):
    """Exactly ToTensor followed by channel-wise mean=.5, standard deviation=.5."""
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("Expected one native RGB 512 x 512 image.")
    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div(255)
    return tensor.sub(0.5).div(0.5)


def jpeg(image, quality=90, subsampling=2):
    stream = io.BytesIO()
    image.save(
        stream,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
        progressive=False,
    )
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()


def gaussian(image, sigma, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    pixels = np.array(image, dtype=np.float64, copy=True)
    pixels += rng.normal(0.0, sigma, size=pixels.shape)
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))


def noise_parameters(seed, config):
    position = (
        "pre"
        if random.Random(_subseed(seed, "position")).random() < config["pre_probability"]
        else "post"
    )
    maximum = config[f"{position}_sigma_max"]
    endpoint = False
    if config["use_endpoint_mixture"]:
        endpoint = (
            random.Random(_subseed(seed, "strong_endpoint_v68")).random()
            < config[f"{position}_endpoint_probability"]
        )
    sigma = 4.0 if endpoint else maximum * random.Random(_subseed(seed, "severity")).random()
    return position, sigma, _subseed(seed, "pixels")


def render_training(image, key, config, processing):
    """Paired view construction; sample metadata never enters the network."""
    _, family, item_seed, *flags = key
    if config["kind"] == "anchor":
        rng = random.Random(item_seed)
        left, top = rng.randint(0, 4), rng.randint(0, 4)
        return jpeg(
            image.crop((left, top, left + 508, top + 508)).resize(
                (512, 512), Image.Resampling.LANCZOS
            )
        )
    enabled, noise_seed, forced = False, 0, False
    if flags:
        enabled, noise_seed = flags[:2]
        forced = flags[2] if len(flags) == 3 else False
    if forced:
        if enabled:
            raise ValueError("Clean bilinear rendering cannot replace a noisy row.")
        params = augmentation_parameters(family, item_seed, processing)
        left, top = params["left"], params["top"]
        return jpeg(
            image.crop((left, top, left + 508, top + 508)).resize(
                (512, 512), Image.Resampling.BILINEAR
            )
        )
    position, sigma, pixel_seed = (
        noise_parameters(noise_seed, config) if enabled else ("none", 0.0, 0)
    )
    if position == "pre":
        image = gaussian(image, sigma, pixel_seed)
    image = augment(image, family, item_seed, processing)
    if position == "post":
        image = gaussian(image, sigma, pixel_seed)
    return image


class TrainingImages(Dataset):
    def __init__(self, directory, rows, config, processing):
        if not rows or any(row["split"] != "train" for row in rows):
            raise ValueError("Training may only read the original training split.")
        self.directory = Path(directory).resolve()
        self.rows, self.config, self.processing = rows, config, processing

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        row = self.rows[key[0]]
        path = self.directory / row["filename"]
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError("Image path escapes its dataset directory.")
        with Image.open(path) as source:
            image = source.convert("RGB")
            if hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]:
                raise ValueError(f"Original training pixels changed: {row['filename']}")
            return normalize(render_training(image, key, self.config, self.processing)), row[
                "label"
            ]


class PredictionImages(Dataset):
    def __init__(self, directory, names):
        self.directory, self.names = Path(directory), names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.directory / name) as source:
            return normalize(source.convert("RGB")), name
