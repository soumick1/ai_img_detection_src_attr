"""Pillow processing recipe, preserving the original private RNG streams."""

import io
import random

import numpy as np
from PIL import Image, ImageFilter

FAMILIES = ("exact", "variation")
KERNELS = ("lanczos", "bicubic", "bilinear")
PREFILTERS = {"45": "none", "46": "gaussian_blur", "47": "uniform_rgb_dither"}


def recipe(arm):
    if not isinstance(arm, str) or arm not in PREFILTERS:
        raise ValueError("Require arm 45, 46 or 47.")
    return {
        "arm": arm,
        "image_size": 512,
        "train_image_size": 512,
        "input_mode": "native_crop",
        "horizontal_flip": False,
        "family_weights": [1, 1],
        "spatial_crop_size": 508,
        "crop_offset_min": 0,
        "crop_offset_max": 4,
        "exact_kernel": "lanczos",
        "exact_jpeg_quality": 90,
        "exact_jpeg_subsampling": 2,
        "variation_kernels": list(KERNELS),
        "variation_kernel_weights": [2, 1, 1],
        "variation_jpeg_quality_min": 85,
        "variation_jpeg_quality_max": 95,
        "variation_jpeg_subsampling": [2, 1, 0],
        "variation_subsampling_weights": [8, 1, 1],
        "variation_prefilter": PREFILTERS[arm],
        "prefilter_probability": 0.0 if arm == "45" else 0.5,
        "blur_sigma_min": 0.2,
        "blur_sigma_max": 0.5,
        "dither_min_pixel_units": -1.0,
        "dither_max_pixel_units": 1.0,
        "operation_order": [
            "optional_variation_prefilter",
            "crop508",
            "resize512",
            "jpeg",
            "tensor_normalize",
        ],
    }


def check_config_recipe(config):
    expected = recipe(config.get("arm"))
    if any((config.get(key) != value for key, value in expected.items())):
        raise ValueError("The frozen broad-processing data recipe differs.")
    if any(
        (
            key in config
            for key in ("color_jitter", "noise_std_min", "noise_std_max", "random_resized_crop")
        )
    ):
        raise ValueError("No undeclared noise, color or resize augmentation is allowed.")
    if "batch_size" in config and config["batch_size"] != 64:
        raise ValueError("Production physical batches must contain 64 images.")


def augmentation_parameters(family, seed, config):
    check_config_recipe(config)
    if family not in FAMILIES or type(seed) is not int or seed < 0:
        raise ValueError("Require a declared family and a private nonnegative item seed.")
    base = random.Random(seed)
    result = {
        "left": base.randrange(5),
        "top": base.randrange(5),
        "kernel": "lanczos",
        "quality": 90,
        "subsampling": 2,
        "prefilter": "none",
        "blur_sigma": None,
        "dither_seed": None,
    }
    if family == "variation":
        result.update(
            kernel=KERNELS[(0, 0, 1, 2)[base.randrange(4)]],
            quality=base.randint(85, 95),
            subsampling=((2,) * 8 + (1, 0))[base.randrange(10)],
        )
        pre = random.Random(seed ^ 3026077255)
        enabled, sigma, noise_seed = (
            pre.random() < 0.5,
            pre.uniform(0.2, 0.5),
            pre.getrandbits(64),
        )
        if config["arm"] != "45" and enabled:
            result["prefilter"] = PREFILTERS[config["arm"]]
            if config["arm"] == "46":
                result["blur_sigma"] = sigma
            else:
                result["dither_seed"] = noise_seed
    return result


def dither_pixels(image, seed):
    """Independent RGB uniform[-1,1] pixel units, one round/clip to uint8."""
    pixels = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(seed).uniform(-1.0, 1.0, pixels.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8))


def augment(image, family, seed, config):
    if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("Require original decoded RGB512 pixels.")
    p = augmentation_parameters(family, seed, config)
    if p["prefilter"] == "gaussian_blur":
        image = image.filter(ImageFilter.GaussianBlur(p["blur_sigma"]))
    elif p["prefilter"] == "uniform_rgb_dither":
        image = dither_pixels(image, p["dither_seed"])
    image = image.crop((p["left"], p["top"], p["left"] + 508, p["top"] + 508))
    image = image.resize((512, 512), getattr(Image.Resampling, p["kernel"].upper()))
    stream = io.BytesIO()
    image.save(
        stream,
        format="JPEG",
        quality=p["quality"],
        subsampling=p["subsampling"],
        optimize=False,
        progressive=False,
    )
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()
