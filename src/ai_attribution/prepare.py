"""Extract provided archives safely; retain the recorded split and pixel identity."""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image

from .manifest import read_manifest
from .predict import read_names


def members(archive):
    for item in archive.infolist():
        if item.is_dir() or "__MACOSX" in item.filename.split("/"):
            continue
        if item.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(item))) as nested:
                yield from members(nested)
        else:
            yield Path(item.filename.replace("\\", "/")).name, archive.read(item)


def extract(path, destination, expected, verify_pixels):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    found = set()
    with zipfile.ZipFile(path) as archive:
        for name, payload in members(archive):
            if name not in expected:
                continue
            if name in found:
                raise ValueError(f"Repeated image name in the archive: {name}")
            found.add(name)
            with Image.open(io.BytesIO(payload)) as image:
                rgb = image.convert("RGB")
                if rgb.size != (512, 512):
                    raise ValueError(f"Unexpected source geometry: {name}")
                if verify_pixels and hashlib.sha256(rgb.tobytes()).hexdigest() != expected[name]:
                    raise ValueError(f"Original decoded pixels differ: {name}")
            target = destination / name
            if target.exists():
                if target.read_bytes() != payload:
                    raise FileExistsError(
                        f"Refusing to replace a different existing image: {target}"
                    )
            else:
                target.write_bytes(payload)
    if found != set(expected):
        raise ValueError(f"Archive is missing {len(set(expected) - found)} expected images.")
    print(f"Verified {len(found)} images in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-archive", help="train.zip or the development archive containing train.zip"
    )
    parser.add_argument("--test-archive", help="The actual final test_images.zip release")
    parser.add_argument("--output", default="data")
    parser.add_argument("--manifest", default="configs/manifest.csv")
    parser.add_argument("--filenames", default="assets/test_filenames.txt")
    args = parser.parse_args()
    if not args.train_archive and not args.test_archive:
        parser.error("Supply at least one provided archive.")
    if args.train_archive:
        expected = {r["filename"]: r["pixel_sha256"] for r in read_manifest(args.manifest)}
        extract(args.train_archive, Path(args.output) / "train", expected, True)
    if args.test_archive:
        extract(
            args.test_archive,
            Path(args.output) / "test",
            dict.fromkeys(read_names(args.filenames)),
            False,
        )


if __name__ == "__main__":
    main()
