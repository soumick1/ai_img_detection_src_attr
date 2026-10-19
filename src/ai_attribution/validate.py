"""Validate a CodaBench prediction JSON or single-entry ZIP."""

import argparse
import json
import zipfile
from pathlib import Path

from .predict import read_names, validate_mapping


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate prediction key: {key}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission")
    parser.add_argument("--filenames", default="assets/test_filenames.txt")
    args = parser.parse_args()
    path = Path(args.submission)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["predictions.json"] or archive.testzip() is not None:
                raise ValueError("ZIP must contain only predictions.json at its top level.")
            payload = archive.read("predictions.json")
    else:
        payload = path.read_bytes()
    predictions = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(predictions, dict):
        raise ValueError("The JSON document must be an object.")
    names = read_names(args.filenames)
    validate_mapping(predictions, names)
    print(f"Valid submission: {len(names)} predictions.")


if __name__ == "__main__":
    main()
