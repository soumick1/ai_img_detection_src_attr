"""Select between completed candidates using labeled stress results only."""

import argparse
import json
from pathlib import Path

from .utils import write_json
from .validation import quality_gate, selection_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs=3, required=True, help="Three completed runs, in endpoint order 25, 50, 75."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    shared = None
    for index, folder in enumerate(args.runs):
        report = json.loads((Path(folder) / "evaluation.json").read_text(encoding="utf-8"))
        if report["completed_epochs"] != 10 or report["training_epochs"] != 120:
            raise ValueError("All three final branches must complete ten epochs.")
        config = dict(report["config"])
        if config.pop("post_endpoint_probability") != (0.25, 0.5, 0.75)[index]:
            raise ValueError("Supply branches in the declared endpoint order.")
        config.pop("name")
        current = (report["parent_sha256"], report["manifest_sha256"], config)
        if shared is None:
            shared = current
        elif current != shared:
            raise ValueError("Candidate parent, data split or paired training settings differ.")
        if quality_gate(report["metrics"])["passed"]:
            candidates.append((selection_score(report["metrics"]), -index, folder))
    candidates.sort(reverse=True)
    write_json(
        args.output,
        {
            "selected": candidates[0][2] if candidates else None,
            "ranking": [row[2] for row in candidates],
            "test_predictions_used": False,
        },
    )


if __name__ == "__main__":
    main()
