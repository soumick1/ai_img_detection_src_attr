"""Render paper figures from recorded measurements without fitting any model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((root / "results/validation.json").read_text())["conditions"]
    values = {m["condition"]: m["accuracy"] * 100 for m in metrics}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    sigmas = [1, 2, 4, 8]
    for position, color, marker, label in [
        ("pre", "#1f5d8f", "o", "Before processing"),
        ("post", "#bd5a2a", "s", "After JPEG decode"),
    ]:
        ax.plot(
            sigmas,
            [values[f"noise_{position}_sigma{s}"] for s in sigmas],
            marker=marker,
            color=color,
            linewidth=1.3,
            markersize=3.5,
            label=label,
        )
    ax.set(
        xlabel="Noise standard deviation (8-bit pixel units)",
        ylabel="Accuracy (%)",
        xticks=sigmas,
        ylim=(75, 101),
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.25)
    ax.axvline(4, color=".5", linestyle=":", linewidth=0.7)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(args.output / "noise_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    histories = json.loads((root / "results/training_history.json").read_text())["histories"]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.72))
    for dose, color, marker in [
        ("25", "#737373", "^"),
        ("50", "#bd5a2a", "s"),
        ("75", "#1f5d8f", "o"),
    ]:
        history = histories[dose]
        epochs = [row["epoch"] for row in history]
        for ax, key, scale in [(axes[0], "train_loss", 1), (axes[1], "train_accuracy", 100)]:
            ax.plot(
                epochs,
                [scale * row[key] for row in history],
                color=color,
                marker=marker,
                markersize=2.2,
                linewidth=0.9,
                label=f"p={int(dose) / 100:.2f}",
            )
    axes[0].set(ylabel="Training loss", yticks=[0.19, 0.195, 0.20], ylim=(0.187, 0.201))
    axes[1].set(ylabel="Training accuracy (%)", yticks=[99.2, 99.6, 100.0], ylim=(99.1, 100.03))
    for ax in axes:
        ax.set(xticks=[1, 5, 10], xlabel="Final-stage epoch")
        ax.grid(axis="y", linewidth=0.4, alpha=0.25)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.4,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.88))
    fig.savefig(args.output / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote two evidence-derived figures to {args.output}")


if __name__ == "__main__":
    main()
