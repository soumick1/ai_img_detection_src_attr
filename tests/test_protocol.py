"""Regression tests for the scientific protocol, independent of private data."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_attribution.data import noise_parameters, normalize
from ai_attribution.engine import train_epoch
from ai_attribution.model import PARAMETER_COUNT, MeanStdPool, build_model
from ai_attribution.predict import validate_mapping
from ai_attribution.samplers import CurriculumSampler
from ai_attribution.utils import learning_rate
from ai_attribution.validation import evaluate, quality_gate, selection_score

ROOT = Path(__file__).resolve().parents[1]


class KnownValidation(torch.utils.data.Dataset):
    """Two condition-major views with one controlled classification error."""

    conditions = ("clean", "q90_420")

    def __len__(self):
        return 12

    def __getitem__(self, index):
        condition, row = divmod(index, 6)
        label = row % 3
        predicted = 1 if condition == 1 and row == 0 else label
        logits = torch.full((3,), -3.0)
        logits[predicted] = 3.0
        return logits, label, row, condition


class ProtocolTests(unittest.TestCase):
    def test_validation_condition_boundaries_and_filename_pairing(self):
        rows = [{"filename": f"{i}.jpg", "label": i % 3} for i in range(6)]
        dataset = KnownValidation()
        # Batch size four deliberately crosses the condition boundary at row six.
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        metrics = evaluate(nn.Identity(), loader, rows, torch.device("cpu"), amp=False)
        self.assertEqual([m["correct"] for m in metrics], [6, 5])
        self.assertEqual(metrics[1]["error_filenames"], ["0.jpg"])
        self.assertEqual(metrics[1]["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 0, 2]])
        reversed_loader = DataLoader(dataset, batch_size=4, sampler=list(reversed(range(12))))
        with self.assertRaisesRegex(RuntimeError, "condition-major"):
            evaluate(nn.Identity(), reversed_loader, rows, torch.device("cpu"), amp=False)

    def test_epoch_visits_and_disjoint_augmentation_quotas(self):
        rows = [{"filename": f"{i}.jpg", "label": i % 3, "split": "train"} for i in range(12000)]
        sampler = CurriculumSampler(rows, 16, 64, 7454)
        sampler.set_epoch(10)
        batches = list(sampler)
        self.assertEqual(len(batches), 188)
        self.assertEqual(sorted(k[0] for b in batches for k in b), list(range(12000)))
        for batch in batches:
            self.assertEqual(sum(k[3] for k in batch), len(batch) // 2)
            self.assertEqual(sum(k[5] for k in batch), len(batch) // 4)
            self.assertTrue(all(not (k[3] and k[5]) for k in batch))
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(11)
        self.assertNotEqual(batches, list(sampler))

    def test_only_post_noise_endpoint_changes(self):
        base = json.loads((ROOT / "configs/ablation_post_25.json").read_text())
        altered = copy.deepcopy(base)
        altered["post_endpoint_probability"] = 0.75
        changed = 0
        for seed in range(500):
            a, b = noise_parameters(seed, base), noise_parameters(seed, altered)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[2], b[2])
            if a[0] == "pre":
                self.assertEqual(a, b)
            elif a != b:
                changed += 1
                self.assertEqual(b[1], 4.0)
        self.assertGreater(changed, 80)

    def test_architecture_input_order_and_spatial_moments(self):
        model = build_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETER_COUNT)
        model.backbone = nn.Identity()
        image = torch.arange(3 * 512 * 512, dtype=torch.float32).reshape(1, 3, 512, 512)
        result = model(image)
        self.assertTrue(torch.equal(result[:, 3:], image))
        self.assertTrue(
            torch.equal(result[:, :3, ::2, ::2], torch.zeros_like(image[:, :, ::2, ::2]))
        )
        self.assertEqual(float(result[0, 0, 0, 1]), 4.0)
        moments = MeanStdPool()(torch.tensor([[[[1.0, 3.0]]]]))
        self.assertAlmostEqual(float(moments[0, 0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(moments[0, 1, 0, 0]), float(np.sqrt(1 + 1e-6)))
        self.assertTrue(
            torch.equal(normalize(Image.new("RGB", (512, 512), "black")), -torch.ones(3, 512, 512))
        )

    def test_retained_noise_endpoint_uses_twenty_epoch_schedule(self):
        config = json.loads((ROOT / "configs/03_noise.json").read_text())
        self.assertEqual(config["epochs"], 15)
        self.assertEqual(config["schedule_epochs"], 20)
        self.assertAlmostEqual(learning_rate(config, 0), 1.5e-5)
        self.assertAlmostEqual(learning_rate(config, 19), 1e-6)
        self.assertGreater(learning_rate(config, 14), 1e-6)

    def test_training_step_updates_weights_and_handles_tail(self):
        torch.manual_seed(3)
        model = nn.Linear(4, 3)
        initial = model.weight.detach().clone()
        loader = DataLoader(
            TensorDataset(torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 0])), batch_size=3
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        loss, accuracy, steps = train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            torch.device("cpu"),
            {"amp": False, "label_smoothing": 0.05, "gradient_clip": 5.0, "batch_size": 3},
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= accuracy <= 1)
        self.assertEqual(steps, 2)
        self.assertFalse(torch.equal(initial, model.weight))

    def test_recorded_selection_and_submission_contract(self):
        record = json.loads((ROOT / "results/validation.json").read_text())
        entries = record["ablation"]
        eligible = [e for e in entries if quality_gate(e["conditions"])["passed"]]
        self.assertEqual([e["post_endpoint"] for e in eligible], [0.5, 0.75])
        best = max(eligible, key=lambda e: selection_score(e["conditions"]))
        self.assertEqual(best["post_endpoint"], 0.75)
        validate_mapping({"a.jpg": 0, "b.jpg": 2}, ["a.jpg", "b.jpg"])
        with self.assertRaises(ValueError):
            validate_mapping({"a.jpg": True}, ["a.jpg"])
        with self.assertRaises(ValueError):
            validate_mapping({"a.jpg": 0}, ["a.jpg", "b.jpg"])


if __name__ == "__main__":
    unittest.main()
