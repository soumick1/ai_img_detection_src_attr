"""Paired, label-independent sampling; every original row is visited once.

Version tags inside private seed strings intentionally preserve the recorded
experiment stream. They are numerical reproducibility identifiers.
"""

import hashlib
import math
import random

import torch
from torch.utils.data import Sampler

QUOTAS = (0, 16, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProcessingSampler(Sampler):
    """All rows once; keys carry private seeds to workers, with no shared RNG."""

    def __init__(self, length, batch_size=64, seed=7454):
        if (
            type(length) is not int
            or length < 2
            or length % 2
            or (type(batch_size) is not int)
            or (batch_size < 2)
            or batch_size % 2
        ):
            raise ValueError("Require a positive even row count and physical batch size.")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        self.count, self.batch_size, self.seed, self.epoch = (length, batch_size, seed, 0)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer.")
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.count / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 10000 + self.epoch)
        indices = torch.randperm(self.count, generator=generator).tolist()
        rng = random.Random((self.seed << 32) + self.epoch)
        for start in range(0, self.count, self.batch_size):
            members = indices[start : start + self.batch_size]
            families = ["exact"] * (len(members) // 2) + ["variation"] * (len(members) // 2)
            rng.shuffle(families)
            yield [(index, family, rng.getrandbits(64)) for index, family in zip(members, families)]


def _subseed(seed, stream):
    require(type(seed) is int and 0 <= seed < 2**64, "Require one private unsigned64 item seed.")
    return int.from_bytes(
        hashlib.sha256(seed.to_bytes(8, "little") + stream.encode("ascii")).digest()[:8], "little"
    )


class NoiseSampler(Sampler):
    """Original family/order/item seeds stay exact; an independent stream adds noise."""

    def __init__(self, length, batch_size=64, seed=7454):
        self.base = ProcessingSampler(length, batch_size, seed)
        self.seed, self.epoch = (seed, 0)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for batch_index, keys in enumerate(self.base):
            material = f"noise_quota_v55\x00{self.seed}\x00{self.epoch}\x00{batch_index}".encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "little"))
            mask = [True] * (len(keys) // 2) + [False] * (len(keys) // 2)
            rng.shuffle(mask)
            yield [
                (index, family, item_seed, enabled, _subseed(item_seed, "independent_noise"))
                for (index, family, item_seed), enabled in zip(keys, mask)
            ]


class CurriculumSampler(Sampler):
    """Original57 keys stay literal; append only the independently selected flag."""

    def __init__(self, rows, quota, batch_size=64, seed=7454):
        require(
            quota in QUOTAS and batch_size % 4 == 0 and (batch_size >= 4),
            "Require a declared quota and physical batch divisible by four.",
        )
        require(
            rows
            and len(rows) % batch_size in (0, batch_size // 2)
            and all(
                (
                    r.get("split") == "train"
                    and type(r.get("label")) is int
                    and (r["label"] in (0, 1, 2))
                    for r in rows
                )
            ),
            "Require original TRAIN rows and complete or half-size tail.",
        )
        require(
            len({sum((r["label"] == c for r in rows)) for c in range(3)}) == 1,
            "Require equal original class totals.",
        )
        self.base = NoiseSampler(len(rows), batch_size, seed)
        self.quota, self.seed, self.epoch = (quota, seed, 0)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for batch_index, keys in enumerate(self.base):
            available = [i for i, key in enumerate(keys) if not key[3]]
            random.Random(f"forced_bilinear_v66:{self.seed}:{self.epoch}:{batch_index}").shuffle(
                available
            )
            forced = set(available[: len(keys) * self.quota // 64])
            yield [(*key, i in forced) for i, key in enumerate(keys)]


class AnchorSampler(Sampler):
    def __init__(self, rows, batch_size=64, seed=7454):
        if (
            not rows
            or any((r["split"] != "train" for r in rows))
            or type(batch_size) is not int
            or (batch_size < 1)
        ):
            raise ValueError("Require labeled training rows and a positive physical batch size.")
        if len({r["filename"] for r in rows}) != len(rows):
            raise ValueError("Duplicate training row.")
        self.count, self.batch_size, self.seed, self.epoch = (len(rows), batch_size, int(seed), 0)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer.")
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.count / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 10000 + self.epoch)
        indices = torch.randperm(self.count, generator=generator).tolist()
        rng = random.Random((self.seed << 32) + self.epoch)
        for start in range(0, self.count, self.batch_size):
            yield [
                (index, "target", rng.getrandbits(64))
                for index in indices[start : start + self.batch_size]
            ]
