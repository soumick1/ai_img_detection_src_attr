"""Single-network RGB + neighboring-pixel-residual classifier."""

import torch
from torch import nn
from torchvision.models import resnet18

PARAMETER_COUNT = 11_173_635
PARAMETER_LIMIT = 12_300_000


class MeanStdPool(nn.Module):
    """Spatial moments in FP32, including population standard deviation."""

    def forward(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            value = features.float()
            mean = value.mean(dim=(-2, -1), keepdim=True)
            variance = value.var(dim=(-2, -1), correction=0, keepdim=True)
            std = torch.sqrt(variance.clamp_min(0) + 1e-6)
            return torch.cat((mean, std), dim=1)


class RGBNPRClassifier(nn.Module):
    """Native 512-pixel classifier; all learned parameters start from scratch."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(6, 64, 3, stride=2, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.avgpool = MeanStdPool()
        self.backbone.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, 3))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 512, 512):
            raise ValueError("Expected normalized RGB images with shape N x 3 x 512 x 512.")
        if not images.is_floating_point():
            raise ValueError("Normalize input pixels to floating point before prediction.")
        reference = images[:, :, ::2, ::2].repeat_interleave(2, 2).repeat_interleave(2, 3)
        residual = (images - reference) * 4.0
        return self.backbone(torch.cat((residual, images), dim=1))


def build_model():
    model = RGBNPRClassifier()
    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETER_COUNT or count > PARAMETER_LIMIT:
        raise RuntimeError(f"Architecture parameter count changed: {count}.")
    return model
