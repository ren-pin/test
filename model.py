import os
import torch
from torch import nn


class SimpleViT(nn.Module):
    """A very small Vision Transformer for demo purposes."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (B, C)
        logits = self.fc(x)
        return logits


def load_model(weight_path=None, device='cpu'):
    """Load model weights with an optional custom path."""
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__), "weights", "emotion_vit.pth")

    model = SimpleViT()
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model
