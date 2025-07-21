"""Model utilities using a lightweight MobileViT architecture."""

import os
import torch
from timm import create_model


def load_model(weight_path="weights/emotion_vit.pth", device="cpu"):
    """Load MobileViT with pretrained ImageNet weights if emotion weights are missing."""

    # Start from ImageNet pretrained weights for better accuracy when custom
    # emotion weights are unavailable.
    model = create_model("mobilevit_xs", pretrained=True, num_classes=7, drop_rate=0.2)

    if weight_path and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(
            "WARNING: Weight file not found. Download pretrained weights for better accuracy."
        )

    model.to(device)
    model.eval()
    return model
