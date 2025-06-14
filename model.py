"""Model utilities using a lightweight MobileViT architecture."""

import os
import torch
from timm import create_model


def load_model(weight_path="weights/emotion_vit.pth", device="cpu"):
    """Load MobileViT with optional pretrained weights."""

    model = create_model("mobilevit_xxs", pretrained=False, num_classes=7)

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
