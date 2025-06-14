import os
import cv2
import torch
import numpy as np
from collections import deque
from model import load_model


LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionRecognizer:
    def __init__(self, weight_path="weights/emotion_vit.pth", device=None, smooth_window=5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(weight_path, self.device)
        self.softmax = torch.nn.Softmax(dim=1)
        # Keep a window of recent probability vectors to smooth predictions
        self.history = deque(maxlen=smooth_window)

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, img):
        tensor = self.preprocess(img)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = self.softmax(logits)
        # accumulate probabilities for smoothing
        self.history.append(probs.squeeze(0))
        avg_prob = torch.stack(list(self.history), dim=0).mean(dim=0)
        score, idx = torch.max(avg_prob, dim=0)
        label = LABELS[idx.item()]
        return label, score.item()
