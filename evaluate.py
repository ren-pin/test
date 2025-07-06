import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from train import FER2013Dataset
from model import load_model


def evaluate(data_path='fer2013/test.csv', weight_path='weights/emotion_vit.pth', batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = FER2013Dataset(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = load_model(weight_path, device)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {acc:.2f}%')


if __name__ == '__main__':
    evaluate()
