import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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

    correct_confs = []
    incorrect_confs = []

    correct = 0
    top3_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    times = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = model(images)
            times.append(time.time() - start_time)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            top3 = torch.topk(outputs, 3, dim=1).indices
            top3_correct += (
                (labels.view(-1, 1) == top3).any(dim=1).sum().item()
            )
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct_mask = preds == labels
            if correct_mask.any():
                correct_confs.extend(confs[correct_mask].cpu().tolist())
            if (~correct_mask).any():
                incorrect_confs.extend(confs[~correct_mask].cpu().tolist())

    acc = 100.0 * correct / total if total > 0 else 0.0
    top3_acc = 100.0 * top3_correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {acc:.2f}%')
    print(f'Test Top-3 Accuracy: {top3_acc:.2f}%')
    if times:
        avg_time = sum(times) / len(times)
        print(f'Average inference time per batch: {avg_time:.4f} seconds')
    print(classification_report(all_labels, all_preds, digits=4))
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))

    if correct_confs:
        arr = np.array(correct_confs)
        print(
            'Correct prediction confidence: mean {:.4f}, var {:.4f}, max {:.4f}, min {:.4f}'.format(
                arr.mean(), arr.var(), arr.max(), arr.min()
            )
        )
    if incorrect_confs:
        arr = np.array(incorrect_confs)
        print(
            'Incorrect prediction confidence: mean {:.4f}, var {:.4f}, max {:.4f}, min {:.4f}'.format(
                arr.mean(), arr.var(), arr.max(), arr.min()
            )
        )


if __name__ == '__main__':
    evaluate()
