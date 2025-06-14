import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model


class FER2013Dataset(Dataset):
    """Dataset loader for FER2013 CSV files."""

    def __init__(self, csv_path, label_map=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = label_map or {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 4, 6: 5}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx, 0])
        pixels = self.df.iloc[idx, 1]
        img = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.label_map[label]
        return img, label


def get_dataloaders(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_ds = FER2013Dataset(os.path.join(data_dir, 'train.csv'), transform=transform)
    val_ds = FER2013Dataset(os.path.join(data_dir, 'val.csv'), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train(data_dir='fer2013', weight_path='weights/emotion_vit.pth', epochs=5, batch_size=64, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model = create_model('mobilevit_xxs', pretrained=False, num_classes=7)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f'Epoch {epoch+1}/{epochs} - val acc: {acc:.2f}%')
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    print(f'Saved model to {weight_path}')


if __name__ == '__main__':
    train()
