import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt


class FER2013Dataset(Dataset):
    """Dataset loader for FER2013 CSV files."""

    def __init__(self, csv_path, label_map=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # FER2013 uses labels [0-6] in the same order as our class list
        self.label_map = label_map or {i: i for i in range(7)}

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
    """Return train/val loaders with stronger augmentation on the train split."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.5),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = FER2013Dataset(os.path.join(data_dir, 'train.csv'), transform=train_transform)
    val_ds = FER2013Dataset(os.path.join(data_dir, 'val.csv'), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train(data_dir='fer2013', weight_path='weights/emotion_vit.pth', epochs=50, batch_size=64, lr=1e-3, patience=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model = create_model('mobilevit_xs', pretrained=True, num_classes=7, drop_rate=0.2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    best_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [] , []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_iter.set_postfix({'loss': loss.item()})
            epoch_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = epoch_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        val_iter = tqdm(val_loader, desc='Validating', leave=False)
        with torch.no_grad():
            for images, labels in val_iter:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/{epochs} - val loss: {val_loss:.4f} - acc: {val_acc:.2f}%')
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(model.state_dict(), weight_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print('Early stopping')
                break
    if not os.path.exists(weight_path):
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model.state_dict(), weight_path)
    print(f'Saved best model to {weight_path}')

    # plot training curves
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='train')
    plt.plot(epochs_range, val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs_range, train_accs, label='train')
    plt.plot(epochs_range, val_accs, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
