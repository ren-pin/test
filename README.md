# Real-time Facial Expression Recognition Demo

This project provides a lightweight demo of real-time facial expression recognition using a MobileViT Transformer model.

## Features

- Capture frames from the webcam
- Detect faces using OpenCV Haar cascades
- Recognize seven basic emotions with a MobileViT Transformer
- Display results with bounding boxes and scores
- Smooth predictions over several frames to reduce flicker


## Requirements

- Python 3.8+
- PyTorch
 - timm
 - tqdm
 - OpenCV (`opencv-python`)
 - NumPy

Install dependencies with:

```bash
pip install torch opencv-python numpy timm tqdm
```

## Running

```bash
python app.py
```

Press `q` to quit the application.

Pretrained weights greatly improve recognition accuracy. Download a MobileViT emotion recognition checkpoint and place it at `weights/emotion_vit.pth`. If the file is missing, the model will start from ImageNet pretrained weights, which work reasonably but are less accurate for emotion recognition.

## Training on FER2013

The repository expects the FER2013 dataset to be placed in a folder named
`fer2013/` containing `train.csv` and `val.csv` files. After installing the
requirements you can train the MobileViT model using:

```bash
python train.py
```

Training outputs a weight file at `weights/emotion_vit.pth` which will be
used automatically by the demo.

The training pipeline now applies random horizontal flips for augmentation and
uses the correct FER2013 label ordering to improve classification accuracy.
