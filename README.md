# Real-time Facial Expression Recognition Demo


This project provides a lightweight demo of real-time facial expression recognition using a MobileViT Transformer model.


## Features

- Capture frames from the webcam
- Detect faces using OpenCV Haar cascades
- Recognize seven basic emotions with a MobileViT Transformer

- Display results with bounding boxes and scores

## Requirements

- Python 3.8+
- PyTorch
- timm
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:

```bash
pip install torch opencv-python numpy timm

```

## Running

```bash
python app.py
```

Press `q` to quit the application.

Pretrained weights greatly improve recognition accuracy. Download a MobileViT emotion recognition checkpoint and place it at `weights/emotion_vit.pth`. If the file is missing, the demo will fall back to random weights and results will be poor.


## Training on FER2013

The repository expects the FER2013 dataset to be placed in a folder named
`fer2013/` containing `train.csv` and `val.csv` files. After installing the
requirements you can train the MobileViT model using:

```bash
python train.py
```

Training outputs a weight file at `weights/emotion_vit.pth` which will be
used automatically by the demo.
