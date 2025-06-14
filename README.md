# Real-time Facial Expression Recognition Demo

This project provides a lightweight demo of real-time facial expression recognition using a small Transformer model.

## Features

- Capture frames from the webcam
- Detect faces using OpenCV Haar cascades
- Recognize seven basic emotions with a tiny Vision Transformer
- Display results with bounding boxes and scores

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:

```bash
pip install torch opencv-python numpy
```

## Running

```bash
python app.py
```

Press `q` to quit the application.

The model weights can be provided via `model.pth` in the project directory. If no weights are found, the model runs with random initialization (for demonstration purposes only).

