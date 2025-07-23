import argparse
import os
import pandas as pd
import numpy as np
import cv2
from emotion_recognizer import EmotionRecognizer


def main():
    parser = argparse.ArgumentParser(description="Run emotion recognition on FER2013 test set and save annotated images")
    parser.add_argument('--csv', default='fer2013/test.csv', help='Path to FER2013 test.csv')
    parser.add_argument('--out_dir', default='out', help='Directory to save output images')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    recognizer = EmotionRecognizer(smooth_window=1)

    for idx, row in df.iterrows():
        pixels = row['pixels']
        arr = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        label, score = recognizer.predict(img)

        vis = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(vis, (0, 0), (223, 223), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(vis, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out_path = os.path.join(args.out_dir, f"{idx:05d}.png")
        cv2.imwrite(out_path, vis)

    print(f"Saved {len(df)} images to {args.out_dir}")


if __name__ == '__main__':
    main()
