import argparse
import os
import cv2
import pandas as pd
import numpy as np

from emotion_recognizer import EmotionRecognizer


def recognize_csv(csv_path='fer2013/test.csv', out_dir='out'):
    """Convert FER2013 CSV rows to images and run emotion recognition."""
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    recognizer = EmotionRecognizer(smooth_window=1)

    for idx, row in df.iterrows():
        pixels = row['pixels']
        img = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        label, score = recognizer.predict(img)

        vis_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(vis_img, (0, 0), (223, 223), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(vis_img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out_path = os.path.join(out_dir, f"{idx:05d}.jpg")
        cv2.imwrite(out_path, vis_img)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Recognize emotions from a FER2013 CSV file')
    parser.add_argument('--csv', default='fer2013/test.csv', help='Path to FER2013 CSV file')
    parser.add_argument('--out', default='out', help='Directory to save annotated images')
    args = parser.parse_args()
    recognize_csv(args.csv, args.out)


if __name__ == '__main__':
    main()
