import cv2
import torch
import numpy as np
from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer


def main():
    detector = FaceDetector()
    recognizer = EmotionRecognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label, score = recognizer.predict(face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
