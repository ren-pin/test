import argparse
import cv2
from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer


def recognize_image(image_path):
    """Recognize emotions in a single image of arbitrary size."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    detector = FaceDetector()
    recognizer = EmotionRecognizer(smooth_window=1)
    faces = detector.detect(img)
    results = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        label, score = recognizer.predict(face_img)
        results.append({
            "box": [int(x), int(y), int(w), int(h)],
            "label": label,
            "score": float(score),
        })
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run emotion recognition on an image")
    parser.add_argument("image", help="Path to an input image")
    args = parser.parse_args()
    results = recognize_image(args.image)
    print(results)


if __name__ == "__main__":
    main()
