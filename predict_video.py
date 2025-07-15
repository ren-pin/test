import argparse
import cv2
from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer


def recognize_video(video_path):
    """Run emotion recognition on a video file and display results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    detector = FaceDetector()
    # Use the same smoothing window as the realtime demo
    recognizer = EmotionRecognizer(smooth_window=5)

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
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run emotion recognition on a video file")
    parser.add_argument("video", help="Path to an mp4 video")
    args = parser.parse_args()
    recognize_video(args.video)


if __name__ == "__main__":
    main()
