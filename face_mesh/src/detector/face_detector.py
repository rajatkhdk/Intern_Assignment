import cv2
import mediapipe as mp
from src.capture.webcam import WebcamCapture

class FaceDetector:
    def __init__(self, max_faces=2, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6, refine_landmarks=True):
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in [0, 1].")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be in [0, 1].")
        if max_faces < 1:
            raise ValueError("max_faces must be at least 1.")

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        print(f"[INFO] FaceDetector ready — max_faces={max_faces}")

    def process(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty or None frame.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True
        return results

    def close(self):
        self._face_mesh.close()
        print("[INFO] FaceDetector closed.")

def main():

    webcam = WebcamCapture()

    detector = FaceDetector()

    webcam.open()

    while True:

        # Read frame
        frame = webcam.read()

        # Process frame
        results = detector.process(frame)

        # Draw landmarks
        # frame = detector.draw_landmarks(frame, results)

        # Show frame
        cv2.imshow("Face Mesh Tracker", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()

    detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()