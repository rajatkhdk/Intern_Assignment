import cv2

from src.capture.webcam import WebcamCapture
from src.detector.face_detector import FaceDetector
from src.render.landmark_renderer import LandmarkRenderer


def main():

    webcam = WebcamCapture()

    detector = FaceDetector()

    renderer = LandmarkRenderer(draw_mesh=True)

    webcam.open()

    while True:

        # 1. Capture webcam frame
        frame = webcam.read()

        # 2. Detect facial landmarks
        results = detector.process(frame)

        # 3. Draw mesh and landmarks
        output = renderer.draw(frame, results)

        # 4. Show final frame
        cv2.imshow("3D Face Mesh Tracker", output)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Toggle mesh
        elif key == ord("m"):
            renderer.toggle_mesh()

    webcam.release()

    detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()