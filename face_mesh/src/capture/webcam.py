import cv2

class WebcamCapture:

    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 540) -> None:
        
        if camera_index < 0:
            raise ValueError(f"camera_index must be >= 0, got {camera_index}")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")

        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera {self.camera_index} opened at {actual_w}×{actual_h}")

    def read(self) -> cv2.Mat:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not opened. Call open() before read().")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame
    
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print(f"[INFO] Camera {self.camera_index} released.")

    
    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def actual_resolution(self) -> tuple[int, int]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not opened. Call open() before accessing resolution.")
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_w, actual_h
    
if __name__ == "__main__":
    capture = WebcamCapture(camera_index=0, width=960, height=540)
    capture.open()
    try:
        while True:
            frame = capture.read()
            cv2.imshow("Webcam Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC to quit
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()