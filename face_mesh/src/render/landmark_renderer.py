import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class LandmarkRenderer:

    def __init__(self, draw_mesh: bool = True) -> None:
        self.draw_mesh = draw_mesh

        self._mesh_spec = DrawingSpec(
            color=(0, 220, 200), thickness=1, circle_radius=0
        )
        self._contour_spec = DrawingSpec(
            color=(0, 255, 100), thickness=1, circle_radius=0
        )
        self._iris_spec = DrawingSpec(
            color=(255, 0, 200), thickness=1, circle_radius=0
        )
        self._dot_spec = DrawingSpec(
            color=(255, 220, 0), thickness=-1, circle_radius=2
        )

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        if results.multi_face_landmarks is None:
            return frame

        output = frame.copy()

        h, w = output.shape[:2]

        for face_landmarks in results.multi_face_landmarks:
            if self.draw_mesh:
                mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mesh_spec,
                )

            mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._contour_spec,
            )

            try:
                mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._iris_spec,
                )
            except (AttributeError, TypeError):
                pass  # FACEMESH_IRISES not available (refine_landmarks=False)

        return output