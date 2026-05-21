# Lve Facial Landmark Tracker

A real time webcam application that detects the face and draws a **3D facial mesh** of 468 landmarks using **OpenCV** and **Google MediaPipe FaceMesh**.

## How it works:

```
Webcam (OpenCV) -> WebcamCapture.read() -> FaceDetector.process() -> LandmarkRenderer.draw()
```

### WebcamCapture.read() :
OpenCV opens a window and accesses webcam to capture a live video.

### FaceDetector.process() :
Detects the 468 landmarks in the face but are invisible to eye.

### LandmarkRenderer.draw() :
Draws three different landmarks connection :

1. FACEMESH_TESSELATION -> (triangular mesh around the face)
2. FACEMESH_CONTOURS -> (outlines of feartures like jawline, eyes, nose, lips)
3. FACEMESH_IRISES -> (Small circles around pupils)

# Running the project
```bash
cd face_mesh
python app.py
```

### Quit webcam
```
q
```