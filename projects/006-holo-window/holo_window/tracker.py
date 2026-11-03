import threading
import time

import numpy as np

from .geometry import OneEuro, eye_from_landmarks, iris_gap

DEFAULT_EYE = np.array([0.0, 0.0, 60.0])
FRAME = (640, 480)


def ease_home(eye, lost_for):
    """Drift back to a centred viewer after the face has been lost for a while."""
    t = min(1.0, max(0.0, lost_for - .5) / 1.5)
    return eye + (DEFAULT_EYE - eye) * t


class FaceTracker:
    """Webcam + MediaPipe Face Landmarker on a background thread.

    The render loop calls eye() whenever it likes and always gets the newest
    smoothed head position, extrapolated a little to hide camera latency.
    """

    def __init__(self, settings, model_path):
        self.settings, self.model_path = settings, model_path
        self.filter = OneEuro()
        self.lock = threading.Lock()
        self.position = DEFAULT_EYE.copy()
        self.velocity = np.zeros(3)
        self.seen = 0.0
        self.found = False
        self.gap = None
        self.preview = None
        self.frame = None
        self.mark = None
        self.fps = 0.0
        self.error = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def open_camera(self):
        import cv2
        camera = cv2.VideoCapture(self.settings.camera, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME[1])
        camera.set(cv2.CAP_PROP_FPS, 30)
        if not camera.isOpened():
            raise RuntimeError(f"Couldn't open camera {self.settings.camera}.")
        return camera

    def loop(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        try:
            camera = self.open_camera()
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO, num_faces=1)
            landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as error:
            self.error = str(error)
            self.running = False
            return

        start, last_stamp, frames, counted = time.monotonic(), -1, 0, time.monotonic()
        while self.running:
            ok, frame = camera.read()
            if not ok:
                time.sleep(.01)
                continue
            now = time.monotonic()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            size = rgb.shape[1], rgb.shape[0]
            stamp = max(last_stamp + 1, int((now - start) * 1000))
            last_stamp = stamp
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)

            mark = None
            if result.face_landmarks:
                points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
                eye = eye_from_landmarks(points, size, self.settings)
                smooth = self.filter(eye, now)
                mark = (points[468, :2] + points[473, :2]) / 2
                with self.lock:
                    self.position, self.velocity = smooth, self.filter.velocity
                    self.seen, self.found = now, True
                    self.gap = iris_gap(points, size)
            else:
                with self.lock:
                    self.found = False

            preview = cv2.resize(rgb, (FRAME[0] // 2, FRAME[1] // 2))[:, ::-1].copy()
            with self.lock:
                self.preview = preview
                self.frame = rgb[:, ::-1].copy()
                self.mark = None if mark is None else (1 - mark[0], mark[1])
            frames += 1
            if now - counted >= 1:
                self.fps, frames, counted = frames / (now - counted), 0, now

        camera.release()
        landmarker.close()

    def eye(self):
        now = time.monotonic()
        with self.lock:
            position, velocity, seen, found = self.position.copy(), self.velocity.copy(), self.seen, self.found
        if not seen:
            return DEFAULT_EYE.copy()
        lead = min(self.settings.predict + (now - seen), .1) if found else 0
        eye = position + velocity * lead
        return eye if found else ease_home(eye, now - seen)

    def status(self):
        if self.error:
            return "camera error: " + self.error
        if not self.seen:
            return "looking for a face..."
        return "tracking" if self.found else "face lost"


class MouseTracker:
    """Stand-in for the webcam: the mouse moves your head, the wheel moves it nearer or further."""

    fps, preview, frame, mark, gap, error = 0.0, None, None, None, None, None

    def __init__(self, settings, size):
        self.settings, self.size = settings, size
        self.position = DEFAULT_EYE.copy()

    def start(self):
        return self

    def stop(self):
        pass

    def move(self, pixel):
        x, y = pixel
        w, h = self.size
        self.position[0] = (x / w - .5) * self.settings.screen_width * 1.6
        self.position[1] = (.5 - y / h) * self.settings.screen_height * 1.6

    def zoom(self, steps):
        self.position[2] = float(np.clip(self.position[2] - steps * 4, 20, 140))

    def eye(self):
        return self.position.copy()

    def status(self):
        return "mouse mode"
