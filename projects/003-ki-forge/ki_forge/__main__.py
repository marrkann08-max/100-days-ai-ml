"""Run with python -m ki_forge; --demo works without a camera or model."""
import argparse
from pathlib import Path
import sys
import time
import urllib.request

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
ROOT = Path(__file__).resolve().parent.parent


def model_path(path):
    if path.is_file() and path.stat().st_size > 1000000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    print('Downloading the hand tracking model from Google (first run only)...')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as source:
            with temporary.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        if temporary.stat().st_size < 1000000:
            raise RuntimeError('Model download is incomplete.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description='Ki Forge: summon energy with your hands.')
    parser.add_argument('--demo', action='store_true', help='Mouse controls; no webcam or model needed')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=Path, default=ROOT/'assets/hand_landmarker.task')
    parser.add_argument('--preview', type=Path, help='Save a deterministic preview and exit without a camera')
    args = parser.parse_args()
    import cv2
    import numpy as np
    from .effects import Renderer, hud
    from .gestures import Energy, Hand, describe

    energy, renderer = Energy(), Renderer(seed=42)
    width, height = 1280, 720
    def background():
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (18, 13, 10)
        for x in range(0, width, 48):
            cv2.line(frame, (x, 116), (x, height-65), (32, 27, 20), 1)
        for y in range(116, height-65, 48):
            cv2.line(frame, (0, y), (width, y), (32, 27, 20), 1)
        return frame

    if args.preview:
        for _ in range(110):
            energy.update([Hand((width/2, height/2), 100, True)], 1/30, width)
            renderer.time += 1/30
        frame = hud(renderer.draw(background(), energy, 1/30), energy, 60, True)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), frame):
            raise RuntimeError('Could not save preview.')
        print(f'Preview saved: {args.preview}')
        return

    capture = tracker = None
    window = 'Ki Forge'
    mouse = [width/2, height/2, False]
    def mouse_event(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse[2] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse[2] = False

    try:
        if not args.demo:
            import mediapipe as mp
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path(args.model))),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=.55,
                min_hand_presence_confidence=.55, min_tracking_confidence=.55)
            tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            capture = cv2.VideoCapture(args.camera)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not capture.isOpened():
                raise RuntimeError('Camera unavailable. Check camera permissions, try --camera 1, or use --demo.')
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, mouse_event)
        previous = time.perf_counter()
        timestamp = 0
        fps = 30.0
        while True:
            now = time.perf_counter()
            dt = min(now-previous, .1)
            previous = now
            if args.demo:
                frame = background()
                hands = [Hand(tuple(mouse[:2]), 100, not mouse[2])]
            else:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames. Close other camera apps and retry.')
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (width, height))
                timestamp = max(timestamp+1, int(now*1000))
                result = tracker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp)
                hands = [describe(points, width, height) for points in result.hand_landmarks]
            if energy.update(hands, dt, width):
                renderer.blast(energy.center, energy.direction, energy.radius)
            fps = fps*.9 + .1/max(dt, .001)
            output = hud(renderer.draw(frame, energy, dt), energy, fps, args.demo)
            cv2.imshow(window, output)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('c'):
                renderer.palette = (renderer.palette+1) % 3
            elif key == ord('r'):
                energy.reset()
                renderer.blasts.clear()
                renderer.particles.clear()
            elif key == ord('s'):
                folder = ROOT/'captures'
                folder.mkdir(exist_ok=True)
                path = folder/f'ki-{time.time_ns()}.png'
                if not cv2.imwrite(str(path), output):
                    raise RuntimeError('Could not save screenshot.')
                print(f'Saved {path}')
    finally:
        if capture is not None:
            capture.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ImportError) as error:
        print(f'Ki Forge: {error}', file=sys.stderr)
        sys.exit(1)
