from pathlib import Path
import urllib.request

MODELS = Path(__file__).resolve().parent.parent / "models"

FACE = ("face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
DEPTH = ("depth_anything_v2_small.onnx",
         "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx")


def ensure(model):
    """Path to a model file, downloading it into models/ the first time."""
    name, url = model
    path = MODELS / name
    if path.exists():
        return path
    MODELS.mkdir(exist_ok=True)
    partial = path.with_suffix(".part")
    print(f"Downloading {name}...")

    def progress(blocks, block_size, total):
        if total > 0:
            print(f"\r  {min(100, blocks * block_size * 100 // total)}%", end="", flush=True)

    urllib.request.urlretrieve(url, partial, progress)
    print()
    partial.replace(path)
    return path
