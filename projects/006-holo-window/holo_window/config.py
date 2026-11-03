from dataclasses import asdict, dataclass, fields
import json
import subprocess
import sys


@dataclass
class Settings:
    screen_width: float = 34.0   # visible picture, cm
    screen_height: float = 19.0
    camera_offset: float = 1.0   # lens above the top edge of the picture, cm
    fov: float = 60.0            # webcam horizontal field of view at 640×480, degrees
    ipd: float = 6.3             # distance between your pupils, cm
    predict: float = .03         # how far ahead to extrapolate head motion, s
    camera: int = 0
    photo_near: float = 4.0      # where the closest part of a photo sits behind the glass, cm
    photo_far: float = 120.0

    @property
    def camera_y(self):
        return self.screen_height / 2 + self.camera_offset

    @classmethod
    def load(cls, path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def detect_screen_size():
    """Physical size of the main display in cm from its EDID, or None."""
    if sys.platform != "win32":
        return None
    command = ("Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBasicDisplayParams | "
               "Select-Object -First 1 -Property MaxHorizontalImageSize,MaxVerticalImageSize | ConvertTo-Json")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=15).stdout
        data = json.loads(out)
        w, h = float(data["MaxHorizontalImageSize"]), float(data["MaxVerticalImageSize"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None
    return (w, h) if w > 5 and h > 5 else None


def load_settings(path):
    settings = Settings.load(path)
    if settings is None:
        settings = Settings()
        size = detect_screen_size()
        if size:
            settings.screen_width, settings.screen_height = size
        try:
            settings.save(path)
        except OSError:
            pass
    return settings
