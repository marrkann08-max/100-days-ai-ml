import argparse
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
import moderngl
import numpy as np
import pygame as pg

from . import models
from .config import load_settings
from .geometry import fov_for, focal_length, off_axis
from .scenes import BACKGROUND, Photo, Room
from .tracker import DEFAULT_EYE, FRAME, FaceTracker, MouseTracker

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "assets" / "mars-murray-buttes.jpg"
CONFIG = ROOT / "config.json"
EXPORTS = ROOT / "exports"
CALIBRATION_DISTANCE = 50.0
MAX_PHOTO = 2048

TEXT = (231, 234, 223)
MUTED = (150, 166, 168)
MINT = (194, 235, 180)
ORANGE = (255, 139, 92)

OVERLAY_VERTEX = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); uv = in_uv; }
"""
OVERLAY_FRAGMENT = """
#version 330
uniform sampler2D overlay;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(overlay, uv); }
"""


def read_photo(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_PHOTO, MAX_PHOTO))
        return np.asarray(image)


class Renderer:
    """Draws one scene for one eye position into whatever framebuffer is bound."""

    def __init__(self, ctx, settings):
        self.ctx, self.settings = ctx, settings
        self.scenes = [Room(ctx, settings)]

    def add_photo(self, name, rgb, disparity):
        photo = Photo(self.ctx, self.settings, name, rgb, disparity)
        photo.rgb, photo.disparity = rgb, disparity
        self.scenes.append(photo)
        return len(self.scenes) - 1

    def rebuild(self, index):
        old = self.scenes[index]
        new = Photo(self.ctx, self.settings, old.name, old.rgb, old.disparity)
        new.rgb, new.disparity = old.rgb, old.disparity
        old.release()
        self.scenes[index] = new

    def render(self, scene, eye, window):
        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.clear(*BACKGROUND, depth=1.0)
        self.scenes[scene].draw(off_axis(eye, *window))


class DepthWorker:
    """Runs Depth Anything off the render thread so the window never freezes."""

    def __init__(self):
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.estimator = None
        self.busy = 0
        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, name, rgb):
        self.busy += 1
        self.jobs.put((name, rgb))

    def loop(self):
        from .depth import DepthEstimator
        while True:
            name, rgb = self.jobs.get()
            try:
                if self.estimator is None:
                    self.estimator = DepthEstimator(models.ensure(models.DEPTH))
                self.results.put((name, rgb, self.estimator(rgb), None))
            except Exception as error:
                self.results.put((name, rgb, None, str(error)))


class App:
    def __init__(self, settings, windowed=False, mouse=False, photos=()):
        self.settings = settings
        pg.init()
        flags = pg.OPENGL | pg.DOUBLEBUF
        info = pg.display.Info()
        self.desktop = info.current_w, info.current_h
        if windowed:
            size = int(self.desktop[0] * .75), int(self.desktop[1] * .75)
        else:
            size, flags = self.desktop, flags | pg.FULLSCREEN
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(size, flags, vsync=1)
        pg.display.set_caption("Holo Window")
        self.size = pg.display.get_window_size()
        self.ctx = moderngl.create_context()
        self.renderer = Renderer(self.ctx, settings)
        self.scene = 0
        self.window = (settings.screen_width * self.size[0] / self.desktop[0],
                       settings.screen_height * self.size[1] / self.desktop[1])

        self.sans = pg.font.SysFont("segoeui", 20)
        self.title = pg.font.SysFont("segoeui", 28, bold=True)
        self.mono = pg.font.SysFont("consolas", 15)
        self.overlay = self.ctx.program(vertex_shader=OVERLAY_VERTEX, fragment_shader=OVERLAY_FRAGMENT)
        self.overlay_buffer = self.ctx.buffer(reserve=4 * 4 * 4)
        self.overlay_vao = self.ctx.vertex_array(self.overlay, [(self.overlay_buffer, "2f 2f", "in_pos", "in_uv")])
        self.overlay_texture = None

        self.depth = DepthWorker()
        self.mouse_mode = mouse
        self.tracker = self.make_tracker()
        self.hud = True
        self.message, self.message_until = "", 0.0
        self.calibrating = False
        self.frames, self.fps, self.counted = 0, 0.0, time.monotonic()
        self.running = True
        for path in photos:
            self.load(path)

    def make_tracker(self):
        if self.mouse_mode:
            return MouseTracker(self.settings, self.size).start()
        return FaceTracker(self.settings, models.ensure(models.FACE)).start()

    def say(self, text, seconds=4.0):
        self.message, self.message_until = text, time.monotonic() + seconds

    def load(self, path):
        path = Path(path)
        try:
            rgb = read_photo(path)
        except Exception as error:
            self.say(f"Couldn't open {path.name}: {error}")
            return
        self.depth.submit(path.stem, rgb)
        self.say(f"Estimating depth for {path.stem}...", 30)

    def selfie(self):
        frame = getattr(self.tracker, "frame", None)
        if frame is None:
            self.say("The camera is still starting, try again in a moment.")
            return
        self.depth.submit("3D selfie", frame.copy())
        self.say("Estimating depth for your selfie...", 30)

    def collect(self):
        while not self.depth.results.empty():
            name, rgb, disparity, error = self.depth.results.get()
            self.depth.busy -= 1
            if error:
                self.say(f"Depth failed: {error}")
                continue
            self.scene = self.renderer.add_photo(name, rgb, disparity)
            self.say(f"{name}: move your head to look around inside it.")

    def change_depth(self, factor):
        scene = self.renderer.scenes[self.scene]
        if not isinstance(scene, Photo):
            self.say("Depth only changes for photos.")
            return
        s = self.settings
        s.photo_far = float(np.clip(s.photo_far * factor, 20, 400))
        self.renderer.rebuild(self.scene)
        self.say(f"Photo depth: back wall {s.photo_far:.0f} cm behind the glass.")
        self.save()

    def calibrate(self):
        if not self.calibrating:
            if self.mouse_mode:
                self.say("Calibration needs the webcam.")
                return
            self.calibrating = True
            self.say(f"Sit with your eyes {CALIBRATION_DISTANCE:.0f} cm from the screen, look at it, then press C again.", 60)
            return
        self.calibrating = False
        gap = self.tracker.gap
        if not gap:
            self.say("No face found, calibration cancelled.")
            return
        focal = CALIBRATION_DISTANCE * gap / self.settings.ipd
        self.settings.fov = round(fov_for(FRAME[0], focal), 2)
        self.save()
        self.say(f"Calibrated: webcam field of view {self.settings.fov:.1f}°.")

    def save(self):
        try:
            self.settings.save(CONFIG)
        except OSError:
            pass

    def handle(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.DROPFILE:
            self.load(event.file)
        elif event.type == pg.MOUSEMOTION and self.mouse_mode:
            self.tracker.move(event.pos)
        elif event.type == pg.MOUSEWHEEL and self.mouse_mode:
            self.tracker.zoom(event.y)
        elif event.type == pg.KEYDOWN:
            count = len(self.renderer.scenes)
            if event.key == pg.K_ESCAPE:
                self.running = False
            elif event.key in (pg.K_RIGHT, pg.K_TAB):
                self.scene = (self.scene + 1) % count
            elif event.key == pg.K_LEFT:
                self.scene = (self.scene - 1) % count
            elif event.key == pg.K_h:
                self.hud = not self.hud
            elif event.key == pg.K_p:
                self.selfie()
            elif event.key == pg.K_c:
                self.calibrate()
            elif event.key == pg.K_RIGHTBRACKET:
                self.change_depth(1.25)
            elif event.key == pg.K_LEFTBRACKET:
                self.change_depth(.8)
            elif event.key == pg.K_m:
                self.tracker.stop()
                self.mouse_mode = not self.mouse_mode
                self.tracker = self.make_tracker()
                self.say("Mouse mode: move the mouse, scroll to lean in." if self.mouse_mode else "Webcam tracking.")

    # Heads-up display

    def hud_surface(self, eye):
        lines = []
        surface = None
        if self.hud:
            preview = self.tracker.preview
            surface = pg.Surface((360, 470 if preview is not None else 230), pg.SRCALPHA)
            surface.fill((10, 14, 17, 205))
            scene = self.renderer.scenes[self.scene]
            y = 12
            surface.blit(self.title.render("HOLO WINDOW", True, TEXT), (16, y))
            y += 40
            status = self.tracker.status()
            fps = f"  {self.tracker.fps:.0f} fps" if self.tracker.fps else ""
            for text, color in (
                (f"{scene.name}  ({self.scene + 1}/{len(self.renderer.scenes)})", MINT),
                (f"eye x {eye[0]:+5.1f}  y {eye[1]:+5.1f}  z {eye[2]:5.1f} cm", TEXT),
                (f"{status}{fps}   render {self.fps:.0f} fps", ORANGE if "error" in status or "lost" in status else MUTED),
            ):
                surface.blit(self.mono.render(text, True, color), (16, y))
                y += 22
            if preview is not None:
                y += 6
                image = pg.image.frombuffer(preview.tobytes(), (preview.shape[1], preview.shape[0]), "RGB")
                surface.blit(image, (16, y))
                mark = self.tracker.mark
                if mark is not None:
                    pg.draw.circle(surface, MINT, (16 + mark[0] * preview.shape[1], y + mark[1] * preview.shape[0]), 5, 2)
                y += preview.shape[0] + 8
            for text in ("←/→ scene   drop a photo   P 3D selfie",
                         "[ ] photo depth   C calibrate   M mouse",
                         "H hide panel   Esc quit"):
                surface.blit(self.mono.render(text, True, MUTED), (16, y))
                y += 20
        if time.monotonic() < self.message_until or self.depth.busy:
            lines.append(self.message)
        return surface, lines

    def draw_overlay(self, surface, x, y):
        w, h = surface.get_size()
        if self.overlay_texture is None or self.overlay_texture.size != (w, h):
            if self.overlay_texture:
                self.overlay_texture.release()
            self.overlay_texture = self.ctx.texture((w, h), 4)
        self.overlay_texture.write(pg.image.tobytes(surface, "RGBA", True))
        sw, sh = self.size
        x0, x1 = -1 + 2 * x / sw, -1 + 2 * (x + w) / sw
        y0, y1 = 1 - 2 * (y + h) / sh, 1 - 2 * y / sh
        quad = np.array([x0, y0, 0, 0, x1, y0, 1, 0, x0, y1, 0, 1, x1, y1, 1, 1], "f4")
        self.overlay_buffer.write(quad.tobytes())
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.overlay_texture.use(0)
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def draw(self):
        eye = self.tracker.eye()
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.renderer.render(self.scene, eye, self.window)
        panel, lines = self.hud_surface(eye)
        if panel:
            self.draw_overlay(panel, 20, 20)
        for text in lines:
            label = self.sans.render(text, True, TEXT)
            box = pg.Surface((label.get_width() + 32, label.get_height() + 16), pg.SRCALPHA)
            box.fill((10, 14, 17, 215))
            box.blit(label, (16, 8))
            self.draw_overlay(box, (self.size[0] - box.get_width()) // 2, self.size[1] - box.get_height() - 40)
        pg.display.flip()

    def loop(self):
        if self.mouse_mode:
            self.say("Mouse mode: move the mouse, scroll to lean in.")
        else:
            self.say("Sit back about an arm's length. Close one eye for the strongest effect.", 8)
        while self.running:
            for event in pg.event.get():
                self.handle(event)
            self.collect()
            self.draw()
            self.frames += 1
            now = time.monotonic()
            if now - self.counted >= 1:
                self.fps, self.frames, self.counted = self.frames / (now - self.counted), 0, now
        self.tracker.stop()
        pg.quit()


# Offscreen exports


def offscreen(settings, photos, size=(1280, 720)):
    from .depth import DepthEstimator
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 3)], depth_attachment=ctx.depth_renderbuffer(size))
    renderer = Renderer(ctx, settings)
    if photos:
        estimator = DepthEstimator(models.ensure(models.DEPTH))
        for path in photos:
            rgb = read_photo(path)
            renderer.add_photo(Path(path).stem, rgb, estimator(rgb))
    return ctx, fbo, renderer


def snapshot(fbo, renderer, scene, eye, window):
    from PIL import Image
    fbo.use()
    renderer.render(scene, eye, window)
    return Image.frombytes("RGB", fbo.size, fbo.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def sweep(t, width):
    """A slow figure-of-eight head path for demo exports."""
    return np.array([math.sin(t) * width * .42, math.sin(2 * t) * width * .1, 58 + math.cos(t) * 6])


def export_screenshot(settings, photos, path):
    from PIL import Image
    ctx, fbo, renderer = offscreen(settings, photos)
    window = settings.screen_width, settings.screen_height
    w, h = fbo.size[0] // 2, fbo.size[1] // 2
    rows = len(renderer.scenes)
    sheet = Image.new("RGB", (w * 3 + 20, h * rows + 10 * (rows - 1)), (13, 18, 22))
    for row in range(rows):
        for col, x in enumerate((-.4, 0, .4)):
            eye = DEFAULT_EYE + (x * settings.screen_width, 0, 0)
            image = snapshot(fbo, renderer, row, eye, window).resize((w, h))
            sheet.paste(image, (col * (w + 10), row * (h + 10)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    ctx.release()
    print("Saved", path)


def export_gif(settings, photos, path, frames=36, size=(480, 270)):
    ctx, fbo, renderer = offscreen(settings, photos)
    window = settings.screen_width, settings.screen_height
    images = []
    for scene in range(len(renderer.scenes)):
        for i in range(frames):
            eye = sweep(2 * math.pi * i / frames, settings.screen_width)
            image = snapshot(fbo, renderer, scene, eye, window).resize(size)
            images.append(image.quantize(96))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)
    ctx.release()
    print("Saved", path)


def dpi_aware():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Turn your screen into a window with webcam head tracking.")
    parser.add_argument("photos", nargs="*", type=Path, help="Photos to open as 3D windows.")
    parser.add_argument("--no-sample", action="store_true", help="Don't load the bundled Mars photo.")
    parser.add_argument("--mouse", action="store_true", help="Move your viewpoint with the mouse instead of the webcam.")
    parser.add_argument("--windowed", action="store_true", help="Run in a window instead of fullscreen.")
    parser.add_argument("--camera", type=int, help="Webcam index.")
    parser.add_argument("--screenshot", type=Path, help="Render left/centre/right views offscreen, save and exit.")
    parser.add_argument("--export-gif", type=Path, help="Render a head-sweep animation offscreen, save and exit.")
    args = parser.parse_args()

    settings = load_settings(CONFIG)
    if args.camera is not None:
        settings.camera = args.camera
    photos = ([] if args.no_sample else [SAMPLE]) + list(args.photos)

    if args.screenshot:
        export_screenshot(settings, photos, args.screenshot)
        return
    if args.export_gif:
        export_gif(settings, photos, args.export_gif)
        return
    dpi_aware()
    App(settings, args.windowed, args.mouse, photos).loop()


if __name__ == "__main__":
    main()
