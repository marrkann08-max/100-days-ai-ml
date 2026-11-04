import argparse
import os
from dataclasses import replace
from pathlib import Path
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import moderngl
import numpy as np
import pygame as pg

from .discover import Discovery
from .rules import Rules, random_rules
from .world import PALETTE, Trails, World

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ROOT / "species"
DISCOVERED = SPECIES / "discovered"
PARTICLES = (4000, 8000, 12000, 16000)

TEXT = (231, 234, 223)
MUTED = (150, 166, 168)
MINT = (194, 235, 180)
ORANGE = (255, 139, 92)
PANEL = (10, 14, 17, 210)

OVERLAY_VERTEX = """
#version 430
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); uv = in_uv; }
"""
OVERLAY_FRAGMENT = """
#version 430
uniform sampler2D overlay;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(overlay, uv); }
"""


def load_gallery():
    files = sorted(SPECIES.glob("*.json")) + sorted(DISCOVERED.glob("*.json"))
    return [Rules.load(path) for path in files]


def cell_color(value):
    """Green for attraction, red for repulsion, dark for neutral."""
    v = float(np.clip(value, -1, 1))
    base = np.array((26, 32, 36))
    tip = np.array((90, 220, 120)) if v > 0 else np.array((240, 80, 80))
    return tuple(int(c) for c in base + (tip - base) * abs(v))


class App:
    def __init__(self, particles=8000, windowed=False, species=None):
        pg.init()
        info = pg.display.Info()
        size = (int(info.current_w * .75), int(info.current_h * .75)) if windowed else (info.current_w, info.current_h)
        flags = pg.OPENGL | pg.DOUBLEBUF | (0 if windowed else pg.FULLSCREEN)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(size, flags, vsync=1)
        pg.display.set_caption("Primordial")
        self.size = pg.display.get_window_size()
        self.ctx = moderngl.create_context(require=430)

        self.title = pg.font.SysFont("segoeui", 28, bold=True)
        self.sans = pg.font.SysFont("segoeui", 20)
        self.mono = pg.font.SysFont("consolas", 15)
        self.small = pg.font.SysFont("consolas", 13)
        self.overlay = self.ctx.program(vertex_shader=OVERLAY_VERTEX, fragment_shader=OVERLAY_FRAGMENT)
        self.overlay_buffer = self.ctx.buffer(reserve=64)
        self.overlay_vao = self.ctx.vertex_array(self.overlay, [(self.overlay_buffer, "2f 2f", "in_pos", "in_uv")])
        self.overlay_textures = {}

        self.rng = np.random.default_rng()
        self.gallery = load_gallery()
        self.index = 0
        if species:
            names = [r.name.lower() for r in self.gallery]
            self.index = names.index(species.lower()) if species.lower() in names else 0
        rules = self.gallery[self.index] if self.gallery else random_rules(self.rng)
        self.label = "gallery"
        self.world = World(self.ctx, rules, particles, aspect=self.size[0] / self.size[1])
        self.trails = Trails(self.ctx, self.size)
        self.trail_keep = .82
        self.point_size = max(2.0, self.size[1] / 360)

        self.discovery = None
        self.search = None
        self.paused = False
        self.panel = True
        self.matrix_rect = None
        self.message, self.message_until = "", 0.0
        self.frames, self.fps, self.counted = 0, 0.0, time.monotonic()
        self.running = True

    def say(self, text, seconds=4.0):
        self.message, self.message_until = text, time.monotonic() + seconds

    # Species

    def use(self, rules, label, reshuffle=True):
        self.world.set_rules(rules, reshuffle=reshuffle)
        self.label = label
        if reshuffle:
            self.trails.clear()

    def browse(self, step):
        if not self.gallery:
            return
        self.index = (self.index + step) % len(self.gallery)
        self.use(self.gallery[self.index], "gallery")

    def randomise(self):
        self.use(random_rules(self.rng, types=int(self.rng.choice((4, 5, 6, 7)))), "random")

    def save(self):
        DISCOVERED.mkdir(parents=True, exist_ok=True)
        rules = self.world.rules.with_name()
        path = DISCOVERED / f"{rules.name.lower()}.json"
        rules.save(path)
        if all(r.fingerprint() != rules.fingerprint() for r in self.gallery):
            self.gallery.append(rules)
        self.say(f"Saved {rules.name} to species/discovered/")

    def resize(self, particles):
        rules = self.world.rules
        self.world.release()
        self.world = World(self.ctx, rules, particles, aspect=self.size[0] / self.size[1])
        self.trails.clear()
        self.say(f"{particles} particles")

    def edit(self, row, col, delta):
        rules = self.world.rules
        m = rules.matrix.copy()
        m[row, col] = np.clip(round(m[row, col] + delta, 2), -1, 1)
        self.use(replace(rules, matrix=m, name="", score=0.0, notes={}).with_name(), "edited", reshuffle=False)

    # Discovery

    def toggle_discovery(self):
        if self.discovery:
            self.finish_discovery(stopped=True)
            return
        self.discovery = Discovery(self.ctx, seed=int(self.rng.integers(1 << 30)), chunk=10,
                                   population=12, generations=6, elite=3, fresh=3)
        self.search = self.discovery.run()
        self.say("Discovering: evolving rule sets that look alive...", 3)

    def run_discovery(self, budget=.007):
        """Advance the search a few milliseconds per frame so the window stays smooth.

        GPU work is queued asynchronously, so each small chunk is followed by
        finish(): that way the budget counts real GPU time, not just queueing.
        """
        if not self.discovery:
            return
        end = time.perf_counter() + budget
        try:
            while time.perf_counter() < end:
                next(self.search)
                self.ctx.finish()
        except StopIteration:
            self.finish_discovery()

    def finish_discovery(self, stopped=False):
        found = self.discovery.best(5)
        self.discovery.release()
        self.discovery = self.search = None
        if not found:
            self.say("Discovery stopped before any species were scored.")
            return
        DISCOVERED.mkdir(parents=True, exist_ok=True)
        known = {r.fingerprint() for r in self.gallery}
        for rules in found:
            rules.save(DISCOVERED / f"{rules.name.lower()}.json")
            if rules.fingerprint() not in known:
                self.gallery.append(rules)
        best = found[0]
        self.index = next(i for i, r in enumerate(self.gallery) if r.fingerprint() == best.fingerprint())
        self.use(best, "discovered")
        verb = "Stopped early. Best so far" if stopped else "Discovered"
        self.say(f"{verb}: {best.name} (life score {best.score:.2f}). {len(found)} species added to the gallery.", 6)

    # Input

    def world_point(self, pixel):
        return pixel[0] / self.size[0] * self.world.size[0], pixel[1] / self.size[1] * self.world.size[1]

    def matrix_cell(self, pixel):
        if not self.panel or not self.matrix_rect:
            return None
        x0, y0, cell, n = self.matrix_rect
        col, row = int((pixel[0] - x0) // cell), int((pixel[1] - y0) // cell)
        return (row, col) if 0 <= row < n and 0 <= col < n else None

    def handle(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.MOUSEBUTTONDOWN and event.button in (1, 3):
            hit = self.matrix_cell(event.pos)
            if hit:
                step = .5 if pg.key.get_mods() & pg.KMOD_SHIFT else .1
                self.edit(*hit, step if event.button == 1 else -step)
        elif event.type == pg.KEYDOWN:
            key = event.key
            if key == pg.K_ESCAPE:
                self.running = False
            elif key == pg.K_SPACE:
                self.paused = not self.paused
            elif key in (pg.K_RIGHT, pg.K_n):
                self.browse(1)
            elif key in (pg.K_LEFT, pg.K_b):
                self.browse(-1)
            elif key == pg.K_r:
                self.randomise()
            elif key == pg.K_d:
                self.toggle_discovery()
            elif key == pg.K_s:
                self.save()
            elif key == pg.K_c:
                self.world.scatter()
                self.trails.clear()
            elif key == pg.K_t:
                self.trail_keep = 0.0 if self.trail_keep else .82
            elif key == pg.K_h:
                self.panel = not self.panel
            elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                self.resize(PARTICLES[key - pg.K_1])

    def stir(self):
        buttons = pg.mouse.get_pressed()
        pos = pg.mouse.get_pos()
        if (buttons[0] or buttons[2]) and not self.matrix_cell(pos) and not self.over_panel(pos):
            x, y = self.world_point(pos)
            self.world.stir(x, y, 6.0 if buttons[0] else -6.0, .12 * self.world.size[1])
        else:
            self.world.stir(0, 0, 0.0)

    def over_panel(self, pos):
        return self.panel and getattr(self, "panel_rect", None) and pg.Rect(self.panel_rect).collidepoint(pos)

    # Drawing

    def panel_surface(self):
        rules = self.world.rules
        n = rules.types
        cell = 30
        width = max(330, 70 + cell * n)
        surface = pg.Surface((width, 640), pg.SRCALPHA)
        surface.fill(PANEL)
        y = 12
        surface.blit(self.title.render("PRIMORDIAL", True, TEXT), (16, y))
        y += 40
        name = rules.name or "Unnamed"
        surface.blit(self.sans.render(name, True, MINT), (16, y))
        y += 28
        where = f"{self.label}" + (f"  {self.index + 1}/{len(self.gallery)}" if self.label == "gallery" else "")
        score = f"   life score {rules.score:.2f}" if rules.score else ""
        surface.blit(self.small.render(f"{where}   {n} types{score}", True, MUTED), (16, y))
        y += 26

        surface.blit(self.small.render("attraction rules (row is pulled to column)", True, MUTED), (16, y))
        y += 20
        x0 = 40
        for i in range(n):
            color = tuple(int(c * 255) for c in PALETTE[i])
            pg.draw.circle(surface, color, (x0 + i * cell + cell // 2, y + 8), 6)
            pg.draw.circle(surface, color, (24, y + 20 + i * cell + cell // 2), 6)
        y += 20
        for i in range(n):
            for j in range(n):
                rect = pg.Rect(x0 + j * cell + 1, y + i * cell + 1, cell - 2, cell - 2)
                pg.draw.rect(surface, cell_color(rules.matrix[i, j]), rect, border_radius=4)
        self.matrix_rect = (20 + x0, 20 + y, cell, n)
        y += n * cell + 8
        surface.blit(self.small.render("click +0.1  right-click −0.1  shift ×5", True, MUTED), (16, y))
        y += 28

        stats = f"{self.world.n} particles   {self.fps:.0f} fps" + ("   paused" if self.paused else "")
        surface.blit(self.mono.render(stats, True, TEXT), (16, y))
        y += 26

        if self.discovery:
            d = self.discovery
            surface.blit(self.mono.render(f"DISCOVERING  gen {d.generation}/{d.generations}  tested {d.evaluated}", True, ORANGE), (16, y))
            y += 22
            bar = pg.Rect(16, y, width - 32, 6)
            pg.draw.rect(surface, (40, 50, 56), bar, border_radius=3)
            pg.draw.rect(surface, ORANGE, (bar.x, bar.y, bar.w * d.progress(), bar.h), border_radius=3)
            y += 14
            history = d.history
            if len(history) >= 2:
                chart = pg.Rect(16, y, width - 32, 50)
                top = max(h["best"] for h in history) or 1
                for key, color in (("mean", MUTED), ("best", MINT)):
                    points = [(chart.x + chart.w * i / (len(history) - 1), chart.bottom - chart.h * h[key] / top)
                              for i, h in enumerate(history)]
                    pg.draw.lines(surface, color, False, points, 2)
                y += 58
            else:
                y += 6
        keys = ("←/→ gallery   R random   D discover",
                "drag: push   right-drag: pull",
                "S save   C reset   T trails   1-4 count",
                "Space pause   H hide   Esc quit")
        for text in keys:
            surface.blit(self.small.render(text, True, MUTED), (16, y))
            y += 18
        return surface.subsurface((0, 0, width, y + 10)).copy()

    def draw_overlay(self, surface, x, y):
        w, h = surface.get_size()
        texture = self.overlay_textures.get((w, h))
        if texture is None:
            texture = self.overlay_textures[(w, h)] = self.ctx.texture((w, h), 4)
        texture.write(pg.image.tobytes(surface, "RGBA", True))
        sw, sh = self.size
        x0, x1 = -1 + 2 * x / sw, -1 + 2 * (x + w) / sw
        y0, y1 = 1 - 2 * (y + h) / sh, 1 - 2 * y / sh
        self.overlay_buffer.write(np.array([x0, y0, 0, 0, x1, y0, 1, 0, x0, y1, 0, 1, x1, y1, 1, 1], "f4").tobytes())
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        texture.use(0)
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def draw(self):
        texture = self.trails.render(self.world, self.point_size, self.trail_keep)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.trails.present(texture)
        if self.panel:
            surface = self.panel_surface()
            self.panel_rect = (20, 20, *surface.get_size())
            self.draw_overlay(surface, 20, 20)
        if time.monotonic() < self.message_until:
            label = self.sans.render(self.message, True, TEXT)
            box = pg.Surface((label.get_width() + 32, label.get_height() + 16), pg.SRCALPHA)
            box.fill(PANEL)
            box.blit(label, (16, 8))
            self.draw_overlay(box, (self.size[0] - box.get_width()) // 2, self.size[1] - box.get_height() - 40)
        pg.display.flip()

    def update(self):
        for event in pg.event.get():
            self.handle(event)
        self.stir()
        if not self.paused:
            self.world.step()
        self.run_discovery()
        self.draw()
        self.frames += 1
        now = time.monotonic()
        if now - self.counted >= 1:
            self.fps, self.frames, self.counted = self.frames / (now - self.counted), 0, now

    def loop(self):
        self.say("Drag to stir, ←/→ to browse species, D to let evolution discover new ones.", 6)
        while self.running:
            self.update()
        if self.discovery:
            self.discovery.release()
        pg.quit()


# Headless tools


def headless_discover(generations, seed):
    ctx = moderngl.create_standalone_context(require=430)
    d = Discovery(ctx, seed=seed, generations=generations)
    start = time.perf_counter()
    shown = 0

    def report():
        nonlocal shown
        for h in d.history[shown:]:
            print(f"gen {h['generation']:>2}  best {h['best']:.3f}  mean {h['mean']:.3f}  tested {d.evaluated}")
        shown = len(d.history)

    for _ in d.run():
        report()
    report()
    DISCOVERED.mkdir(parents=True, exist_ok=True)
    print(f"\n{d.evaluated} species tested in {time.perf_counter() - start:.0f} s. Best:")
    for rules in d.best(5):
        rules.save(DISCOVERED / f"{rules.name.lower()}.json")
        n = rules.notes
        print(f"  {rules.name:<10} score {rules.score:.3f}  organisms {n['clusters']:>2}  "
              f"mixing {n['mixing']:.2f}  travel {n['travel']:.2f}")
    print(f"Saved to {DISCOVERED}")


def render_species(ctx, rules, size, particles, steps, seed=1, point_size=None):
    from PIL import Image
    world = World(ctx, rules, particles, seed=seed, aspect=size[0] / size[1])
    trails = Trails(ctx, size)
    world.step(steps - 12)
    texture = None
    for _ in range(12):
        world.step()
        texture = trails.render(world, point_size or max(2.0, size[1] / 360), .82)
    fbo = ctx.simple_framebuffer(size)
    fbo.use()
    trails.present(texture)
    image = Image.frombytes("RGB", size, fbo.read()).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    for item in (fbo, trails, world):
        item.release()
    return image


def export_gallery(path, particles):
    from PIL import Image, ImageDraw, ImageFont
    ctx = moderngl.create_standalone_context(require=430)
    gallery = load_gallery()[:12]
    w, h, cols = 640, 360, 5
    rows = (len(gallery) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 8, rows * h + (rows - 1) * 8), (13, 18, 22))
    try:
        font = ImageFont.truetype("segoeui.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, rules in enumerate(gallery):
        image = render_species(ctx, rules, (w, h), particles // 2, 900, point_size=2.2)
        ImageDraw.Draw(image).text((14, 8), rules.name, fill=(231, 234, 223), font=font)
        sheet.paste(image, ((i % cols) * (w + 8), (i // cols) * (h + 8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print("Saved", path)


def export_gif(path, species, particles, frames=90):
    from PIL import Image
    ctx = moderngl.create_standalone_context(require=430)
    gallery = load_gallery()
    rules = next((r for r in gallery if r.name.lower() == (species or "").lower()), gallery[0])
    size = (640, 360)
    world = World(ctx, rules, particles // 2, seed=3, aspect=16 / 9)
    trails = Trails(ctx, size)
    fbo = ctx.simple_framebuffer(size)
    world.step(600)
    images = []
    for _ in range(frames):
        for _ in range(2):
            world.step()
            texture = trails.render(world, 2.2, .82)
        fbo.use()
        trails.present(texture)
        image = Image.frombytes("RGB", size, fbo.read()).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        images.append(image.quantize(96))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=40, loop=0, optimize=True)
    print("Saved", path)


def dpi_aware():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Particle life on the GPU, with evolution discovering life-like species.")
    parser.add_argument("--species", help="Start with a gallery species by name.")
    parser.add_argument("--particles", type=int, choices=PARTICLES, default=8000)
    parser.add_argument("--windowed", action="store_true", help="Run in a window instead of fullscreen.")
    parser.add_argument("--discover", type=int, metavar="GENERATIONS", help="Search for species without a window, then exit.")
    parser.add_argument("--seed", type=int, help="Seed for --discover.")
    parser.add_argument("--gallery", type=Path, help="Render every gallery species into one image and exit.")
    parser.add_argument("--export-gif", type=Path, help="Render an animation of --species and exit.")
    args = parser.parse_args()

    if args.discover:
        headless_discover(args.discover, args.seed)
    elif args.gallery:
        export_gallery(args.gallery, args.particles)
    elif args.export_gif:
        export_gif(args.export_gif, args.species, args.particles)
    else:
        dpi_aware()
        App(args.particles, args.windowed, args.species).loop()


if __name__ == "__main__":
    main()
