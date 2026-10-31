import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg
import numpy as np

from .engine import PixelProgram, AssemblyError, ExecutionError, WIDTH, HEIGHT
from .presets import PRESETS
from .visuals import rgb, export_gif

BG = (13, 18, 22)
PANEL = (20, 27, 32)
LINE = (45, 57, 63)
TEXT = (231, 234, 223)
MUTED = (138, 157, 159)
ORANGE = (255, 139, 92)
MINT = (194, 235, 180)


class Editor:
    def __init__(self, text):
        self.text = text
        self.cursor = 0
        self.scroll = 0
        self.focus = False
        self.rect = pg.Rect(0, 0, 1, 1)
        self.undo = []

    def replace(self, text):
        self.undo.append((self.text, self.cursor))
        self.undo = self.undo[-80:]
        self.text, self.cursor = text, min(self.cursor, len(text))

    def insert(self, text):
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
        text = text[:max(0, 12000 - len(self.text))]
        at = self.cursor
        self.replace(self.text[:at] + text + self.text[at:])
        self.cursor = at + len(text)

    def location(self):
        before = self.text[:self.cursor]
        return before.count("\n"), len(before.rsplit("\n", 1)[-1])

    def click(self, pos, char_width):
        self.focus = self.rect.collidepoint(pos)
        if self.focus:
            lines = self.text.split("\n")
            row = max(0, min(len(lines) - 1, (pos[1] - self.rect.y - 16) // 24 + self.scroll))
            col = max(0, min(len(lines[row]), round((pos[0] - self.rect.x - 52) / char_width)))
            self.cursor = sum(len(s) + 1 for s in lines[:row]) + col

    def key(self, event):
        ctrl = bool(event.mod & pg.KMOD_CTRL)
        if ctrl and event.key == pg.K_z:
            if self.undo:
                self.text, self.cursor = self.undo.pop()
        elif ctrl and event.key == pg.K_v:
            try:
                self.insert(pg.scrap.get_text())
            except pg.error:
                pass
        elif ctrl and event.key == pg.K_c:
            pg.scrap.put_text(self.text)
        elif event.key == pg.K_BACKSPACE and self.cursor:
            at = self.cursor
            self.replace(self.text[:at - 1] + self.text[at:])
            self.cursor = at - 1
        elif event.key == pg.K_DELETE:
            self.replace(self.text[:self.cursor] + self.text[self.cursor + 1:])
        elif event.key == pg.K_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif event.key == pg.K_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
        elif event.key in (pg.K_UP, pg.K_DOWN, pg.K_HOME, pg.K_END):
            row, col = self.location()
            lines = self.text.split("\n")
            if event.key == pg.K_UP:
                row = max(0, row - 1)
            elif event.key == pg.K_DOWN:
                row = min(len(lines) - 1, row + 1)
            elif event.key == pg.K_HOME:
                col = 0
            else:
                col = len(lines[row])
            self.cursor = sum(len(s) + 1 for s in lines[:row]) + min(col, len(lines[row]))
        elif event.key == pg.K_RETURN:
            self.insert("\n")
        elif event.key == pg.K_TAB:
            self.insert("    ")


class App:
    def __init__(self, size=(1380, 900)):
        pg.init()
        pg.display.set_caption("Opcode Canvas / Day 004")
        self.screen = pg.display.set_mode(size, pg.RESIZABLE)
        self.sans = pg.font.SysFont("segoeui", 19)
        self.title = pg.font.SysFont("segoeui", 37, bold=True)
        self.mono = pg.font.SysFont("consolas", 16)
        self.small = pg.font.SysFont("consolas", 13)
        self.editor = Editor(PRESETS[0].source)
        self.preset = 0
        self.program = PixelProgram(self.editor.text)
        self.compiled_source = self.editor.text
        self.tick = 0
        self.paused = False
        self.speed = 1
        self.pixel = (64, 40)
        self.debug = self.program.debugger(*self.pixel, self.tick)
        self.previous = self.debug.snapshot()
        self.frame = self.program.render(0)
        self.status = "Ready. Edit a constant or instruction, then run."
        self.error = ""
        self.buttons = {}
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = None
        self.export = None
        self.export_path = None
        self.want_frame = False
        self.running = True
        self.canvas_rect = pg.Rect(0, 0, 1, 1)
        pg.key.set_repeat(400, 35)
        pg.key.start_text_input()

    def text(self, value, pos, color=TEXT, font=None):
        self.screen.blit((font or self.sans).render(str(value), True, color), pos)

    def button(self, name, label, rect, active=False):
        rect = pg.Rect(rect)
        self.buttons[name] = rect
        hover = rect.collidepoint(pg.mouse.get_pos())
        color = MINT if active else ((38, 48, 54) if hover else PANEL)
        pg.draw.rect(self.screen, color, rect, border_radius=6)
        pg.draw.rect(self.screen, LINE, rect, 1, border_radius=6)
        rendered = self.small.render(label, True, BG if active else TEXT)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def reset_debug(self):
        self.debug = self.program.debugger(*self.pixel, self.tick)
        self.previous = self.debug.snapshot()

    def compile(self):
        try:
            program = PixelProgram(self.editor.text)
        except AssemblyError as exc:
            self.error = str(exc)
            return
        self.program = program
        self.compiled_source = self.editor.text
        self.error = ""
        self.status = f"Assembled {len(program.shader)} bytes. Running your instructions."
        self.reset_debug()
        self.want_frame = True

    def action(self, name):
        if name.startswith("preset"):
            self.preset = int(name[-1])
            self.editor.replace(PRESETS[self.preset].source)
            self.editor.cursor, self.editor.scroll = 0, 0
            self.tick = 0
            self.compile()
        elif name == "run":
            self.compile()
        elif name == "pause":
            self.paused = not self.paused
            self.reset_debug()
        elif name == "frame":
            self.paused = True
            self.tick += self.speed
            self.want_frame = True
            self.reset_debug()
        elif name == "reset":
            self.tick = 0
            self.want_frame = True
            self.reset_debug()
        elif name == "speed":
            self.speed = {1: 2, 2: 4, 4: 1}[self.speed]
        elif name == "step":
            self.paused = True
            try:
                self.previous = self.debug.snapshot()
                self.debug.step()
                self.status = (f"Pixel returned colour index {self.debug.snapshot()['EAX'] & 255}."
                               if self.debug.done else f"Executed {self.debug.steps} instruction(s).")
                self.error = ""
            except ExecutionError as exc:
                self.error = str(exc)
        elif name == "debug-reset":
            self.reset_debug()
            self.status = "Pixel CPU reset. Step executes one real x86 instruction."
        elif name == "gif" and self.export is None:
            self.export_path = Path("exports") / f"opcode-{datetime.now():%Y%m%d-%H%M%S-%f}.gif"
            self.export = self.pool.submit(export_gif, self.compiled_source, self.export_path, self.tick)
            self.status = "Rendering 48 assembly frames for GIF export..."
        elif name == "save":
            path = Path("exports") / f"shader-{datetime.now():%Y%m%d-%H%M%S-%f}.asm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.editor.text, encoding="utf-8")
            self.status = f"Saved {path}"

    def update(self):
        if self.pending and self.pending[0].done():
            task, program, tick = self.pending
            self.pending = None
            try:
                frame = task.result()
                if program is self.program:
                    self.frame = frame
                    if not self.paused:
                        self.tick = tick + self.speed
            except ExecutionError as exc:
                if program is self.program:
                    self.error = str(exc)
                    self.paused = True
        if not self.pending and (not self.paused or self.want_frame):
            self.want_frame = False
            self.pending = (self.pool.submit(self.program.render, self.tick), self.program, self.tick)
        if self.export and self.export.done():
            try:
                path = self.export.result()
                self.status = f"GIF saved: {path}"
            except Exception as exc:
                self.error = f"Export failed: {exc}"
            self.export = None

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.VIDEORESIZE:
                self.screen = pg.display.set_mode((max(1040, event.w), max(780, event.h)), pg.RESIZABLE)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self.editor.click(event.pos, self.mono.size("M")[0])
                for name, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.action(name)
                        break
                else:
                    if self.canvas_rect.collidepoint(event.pos):
                        self.pixel = (min(WIDTH - 1, (event.pos[0] - self.canvas_rect.x) * WIDTH // self.canvas_rect.w),
                                      min(HEIGHT - 1, (event.pos[1] - self.canvas_rect.y) * HEIGHT // self.canvas_rect.h))
                        self.paused = True
                        self.tick = self.frame.tick
                        self.reset_debug()
                        self.status = f"Inspecting pixel {self.pixel}. Use Step instruction below."
            elif event.type == pg.MOUSEWHEEL and self.editor.rect.collidepoint(pg.mouse.get_pos()):
                self.editor.scroll = max(0, min(len(self.editor.text.splitlines()) - 1,
                                                self.editor.scroll - event.y * 3))
            elif event.type == pg.TEXTINPUT and self.editor.focus:
                self.editor.insert(event.text)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN and event.mod & pg.KMOD_CTRL:
                    self.compile()
                elif event.key == pg.K_ESCAPE:
                    self.editor.focus = False
                elif self.editor.focus:
                    self.editor.key(event)
                elif event.key == pg.K_SPACE:
                    self.action("pause")

    def draw(self):
        s = self.screen
        w, h = s.get_size()
        s.fill(BG)
        self.buttons = {}
        self.text("OP /", (28, 20), ORANGE, self.title)
        self.text("OPCODE CANVAS", (150, 25), font=self.title)
        self.text("004 / PYTHON + X86 ASSEMBLY", (w - 283, 33), MUTED, self.small)
        pg.draw.line(s, LINE, (28, 82), (w - 28, 82))
        self.text(PRESETS[self.preset].caption, (28, 99), MUTED)
        for i, preset in enumerate(PRESETS):
            self.button(f"preset{i}", f"0{i + 1}  {preset.name}", (28 + i * 167, 137, 157, 35), self.preset == i)
        self.button("save", "Save .asm", (w - 258, 137, 105, 35))
        self.button("gif", "Exporting..." if self.export else "Export GIF", (w - 143, 137, 115, 35))
        left_w = max(380, int(w * .34))
        panel_y, panel_h = 192, h - 340
        right_x, right_w = left_w + 44, w - left_w - 72
        pg.draw.rect(s, PANEL, (28, panel_y, left_w, panel_h), border_radius=8)
        self.text("PROGRAM / INTEL SYNTAX", (44, panel_y + 15), MUTED, self.small)
        dirty = self.editor.text != self.compiled_source
        self.button("run", "Run *" if dirty else "Run", (left_w - 48, panel_y + 9, 64, 29), True)
        self.editor.rect = pg.Rect(28, panel_y + 51, left_w, panel_h - 104)
        old_clip = s.get_clip()
        s.set_clip(self.editor.rect)
        row, col = self.editor.location()
        visible = max(1, (self.editor.rect.h - 30) // 24)
        if self.editor.focus:
            self.editor.scroll = min(self.editor.scroll, row)
            self.editor.scroll = max(self.editor.scroll, row - visible + 1)
        for n, line in enumerate(self.editor.text.split("\n")):
            y = self.editor.rect.y + 16 + (n - self.editor.scroll) * 24
            self.text(f"{n + 1:02}", (42, y), (81, 103, 110), self.small)
            self.text(line, (80, y), MUTED if line.lstrip().startswith(";") else MINT, self.mono)
        if self.editor.focus and int(time.monotonic() * 2) % 2 == 0:
            x = 80 + col * self.mono.size("M")[0]
            y = self.editor.rect.y + 16 + (row - self.editor.scroll) * 24
            pg.draw.line(s, ORANGE, (x, y), (x, y + 18), 2)
        s.set_clip(old_clip)
        self.text("CTRL+ENTER run  /  ESC leave editor", (44, panel_y + panel_h - 43), MUTED, self.small)
        self.text(f"{len(self.program.shader)} bytes  /  x86-32  /  emulated CPU", (44, panel_y + panel_h - 24), MUTED, self.small)
        self.text("FRAMEBUFFER / EVERY PIXEL COMPUTED IN ASSEMBLY", (right_x, panel_y + 8), MUTED, self.small)
        available = pg.Rect(right_x, panel_y + 40, right_w, panel_h - 130)
        scale = min(available.w / WIDTH, available.h / HEIGHT)
        self.canvas_rect = pg.Rect(0, 0, int(WIDTH * scale), int(HEIGHT * scale))
        self.canvas_rect.center = available.center
        surface = pg.surfarray.make_surface(np.transpose(rgb(self.frame.pixels), (1, 0, 2)))
        s.blit(pg.transform.scale(surface, self.canvas_rect.size), self.canvas_rect)
        if self.paused:
            cx = self.canvas_rect.x + int((self.pixel[0] + .5) * scale)
            cy = self.canvas_rect.y + int((self.pixel[1] + .5) * scale)
            pg.draw.circle(s, TEXT, (cx, cy), 9, 1)
            pg.draw.line(s, TEXT, (cx - 14, cy), (cx + 14, cy))
            pg.draw.line(s, TEXT, (cx, cy - 14), (cx, cy + 14))
        y = panel_y + panel_h - 75
        self.button("pause", "Play" if self.paused else "Pause", (right_x, y, 78, 31))
        self.button("frame", "Frame +", (right_x + 88, y, 84, 31))
        self.button("reset", "Reset", (right_x + 182, y, 76, 31))
        self.button("speed", f"{self.speed}x", (right_x + 268, y, 53, 31))
        self.text(f"t {self.frame.tick:05d} / {self.frame.milliseconds:5.1f} ms", (right_x + 336, y + 8), MUTED, self.small)
        self.text("128 x 80  /  click a pixel to inspect", (right_x, y + 43), MUTED, self.small)
        strip_y = h - 128
        pg.draw.line(s, LINE, (28, strip_y - 13), (w - 28, strip_y - 13))
        self.text(f"PIXEL CPU  [{self.pixel[0]:03}, {self.pixel[1]:03}]", (28, strip_y), ORANGE, self.small)
        self.button("step", "Step instruction", (28, strip_y + 27, 151, 31))
        self.button("debug-reset", "Reset CPU", (189, strip_y + 27, 98, 31))
        values = self.debug.snapshot()
        for i, name in enumerate(("EAX", "EBX", "ECX", "EDX")):
            x = 326 + i * ((w - 354) // 4)
            self.text(name, (x, strip_y), MUTED, self.small)
            changed = values[name] != self.previous[name]
            self.text(f"{values[name]:08X}", (x, strip_y + 25), ORANGE if changed else MINT, self.mono)
        info = (f"RET / colour {values['EAX'] & 255}" if self.debug.done else
                f"NEXT: {self.debug.instruction()}  /  EIP {values['EIP']:08X}  /  {self.debug.steps} steps")
        self.text(info, (326, strip_y + 55), MUTED, self.small)
        message = self.error or self.status
        while self.small.size(message)[0] > w - 56 and len(message) > 5:
            message = message[:-4] + "..."
        self.text(message, (28, h - 31), ORANGE if self.error else MUTED, self.small)
        pg.display.flip()

    def run(self):
        clock = pg.time.Clock()
        try:
            while self.running:
                self.events()
                self.update()
                self.draw()
                clock.tick(30)
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            pg.quit()


def main():
    parser = argparse.ArgumentParser(description="An x86 assembly pixel playground.")
    parser.add_argument("--preset", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--load", type=Path, help="Open a saved .asm program.")
    parser.add_argument("--screenshot", type=Path, help="Save one full-window preview and exit.")
    parser.add_argument("--export-gif", type=Path, help="Render a GIF without opening a window.")
    args = parser.parse_args()
    source = args.load.read_text(encoding="utf-8") if args.load else PRESETS[args.preset - 1].source
    if args.export_gif:
        print(export_gif(source, args.export_gif))
        return
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    app = App()
    app.preset = args.preset - 1
    app.editor.text = source
    app.compile()
    if args.screenshot:
        app.frame = app.program.render(27)
        app.tick = 27
        app.reset_debug()
        app.draw()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pg.image.save(app.screen, str(args.screenshot))
        app.pool.shutdown()
        pg.quit()
    else:
        app.run()


if __name__ == "__main__":
    main()
