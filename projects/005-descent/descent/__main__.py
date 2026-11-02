import argparse
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg
import numpy as np

from .brain import INPUT_LABELS, unpack
from .evolution import SOLIDARITY, Run

BG = (13, 18, 22)
PANEL = (20, 27, 32)
CELL = (18, 25, 29)
LINE = (45, 57, 63)
TEXT = (231, 234, 223)
MUTED = (138, 157, 159)
ORANGE = (255, 139, 92)
MINT = (194, 235, 180)
TEAL = (92, 214, 196)
GOLD = (240, 200, 110)
DEAD = (60, 66, 70)
SPEEDS = (.25, 2, 20, 200)
SOLIDARITY_NAMES = {0.0: "none", .5: "partial", 1.0: "full"}


def blend(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def signal(value):
    """Colour for an activation or weight: orange negative, teal positive."""
    value = max(-1.0, min(1.0, float(value)))
    return blend(LINE, TEAL if value > 0 else ORANGE, abs(value))


def appetite(portion, max_portion):
    """Teal for about one ration, orange for taking everything allowed."""
    return blend(TEAL, ORANGE, (portion - 1) / (max_portion - 1))


class App:
    def __init__(self, size=(1180, 720), seed=None, run=None):
        pg.init()
        pg.display.set_caption("Descent")
        self.screen = pg.display.set_mode(size)
        self.clock = pg.time.Clock()
        self.sans = pg.font.SysFont("segoeui", 18)
        self.title = pg.font.SysFont("segoeui", 34, bold=True)
        self.mono = pg.font.SysFont("consolas", 15)
        self.small = pg.font.SysFont("consolas", 12)
        self.run = run or Run(seed)
        self.paused = False
        self.speed = 0
        self.carry = 0.0
        self.tower = 0
        self.selected = None
        self.running = True
        self.layout()

    def layout(self):
        w, h = self.screen.get_size()
        c = self.run.config
        self.panel = pg.Rect(w - 420, 0, 420, h)
        area = pg.Rect(20, 100, self.panel.x - 40, h - 100 - 36)
        self.gap = 6
        self.col_w = (area.w - self.gap * (c.towers - 1)) / c.towers
        self.row_h = area.h / c.levels
        self.area = area

    def column(self, t):
        return pg.Rect(self.area.x + t * (self.col_w + self.gap), self.area.y, self.col_w, self.area.h)

    def seat(self, t, i):
        """Screen position of an inmate: two seats per level, left and right."""
        c, towers = self.run.config, self.run.towers
        col = self.column(t)
        level = towers.level[t, i]
        same = np.flatnonzero(towers.level[t] == level)
        side = int(np.searchsorted(same, i))
        x = col.x + col.w * (.22 + .2 * side)
        y = col.y + self.row_h * (level + .5)
        return x, y

    def text(self, value, pos, color=TEXT, font=None):
        self.screen.blit((font or self.sans).render(str(value), True, color), pos)

    # Simulation control

    def advance(self):
        self.carry += SPEEDS[self.speed]
        while self.carry >= 1:
            self.carry -= 1
            generation = self.run.generation
            self.run.step()
            if self.run.generation != generation:
                self.selected = None

    def skip_generation(self):
        self.run.run_generation()
        self.selected = None

    def cycle_solidarity(self):
        index = SOLIDARITY.index(self.run.solidarity)
        self.run.solidarity = SOLIDARITY[(index + 1) % len(SOLIDARITY)]

    def focus(self):
        """The inmate whose brain is shown: the clicked one, else the last living inmate the platform served."""
        towers = self.run.towers
        if self.selected is not None:
            return self.selected
        t = self.tower
        served = towers.alive[t] & (towers.level[t] < (towers.stop or self.run.config.levels))
        if not served.any():
            served = towers.alive[t]
        if not served.any():
            return None
        candidates = np.flatnonzero(served)
        return t, int(candidates[np.argmax(towers.level[t, candidates])])

    def click(self, pos):
        c = self.run.config
        for t in range(c.towers):
            if self.column(t).collidepoint(pos):
                self.tower = t
                seats = np.array([self.seat(t, i) for i in range(c.inmates)])
                dist = np.hypot(seats[:, 0] - pos[0], seats[:, 1] - pos[1])
                i = int(np.argmin(dist))
                self.selected = (t, i) if dist[i] < 12 else None
                return

    def handle(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.click(event.pos)
        elif event.type == pg.KEYDOWN:
            towers = self.run.config.towers
            if event.key == pg.K_ESCAPE:
                self.running = False
            elif event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                self.speed = event.key - pg.K_1
            elif event.key == pg.K_g:
                self.skip_generation()
            elif event.key == pg.K_s:
                self.cycle_solidarity()
            elif event.key == pg.K_LEFT:
                self.tower, self.selected = (self.tower - 1) % towers, None
            elif event.key == pg.K_RIGHT:
                self.tower, self.selected = (self.tower + 1) % towers, None
            elif event.key == pg.K_r:
                self.run = Run(solidarity=self.run.solidarity)
                self.selected = None
            elif event.key == pg.K_PERIOD and self.paused:
                self.run.step()

    # Drawing

    def draw_towers(self):
        run, c = self.run, self.run.config
        towers = run.towers
        focus = self.focus()
        stop = towers.stop
        for t in range(c.towers):
            col = self.column(t)
            alive = towers.alive[t].mean()
            header = MINT if t == self.tower else MUTED
            self.text(f"T{t + 1}", (col.x + 4, col.y - 40), header, self.mono)
            self.text(f"{alive:.0%}", (col.x + 4, col.y - 22), header, self.small)
            pg.draw.rect(self.screen, CELL, col)
            if t == self.tower:
                pg.draw.rect(self.screen, LINE, col.inflate(4, 4), 1)
            for level in range(c.levels):
                y = col.y + level * self.row_h
                pg.draw.line(self.screen, BG, (col.x, y), (col.right, y))
                if level < stop or towers.day:
                    food = towers.arrived[t, level] / c.feast
                    bar = pg.Rect(col.x + col.w * .62, y + self.row_h * .38, col.w * .32 * food, self.row_h * .24)
                    pg.draw.rect(self.screen, blend(CELL, GOLD, .45), bar)
            for i in range(c.inmates):
                x, y = self.seat(t, i)
                if towers.alive[t, i]:
                    radius = 3 + towers.energy[t, i] / c.max_energy * 4
                    pg.draw.circle(self.screen, appetite(towers.portion[t, i], c.max_portion), (x, y), radius)
                else:
                    pg.draw.line(self.screen, DEAD, (x - 3, y - 3), (x + 3, y + 3), 2)
                    pg.draw.line(self.screen, DEAD, (x - 3, y + 3), (x + 3, y - 3), 2)
                if focus == (t, i):
                    pg.draw.circle(self.screen, TEXT, (x, y), 10, 1)
            # The platform, between this stop and the next.
            py = col.y + (stop + min(self.carry, 1) * (SPEEDS[self.speed] < 1)) * self.row_h
            plate = pg.Rect(col.x + 2, py - 3, col.w - 4, 6)
            pg.draw.rect(self.screen, LINE, plate, border_radius=3)
            fill = plate.copy()
            fill.w = int(plate.w * towers.food[t] / c.feast)
            pg.draw.rect(self.screen, GOLD, fill, border_radius=3)

        self.text(f"GENERATION {run.generation}   DAY {towers.day + 1}/{c.days}", (20, 14), MINT, self.mono)
        month = towers.day // c.month + 1
        self.text(f"month {month} — levels reshuffle every {c.month} days", (20, 32), MUTED, self.small)
        self.text("level 1", (self.area.x - 2, self.area.bottom + 8), MUTED, self.small)
        self.text(f"bottom = level {c.levels}", (self.area.x + 60, self.area.bottom + 8), MUTED, self.small)

    def draw_stats(self, x, y):
        run, towers, c = self.run, self.run.towers, self.run.config
        bottom = towers.arrived[:, -1].mean() if towers.day else 0
        rows = (
            ("solidarity", SOLIDARITY_NAMES[run.solidarity]),
            ("alive", f"{towers.alive.mean():.0%}"),
            ("food at bottom", f"{bottom:.1f}"),
            ("top greed", f"{towers.greed(slice(0, c.levels // 4)):.2f}"),
            ("speed", (f"{SPEEDS[self.speed]:g}" + " stops/frame") + ("  paused" if self.paused else "")),
        )
        for i, (label, value) in enumerate(rows):
            col, row = i % 3, i // 3
            self.text(label, (x + col * 130, y + row * 44), MUTED, self.small)
            self.text(value, (x + col * 130, y + row * 44 + 14), TEXT, self.mono)

    def draw_graph(self, rect):
        pg.draw.rect(self.screen, BG, rect, border_radius=6)
        history = self.run.history[-120:]
        self.text("per generation", (rect.x + 10, rect.y + 6), MUTED, self.small)
        self.text("survival", (rect.x + 130, rect.y + 6), MINT, self.small)
        self.text("top-level portion", (rect.x + 205, rect.y + 6), ORANGE, self.small)
        if len(history) < 2:
            self.text("finishing the first generation...", (rect.x + 10, rect.centery - 8), MUTED, self.small)
            return
        inner = rect.inflate(-24, -36).move(0, 8)
        top = self.run.config.max_portion

        def line(values, color, width):
            points = [(inner.x + inner.w * i / (len(values) - 1), inner.bottom - inner.h * v)
                      for i, v in enumerate(values)]
            pg.draw.lines(self.screen, color, False, points, width)

        fair = inner.bottom - inner.h / top
        pg.draw.line(self.screen, LINE, (inner.x, fair), (inner.right, fair))
        self.text("1 ration", (inner.x, fair + 2), MUTED, self.small)
        line([h["greed_top"] / top for h in history], ORANGE, 2)
        line([h["survival"] for h in history], MINT, 2)

    def draw_brain(self, rect):
        pg.draw.rect(self.screen, BG, rect, border_radius=6)
        self.text("brain of the ringed inmate", (rect.x + 10, rect.y + 6), MUTED, self.small)
        focus = self.focus()
        if focus is None:
            self.text("click an inmate", (rect.x + 10, rect.centery), MUTED, self.small)
            return
        towers, c = self.run.towers, self.run.config
        t, i = focus
        w1, _, w2, _ = (part[0] for part in unpack(towers.genomes[t, i]))
        portion = towers.portion[t, i]
        acts = (towers.inputs[t, i], towers.hidden[t, i], [portion / c.max_portion * 2 - 1])
        columns = (rect.x + 120, rect.centerx + 30, rect.right - 130)
        top, height = rect.y + 28, rect.h - 54

        def spots(count, column):
            gap = height / count
            return [(column, top + gap * (k + .5)) for k in range(count)]

        nodes = [spots(len(a), col) for a, col in zip(acts, columns)]
        for weights, starts, ends in ((w1, nodes[0], nodes[1]), (w2, nodes[1], nodes[2])):
            for a, start in enumerate(starts):
                for b, end in enumerate(ends):
                    w = weights[a, b]
                    if abs(w) > .2:
                        pg.draw.line(self.screen, signal(w / 2), start, end, 1 + int(abs(w) > 1))
        for values, points in zip(acts, nodes):
            for value, point in zip(values, points):
                pg.draw.circle(self.screen, signal(value), point, 7)
                pg.draw.circle(self.screen, LINE, point, 7, 1)
        for name, value, (px, py) in zip(INPUT_LABELS, acts[0], nodes[0]):
            surf = self.small.render(f"{name} {value:.2f}", True, MUTED)
            self.screen.blit(surf, (px - 14 - surf.get_width(), py - 7))
        px, py = nodes[2][0]
        self.text(f"eat {portion:.2f}", (px + 14, py - 7), appetite(portion, c.max_portion), self.mono)
        status = "alive" if towers.alive[t, i] else "starved"
        info = f"tower {t + 1}  level {towers.level[t, i] + 1}  energy {towers.energy[t, i]:.1f}  {status}"
        self.text(info, (rect.x + 10, rect.bottom - 20), MUTED, self.small)

    def draw(self):
        self.screen.fill(BG)
        self.draw_towers()
        pg.draw.rect(self.screen, PANEL, self.panel)
        x = self.panel.x + 24
        self.text("DESCENT", (x, 16), font=self.title)
        self.text("one feast, many levels. does sharing evolve?", (x, 60), MUTED)
        self.draw_stats(x, 98)
        width = self.panel.w - 48
        self.draw_graph(pg.Rect(x, 192, width, 140))
        self.draw_brain(pg.Rect(x, 346, width, self.panel.h - 346 - 74))
        keys = ("Space pause   1-4 speed   G skip gen   . step",
                "S solidarity   ←/→ tower   click inmate   R restart")
        for i, line in enumerate(keys):
            self.text(line, (x, self.panel.h - 58 + i * 20), MUTED, self.small)
        pg.display.flip()

    def update(self):
        for event in pg.event.get():
            self.handle(event)
        if not self.paused:
            self.advance()
        self.draw()

    def loop(self):
        while self.running:
            self.update()
            self.clock.tick(60)
        pg.quit()


def train(run, generations, quiet=False):
    for _ in range(generations):
        run.run_generation()
        h = run.history[-1]
        if not quiet:
            print(f"gen {h['generation']:>4}  survival {h['survival']:.2f}  alive {h['alive']:.2f}  "
                  f"top portion {h['greed_top']:.2f}  bottom portion {h['greed_bottom']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Neural-net inmates sharing one descending feast.")
    parser.add_argument("--seed", type=int, help="Repeat the same run.")
    parser.add_argument("--solidarity", type=float, choices=SOLIDARITY, default=0.0,
                        help="How much selection acts on whole towers instead of individuals.")
    parser.add_argument("--train", type=int, default=0, metavar="N",
                        help="Evolve N generations before opening the window.")
    parser.add_argument("--headless", action="store_true", help="Train without a window, then exit.")
    parser.add_argument("--screenshot", type=Path, help="Save one window preview and exit.")
    args = parser.parse_args()

    run = Run(args.seed, solidarity=args.solidarity)
    train(run, args.train)
    if args.headless:
        return
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = App(run=run)
        for _ in range(run.config.levels * 12 + 11):
            run.towers.tick()
        app.tower, app.paused = 2, True
        app.draw()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pg.image.save(app.screen, args.screenshot)
        pg.quit()
        print("Saved", args.screenshot)
        return
    App(run=run).loop()


if __name__ == "__main__":
    main()
