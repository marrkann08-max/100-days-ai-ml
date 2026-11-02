import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg
import numpy as np

from .brain import INPUT_LABELS, OUTPUT_LABELS, unpack
from .evolution import Run
from .world import World, wall_distance

BG = (13, 18, 22)
PANEL = (20, 27, 32)
DISH = (17, 25, 29)
LINE = (45, 57, 63)
TEXT = (231, 234, 223)
MUTED = (138, 157, 159)
ORANGE = (255, 139, 92)
MINT = (194, 235, 180)
TEAL = (92, 214, 196)
FOOD = (160, 230, 120)
POISON = (235, 84, 120)
SPEEDS = (1, 4, 12, 40)


def blend(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def signal(value):
    """Colour for an activation or weight: orange negative, teal positive."""
    value = max(-1.0, min(1.0, float(value)))
    return blend(LINE, TEAL if value > 0 else ORANGE, abs(value))


class App:
    def __init__(self, size=(1180, 720), seed=None, run=None):
        pg.init()
        pg.display.set_caption("Petri Minds")
        self.screen = pg.display.set_mode(size)
        self.clock = pg.time.Clock()
        self.sans = pg.font.SysFont("segoeui", 18)
        self.title = pg.font.SysFont("segoeui", 34, bold=True)
        self.mono = pg.font.SysFont("consolas", 15)
        self.small = pg.font.SysFont("consolas", 12)
        self.run = run or Run(seed)
        self.paused = False
        self.speed = 0
        self.selected = None
        self.replay = None
        self.running = True
        self.layout()

    def layout(self):
        w, h = self.screen.get_size()
        side = min(h, w - 460)
        self.dish = pg.Rect(0, 0, side, side)
        self.dish.center = (side // 2, h // 2)
        self.scale = (side / 2 - 18) / self.run.config.radius
        self.panel = pg.Rect(side, 0, w - side, h)

    @property
    def world(self):
        return self.replay or self.run.world

    def to_screen(self, pos):
        return self.dish.centerx + pos[0] * self.scale, self.dish.centery + pos[1] * self.scale

    def text(self, value, pos, color=TEXT, font=None):
        self.screen.blit((font or self.sans).render(str(value), True, color), pos)

    # Simulation control

    def start_replay(self):
        rng = np.random.default_rng()
        self.replay = World(self.run.champion[None, :], rng, self.run.config)
        self.selected = 0

    def toggle_replay(self):
        if self.replay:
            self.replay, self.selected = None, None
        elif self.run.history:
            self.start_replay()

    def advance(self):
        for _ in range(SPEEDS[self.speed]):
            if self.replay:
                self.replay.step()
                if self.replay.done:
                    self.start_replay()
            else:
                generation = self.run.generation
                self.run.step()
                if self.run.generation != generation:
                    self.selected = None

    def skip_generation(self):
        if not self.replay:
            self.run.run_generation()
            self.selected = None

    def focus(self):
        """The creature whose brain is shown: the clicked one, else the best one alive."""
        world = self.world
        if self.selected is not None and world.alive[self.selected]:
            return self.selected
        if not world.alive.any():
            return None
        score = np.where(world.alive, world.fitness(), -np.inf)
        return int(np.argmax(score))

    def click(self, pos):
        if not self.dish.collidepoint(pos):
            return
        world = self.world
        x = (pos[0] - self.dish.centerx) / self.scale
        y = (pos[1] - self.dish.centery) / self.scale
        dist = np.where(world.alive, np.hypot(world.pos[:, 0] - x, world.pos[:, 1] - y), np.inf)
        index = int(np.argmin(dist))
        self.selected = index if dist[index] < 20 else None

    def handle(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.click(event.pos)
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.running = False
            elif event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                self.speed = event.key - pg.K_1
            elif event.key == pg.K_g:
                self.skip_generation()
            elif event.key == pg.K_b:
                self.toggle_replay()
            elif event.key == pg.K_r:
                self.run, self.replay, self.selected = Run(), None, None
            elif event.key == pg.K_PERIOD and self.paused:
                self.paused = False
                self.advance()
                self.paused = True

    # Drawing

    def draw_dish(self):
        world, c = self.world, self.run.config
        center = self.dish.center
        pg.draw.circle(self.screen, DISH, center, c.radius * self.scale)
        pg.draw.circle(self.screen, LINE, center, c.radius * self.scale, 2)
        pellet = max(2, c.pellet * self.scale * .8)
        for x, y in world.food:
            pg.draw.circle(self.screen, FOOD, self.to_screen((x, y)), pellet)
        for x, y in world.poison:
            px, py = self.to_screen((x, y))
            pg.draw.circle(self.screen, POISON, (px, py), pellet + 1, 2)

        hues = [self.run.champion_hue] if self.replay else self.run.hues
        focus = self.focus()
        size = c.body * self.scale * 1.5
        for i in np.flatnonzero(world.alive):
            x, y = self.to_screen(world.pos[i])
            a = world.heading[i]
            nose = (x + math.cos(a) * size * 1.4, y + math.sin(a) * size * 1.4)
            left = (x + math.cos(a + 2.5) * size, y + math.sin(a + 2.5) * size)
            right = (x + math.cos(a - 2.5) * size, y + math.sin(a - 2.5) * size)
            color = pg.Color(0)
            color.hsva = (float(hues[i]), 55, 95, 100)
            pg.draw.polygon(self.screen, color, (nose, left, right))
            if i == focus:
                pg.draw.circle(self.screen, TEXT, (x, y), size * 2.2, 1)
                for offset in (-.6, 0, .6):
                    wall = wall_distance(world.pos[i:i + 1], np.array([a + offset]), c.radius)[0]
                    reach = min(c.sight, wall) * self.scale
                    end = (x + math.cos(a + offset) * reach, y + math.sin(a + offset) * reach)
                    pg.draw.line(self.screen, LINE, (x, y), end, 1)

        mode = "CHAMPION REPLAY" if self.replay else f"GENERATION {self.run.generation}"
        self.text(mode, (24, 18), MINT, self.mono)
        bar = pg.Rect(24, 40, 160, 4)
        pg.draw.rect(self.screen, LINE, bar)
        pg.draw.rect(self.screen, MINT, (bar.x, bar.y, bar.w * world.time / c.steps, bar.h))

    def draw_stats(self, x, y):
        world, history = self.world, self.run.history
        best = max((h["best"] for h in history), default=0)
        rows = (
            ("alive", f"{int(world.alive.sum())}/{len(world.alive)}"),
            ("food eaten", int(world.eaten.sum())),
            ("poison eaten", int(world.poisoned.sum())),
            ("best ever", f"{best:.1f}"),
            ("speed", f"{SPEEDS[self.speed]}x" + ("  paused" if self.paused else "")),
        )
        for i, (label, value) in enumerate(rows):
            col, row = i % 3, i // 3
            self.text(label, (x + col * 140, y + row * 44), MUTED, self.small)
            self.text(value, (x + col * 140, y + row * 44 + 14), TEXT, self.mono)

    def draw_graph(self, rect):
        pg.draw.rect(self.screen, BG, rect, border_radius=6)
        history = self.run.history[-80:]
        self.text("fitness per generation", (rect.x + 10, rect.y + 6), MUTED, self.small)
        if len(history) < 2:
            self.text("evolving the first generation...", (rect.x + 10, rect.centery - 8), MUTED, self.small)
            return
        top = max(h["best"] for h in history) or 1
        inner = rect.inflate(-24, -36).move(0, 8)

        def line(key, color, width):
            points = [
                (inner.x + inner.w * i / (len(history) - 1),
                 inner.bottom - inner.h * max(0, h[key]) / top)
                for i, h in enumerate(history)
            ]
            pg.draw.lines(self.screen, color, False, points, width)

        line("mean", MUTED, 1)
        line("best", MINT, 2)
        self.text(f"{top:.0f}", (rect.right - 34, rect.y + 6), MINT, self.small)
        self.text("best", (rect.x + 200, rect.y + 6), MINT, self.small)
        self.text("mean", (rect.x + 245, rect.y + 6), MUTED, self.small)

    def draw_brain(self, rect):
        pg.draw.rect(self.screen, BG, rect, border_radius=6)
        index = self.focus()
        label = "champion brain" if self.replay else "brain of the ringed creature"
        self.text(label, (rect.x + 10, rect.y + 6), MUTED, self.small)
        if index is None:
            return
        world = self.world
        w1, _, w2, _ = (part[0] for part in unpack(world.genomes[index]))
        acts = (world.inputs[index], world.hidden[index], world.outputs[index])
        columns = (rect.x + 120, rect.centerx + 30, rect.right - 120)
        top, height = rect.y + 30, rect.h - 56

        def spots(count, column):
            gap = height / count
            return [(column, top + gap * (i + .5)) for i in range(count)]

        nodes = [spots(len(a), col) for a, col in zip(acts, columns)]
        for weights, starts, ends in ((w1, nodes[0], nodes[1]), (w2, nodes[1], nodes[2])):
            for i, start in enumerate(starts):
                for j, end in enumerate(ends):
                    w = weights[i, j]
                    if abs(w) > .25:
                        pg.draw.line(self.screen, signal(w / 2), start, end, 1 + int(abs(w) > 1))
        for layer, (values, points) in enumerate(zip(acts, nodes)):
            for value, point in zip(values, points):
                pg.draw.circle(self.screen, signal(value), point, 7)
                pg.draw.circle(self.screen, LINE, point, 7, 1)
        for name, (px, py) in zip(INPUT_LABELS, nodes[0]):
            surf = self.small.render(name, True, MUTED)
            self.screen.blit(surf, (px - 14 - surf.get_width(), py - 7))
        for name, value, (px, py) in zip(OUTPUT_LABELS, acts[2], nodes[2]):
            self.text(f"{name} {value:+.2f}", (px + 14, py - 7), TEXT, self.small)
        stats = f"food {world.eaten[index]}  poison {world.poisoned[index]}  energy {world.energy[index]:.0f}"
        self.text(stats, (rect.x + 10, rect.bottom - 18), MUTED, self.small)

    def draw(self):
        self.screen.fill(BG)
        self.draw_dish()
        pg.draw.rect(self.screen, PANEL, self.panel)
        x = self.panel.x + 26
        self.text("PETRI MINDS", (x, 16), font=self.title)
        self.text("tiny neural brains, evolved by selection", (x, 60), MUTED)
        self.draw_stats(x, 100)
        width = self.panel.w - 52
        self.draw_graph(pg.Rect(x, 196, width, 130))
        brain = pg.Rect(x, 340, width, self.panel.h - 340 - 74)
        self.draw_brain(brain)
        keys = ("Space pause   1-4 speed   G skip generation   . step",
                "Click a creature to follow it   B champion replay   R restart")
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
            print(f"gen {h['generation']:>3}  best {h['best']:6.1f}  mean {h['mean']:5.1f}  "
                  f"food {h['food']:>5}  poison {h['poison']:>4}  survivors {h['survivors']:>2}")


def main():
    parser = argparse.ArgumentParser(description="Neural-net creatures evolving in a petri dish.")
    parser.add_argument("--seed", type=int, help="Repeat the same run.")
    parser.add_argument("--train", type=int, default=0, metavar="N",
                        help="Evolve N generations before opening the window.")
    parser.add_argument("--headless", action="store_true", help="Train without a window, then exit.")
    parser.add_argument("--screenshot", type=Path, help="Save one window preview and exit.")
    args = parser.parse_args()

    run = Run(args.seed)
    train(run, args.train)
    if args.headless:
        return
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        app = App(run=run)
        for _ in range(240):
            app.run.world.step()
        app.draw()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pg.image.save(app.screen, args.screenshot)
        pg.quit()
        print("Saved", args.screenshot)
        return
    App(run=run).loop()


if __name__ == "__main__":
    main()
