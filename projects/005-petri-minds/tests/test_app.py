import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import unittest

import pygame as pg

from petri_minds.__main__ import App
from petri_minds.evolution import Run
from petri_minds.world import Config


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = App(run=Run(2, Config(population=20, steps=120)))
        self.app.draw()

    def tearDown(self):
        pg.quit()

    def key(self, key):
        self.app.handle(pg.event.Event(pg.KEYDOWN, key=key, mod=0))

    def test_click_selects_nearest_creature(self):
        world = self.app.world
        target = int(world.alive.nonzero()[0][3])
        pos = tuple(int(v) for v in self.app.to_screen(world.pos[target]))
        self.app.click(pos)
        self.assertEqual(self.app.selected, target)
        self.assertEqual(self.app.focus(), target)
        self.app.click((self.app.dish.x + 2, self.app.dish.y + 2))
        self.assertIsNone(self.app.selected)

    def test_speed_keys_and_skip(self):
        self.key(pg.K_3)
        self.app.advance()
        self.assertEqual(self.app.world.time, 12)
        self.key(pg.K_g)
        self.assertEqual(self.app.run.generation, 2)
        self.app.draw()

    def test_champion_replay_toggles(self):
        self.key(pg.K_b)
        self.assertIsNone(self.app.replay)  # nothing evolved yet
        self.app.skip_generation()
        self.key(pg.K_b)
        self.assertEqual(len(self.app.replay.alive), 1)
        for _ in range(130):
            self.app.advance()
        self.app.draw()
        self.key(pg.K_b)
        self.assertIsNone(self.app.replay)

    def test_space_pauses(self):
        self.key(pg.K_SPACE)
        self.app.update()
        self.assertEqual(self.app.world.time, 0)
        self.key(pg.K_PERIOD)
        self.assertEqual(self.app.world.time, 1)


if __name__ == "__main__":
    unittest.main()
