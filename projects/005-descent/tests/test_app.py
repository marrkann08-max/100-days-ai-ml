import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import unittest

import pygame as pg

from descent.__main__ import App
from descent.evolution import Run
from descent.tower import Config


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = App(run=Run(2, Config(towers=4, levels=10, days=4, month=2)))
        self.app.draw()

    def tearDown(self):
        pg.quit()

    def key(self, key):
        self.app.handle(pg.event.Event(pg.KEYDOWN, key=key, mod=0))

    def test_click_selects_inmate_and_tower(self):
        pos = tuple(int(v) for v in self.app.seat(2, 5))
        self.app.click(pos)
        self.assertEqual(self.app.tower, 2)
        self.assertEqual(self.app.selected, (2, 5))
        self.assertEqual(self.app.focus(), (2, 5))
        self.app.draw()

    def test_speed_keys_and_skip(self):
        self.key(pg.K_3)
        self.app.advance()
        self.assertEqual(self.app.run.towers.day, 2)
        self.key(pg.K_g)
        self.assertEqual(self.app.run.generation, 2)
        self.app.draw()

    def test_solidarity_and_tower_keys(self):
        self.key(pg.K_s)
        self.assertEqual(self.app.run.solidarity, .5)
        self.key(pg.K_s)
        self.key(pg.K_s)
        self.assertEqual(self.app.run.solidarity, 0)
        self.key(pg.K_LEFT)
        self.assertEqual(self.app.tower, 3)

    def test_space_pauses_and_period_steps(self):
        self.key(pg.K_SPACE)
        self.app.update()
        self.assertEqual(self.app.run.towers.stop, 0)
        self.key(pg.K_PERIOD)
        self.assertEqual(self.app.run.towers.stop, 1)

    def test_draws_through_a_whole_generation(self):
        self.app.speed = 3
        for _ in range(3):
            self.app.update()
        self.assertGreater(self.app.run.generation, 1)


if __name__ == "__main__":
    unittest.main()
