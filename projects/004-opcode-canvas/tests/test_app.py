import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
import unittest

import pygame as pg

from opcode_canvas.__main__ import App
from opcode_canvas.presets import PRESETS


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = App((1040, 780))
        self.app.paused = True
        self.app.draw()

    def tearDown(self):
        self.app.pool.shutdown(wait=True, cancel_futures=True)
        pg.quit()

    def wait_for_frame(self):
        deadline = time.monotonic() + 5
        while (self.app.pending or self.app.want_frame) and time.monotonic() < deadline:
            self.app.update()
            time.sleep(.01)
        self.assertFalse(self.app.pending)

    def test_typing_compile_pixel_pick_and_step(self):
        editor = self.app.editor
        editor.focus = True
        pg.event.post(pg.event.Event(pg.TEXTINPUT, text="; edited\n"))
        self.app.events()
        self.assertTrue(editor.text.startswith("; edited\n"))
        self.app.action("run")
        self.wait_for_frame()
        self.assertEqual(self.app.compiled_source, editor.text)
        self.app.draw()
        pg.event.post(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1,
                                     pos=self.app.canvas_rect.center))
        self.app.events()
        self.assertTrue(self.app.paused)
        self.assertEqual(self.app.pixel, (64, 40))
        self.app.action("step")
        self.assertEqual(self.app.debug.steps, 1)

    def test_failed_compile_preserves_last_program_and_recovers(self):
        previous = self.app.program
        self.app.editor.text = "not_an_instruction eax"
        self.app.compile()
        self.assertIs(self.app.program, previous)
        self.assertTrue(self.app.error)
        self.app.action("preset1")
        self.wait_for_frame()
        self.assertEqual(self.app.compiled_source, PRESETS[1].source)
        self.assertFalse(self.app.error)

    def test_frame_reset_speed_and_runtime_error_recovery(self):
        self.app.action("speed")
        self.app.action("frame")
        self.wait_for_frame()
        self.assertEqual(self.app.frame.tick, 2)
        self.app.action("reset")
        self.wait_for_frame()
        self.assertEqual(self.app.frame.tick, 0)
        self.app.editor.text = "again: jmp again"
        self.app.compile()
        self.wait_for_frame()
        self.assertTrue(self.app.error)
        self.app.action("preset0")
        self.wait_for_frame()
        self.assertFalse(self.app.error)

    def test_all_buttons_fit_minimum_window(self):
        bounds = self.app.screen.get_rect()
        for rect in self.app.buttons.values():
            self.assertTrue(bounds.contains(rect))
        self.assertTrue(bounds.contains(self.app.canvas_rect))


if __name__ == "__main__":
    unittest.main()
