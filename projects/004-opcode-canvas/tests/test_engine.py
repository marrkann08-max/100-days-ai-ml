import unittest

from opcode_canvas.engine import PixelProgram, AssemblyError, ExecutionError, CODE
from opcode_canvas.presets import PRESETS


class PixelTests(unittest.TestCase):
    def test_xor_frame_matches_integer_reference_at_every_pixel(self):
        program = PixelProgram(PRESETS[0].source, 37, 23)
        for tick in (0, 27, 255, 0xffffffff):
            frame = program.render(tick)
            expected = bytes((((x ^ y) + tick) << 2) & 255
                             for y in range(23) for x in range(37))
            self.assertEqual(frame.pixels, expected)

    def test_instruction_edit_changes_real_framebuffer(self):
        source = PRESETS[0].source
        xor = PixelProgram(source).render(0)
        and_ = PixelProgram(source.replace("xor eax", "and eax")).render(0)
        self.assertNotEqual(xor.pixels, and_.pixels)
        self.assertEqual(and_.pixels[7 * 128 + 11], ((11 & 7) << 2) & 255)

    def test_other_presets_match_reference(self):
        for preset, calculate in (
            (PRESETS[1], lambda x, y, t: (((x + t)**2 + (y - t)**2) & 0xffffffff) >> 4),
            (PRESETS[2], lambda x, y, t: (((x ^ (y + t)) * (x >> 3)) ^ ((y + t) >> 2))),
        ):
            frame = PixelProgram(preset.source, 32, 18).render(11)
            expected = bytes(calculate(x, y, 11) & 255 for y in range(18) for x in range(32))
            self.assertEqual(frame.pixels, expected)

    def test_debugger_steps_and_returns_same_pixel_as_renderer(self):
        for preset in PRESETS:
            program = PixelProgram(preset.source)
            debugger = program.debugger(61, 34, 19)
            self.assertEqual(debugger.snapshot()["EIP"], CODE)
            self.assertTrue(debugger.instruction())
            while not debugger.done:
                debugger.step()
            self.assertEqual(debugger.snapshot()["EAX"] & 255,
                             program.render(19).pixels[34 * 128 + 61])
            self.assertEqual(debugger.step(), debugger.snapshot())

    def test_assembler_reports_errors_and_ignores_comments(self):
        for source in ("", "; only a comment", "this_is_not_an_instruction eax, 12"):
            with self.assertRaises(AssemblyError):
                PixelProgram(source)
        self.assertEqual(PixelProgram("mov eax, 73 ; colour\nret").render(0).pixels, bytes([73]) * 10240)

    def test_infinite_loop_is_bounded(self):
        with self.assertRaisesRegex(ExecutionError, "limit"):
            PixelProgram("again: jmp again", 1, 1).render(0)

    def test_invalid_memory_access_is_reported(self):
        with self.assertRaises(ExecutionError):
            PixelProgram("mov eax, dword ptr [0]\nret", 1, 1).render(0)

    def test_debugger_infinite_loop_is_bounded(self):
        debugger = PixelProgram("again: jmp again", 1, 1).debugger(0, 0, 0)
        for _ in range(512):
            debugger.step()
        with self.assertRaises(ExecutionError):
            debugger.step()


if __name__ == "__main__":
    unittest.main()
