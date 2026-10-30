import unittest
from ki_forge.gestures import Energy, Hand


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.energy = Energy()
        self.open = Hand((800, 360), 100, True)
        self.closed = Hand((800, 360), 100, False)

    def charge(self, fps=30):
        for _ in range(fps*3):
            self.energy.update([self.open], 1/fps, 1280)

    def test_charge_is_time_based(self):
        self.charge(30)
        first = self.energy.charge
        self.energy.reset()
        self.charge(60)
        self.assertAlmostEqual(first, self.energy.charge)
        self.assertEqual(self.energy.state, 'READY')

    def test_fist_fires_once_after_debounce(self):
        self.charge()
        shots = [self.energy.update([self.closed], 1/30, 1280) for _ in range(60)]
        self.assertEqual(sum(shots), 1)
        self.assertGreater(self.energy.direction[0], 0)

    def test_tracking_loss_never_fires(self):
        self.charge()
        self.assertFalse(self.energy.update([], .1, 1280))
        self.assertIsNone(self.energy.center)

    def test_unready_fist_does_not_fire(self):
        for _ in range(20):
            self.assertFalse(self.energy.update([self.closed], .1, 1280))

    def test_two_hands_center_ball(self):
        self.energy.update([Hand((300, 300), 100, True),
                            Hand((700, 300), 100, True)], .1, 1280)
        self.assertEqual(self.energy.center, (500, 300))
        self.assertGreater(self.energy.charge, 0)

    def test_reset_clears_state(self):
        self.charge()
        self.energy.reset()
        self.assertEqual(self.energy.charge, 0)
        self.assertIsNone(self.energy.center)

    def test_separating_open_palms_does_not_fire(self):
        self.charge()
        hands = [Hand((100, 300), 80, True), Hand((1100, 300), 80, True)]
        for _ in range(10):
            self.assertFalse(self.energy.update(hands, .1, 1280))


if __name__ == '__main__':
    unittest.main()
