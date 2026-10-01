import unittest

from backend.preview_scaling import fit_size


class FitSizeTests(unittest.TestCase):
    def test_limited_by_width(self):
        self.assertEqual(fit_size(1280, 1024, 640, 800), (640, 512))

    def test_limited_by_height(self):
        self.assertEqual(fit_size(1280, 1024, 2000, 512), (640, 512))

    def test_never_upscales(self):
        self.assertEqual(fit_size(720, 540, 4000, 4000), (720, 540))

    def test_keeps_the_aspect_ratio_for_the_other_sensor(self):
        w, h = fit_size(720, 540, 360, 360)
        self.assertEqual((w, h), (360, 270))

    def test_never_returns_a_zero_dimension(self):
        self.assertEqual(fit_size(1280, 1024, 1, 1), (1, 1))
        self.assertEqual(fit_size(10000, 10, 100, 100), (100, 1))

    def test_degenerate_target_gives_1x1(self):
        self.assertEqual(fit_size(100, 100, 0, 500), (1, 1))

    def test_rejects_a_bad_source(self):
        with self.assertRaises(ValueError):
            fit_size(0, 100, 10, 10)


if __name__ == "__main__":
    unittest.main()
