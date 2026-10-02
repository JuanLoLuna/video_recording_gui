import unittest

import backend.compression_policy as policy


class SuggestedCompressionTests(unittest.TestCase):
    def test_one_camera_uses_mjpeg_up_to_30_fps_and_raw_above(self):
        for fps in (10.0, 25.0, 29.99, 30.0):
            self.assertTrue(policy.suggested_compression(1, fps), fps)
        for fps in (30.01, 60.0, 100.0):
            self.assertFalse(policy.suggested_compression(1, fps), fps)

    def test_several_cameras_default_to_uncompressed_at_every_rate(self):
        for fps in (15.0, 30.0, 59.9, 60.0, 100.0):
            self.assertFalse(policy.suggested_compression(2, fps), fps)

    def test_zero_cameras_before_detect_behaves_like_one(self):
        self.assertTrue(policy.suggested_compression(0, 30.0))

    def test_the_multi_camera_limit_can_be_raised_after_validation(self):
        original = policy.MULTI_CAMERA_MJPEG_MAX_FPS
        self.addCleanup(lambda: setattr(policy, "MULTI_CAMERA_MJPEG_MAX_FPS", original))
        policy.MULTI_CAMERA_MJPEG_MAX_FPS = 30.0
        self.assertTrue(policy.suggested_compression(2, 30.0))
        self.assertFalse(policy.suggested_compression(2, 60.0))
        self.assertTrue(policy.suggested_compression(1, 30.0))  # unchanged

    def test_the_hint_matches_the_policy(self):
        self.assertIn("30 fps", policy.describe_policy(1))
        self.assertIn("several cameras", policy.describe_policy(2))


if __name__ == "__main__":
    unittest.main()
