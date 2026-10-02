"""scripts/multi_controller_smoke.py end to end against fake cameras.

The script is what runs on the rig, so it gets the same treatment as the app:
both codecs through the real controllers, the group, the verifier and the video
decode checks. Skipped when real PySpin is present (there, run the script).
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fake_spinnaker import FakeCamera, FakeSystem, install_pyspin_stub

REAL_PYSPIN = install_pyspin_stub()

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import backend.spinnaker_system as spinnaker_system  # noqa: E402
from backend.camera_registry import CAMERA_SERIALS_ENV  # noqa: E402
from backend.spinnaker_system import SharedSystemHolder  # noqa: E402


def load_smoke():
    spec = importlib.util.spec_from_file_location("multi_controller_smoke", SCRIPTS / "multi_controller_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: run the script on the rig")
class SmokeScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = spinnaker_system._default_holder
        self.addCleanup(lambda: setattr(spinnaker_system, "_default_holder", previous))
        saved = os.environ.get(CAMERA_SERIALS_ENV)
        os.environ.pop(CAMERA_SERIALS_ENV, None)
        self.addCleanup(lambda: os.environ.__setitem__(CAMERA_SERIALS_ENV, saved) if saved else None)
        self.cameras = [FakeCamera("26134271", "Blackfly S", 64, 48), FakeCamera("23227865", "Firefly", 32, 24)]
        spinnaker_system._default_holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))

    def run_script(self, *extra):
        module = load_smoke()
        argv = ["multi_controller_smoke.py", "--seconds", "2.5", "--segment-seconds", "1",
                "--output-dir", self.tmp.name, *extra]
        out = io.StringIO()
        old_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            sys.argv = old_argv
        return code, out.getvalue()

    def test_mjpeg_run_passes_and_proves_the_codec_and_the_pixels(self):
        code, text = self.run_script("--codec", "mjpg")
        self.assertEqual(code, 0, text)
        self.assertIn("RESULT: PASS", text)
        self.assertIn("MJPG", text)  # the codec was read back from the files, not assumed
        self.assertNotIn("BLACK", text)
        self.assertEqual(text.count("PASS"), 3, text)  # two cameras + the overall result

    def test_uncompressed_run_passes_and_reads_back_as_raw(self):
        code, text = self.run_script("--codec", "grey")
        self.assertEqual(code, 0, text)
        self.assertIn("raw/uncompressed", text)

    def test_asking_for_a_codec_the_files_do_not_have_fails(self):
        # Corrupt the expectation to prove the check can fail: claim MJPG but record GREY.
        module = load_smoke()
        module.EXPECTED_CODEC = {"grey": "MJPG", "mjpg": "MJPG"}
        out = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["x", "--seconds", "2.0", "--segment-seconds", "1", "--output-dir", self.tmp.name,
                    "--codec", "grey"]
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            sys.argv = old_argv
        self.assertEqual(code, 1)
        self.assertIn("expected MJPG", out.getvalue())

    def test_a_missing_configured_camera_refuses_to_run(self):
        code, text = self.run_script("--serials", "23227865", "26134271", "99999999")
        self.assertEqual(code, 2, text)
        self.assertIn("refusing to run", text)


if __name__ == "__main__":
    unittest.main()
