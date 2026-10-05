"""backend/calibration_store.py: where calibrations live and whether they still apply."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

REAL_PYSPIN = __import__("fake_spinnaker").install_pyspin_stub()

from backend import calibration_store as cs  # noqa: E402
from backend.camera_control import CameraController  # noqa: E402
from backend.spinnaker_system import SharedSystemHolder  # noqa: E402
from fake_spinnaker import FakeCamera, FakeSystem  # noqa: E402

NOW = datetime(2026, 10, 6, 10, 0, 0)
FP = {"Width": 720, "Height": 540, "OffsetX": 0, "OffsetY": 0, "BinningHorizontal": 1,
      "BinningVertical": 1, "DecimationHorizontal": 1, "DecimationVertical": 1,
      "ReverseX": False, "ReverseY": False, "PixelFormat": "Mono8"}


def intrinsics(serial="23227865", **kw):
    data = dict(serial=serial, model="Firefly", K=[[800, 0, 360], [0, 800, 270], [0, 0, 1]],
                D=[-0.1, 0.02, 0, 0, 0], image_size=[720, 540], fingerprint=dict(FP),
                board={"squares_x": 7}, rms_px=0.3, n_views=30)
    data.update(kw)
    return cs.IntrinsicsRecord(**data)


def setup_for(store_ids, passed=True, verified=True, created=NOW):
    return cs.SetupRecord(
        cameras=[{"serial": s, "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 1], "rms_px": 0.4,
                  "intrinsics_id": i} for s, i in store_ids.items()],
        board={"squares_x": 7}, baseline_mm=600.0, triangulation_rms_mm=0.8, passed=passed,
        verify={"passed": verified, "scale_error_pct": 0.2} if verified is not None else None,
        created_at=created.isoformat(timespec="seconds"))


class DefaultDirTest(unittest.TestCase):
    def test_env_override_wins(self):
        self.assertEqual(cs.default_calibration_dir({cs.CALIBRATION_DIR_ENV: "/data/cal"}, "win32"),
                         Path("/data/cal"))

    def test_windows_uses_localappdata(self):
        self.assertEqual(cs.default_calibration_dir({"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}, "win32"),
                         Path(r"C:\Users\x\AppData\Local") / "SleeveVideoGUI" / "calibration")

    def test_elsewhere_uses_local_share(self):
        self.assertEqual(cs.default_calibration_dir({}, "linux").parts[-3:],
                         ("share", "SleeveVideoGUI", "calibration"))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = cs.CalibrationStore(self.tmp.name)

    def test_nothing_saved(self):
        self.assertIsNone(self.store.current_intrinsics("1"))
        self.assertIsNone(self.store.current_setup())
        self.assertEqual(self.store.intrinsics_history("1"), [])

    def test_save_and_load_round_trip(self):
        path = self.store.save_intrinsics(intrinsics(), now=NOW)
        self.assertEqual(path.name, "20261006_100000.json")
        loaded = self.store.current_intrinsics("23227865")
        self.assertEqual(loaded.K, intrinsics().K)
        self.assertEqual(loaded.id, "20261006_100000")
        self.assertEqual(loaded.created_at, "2026-10-06T10:00:00")
        self.assertEqual(json.loads(path.read_text())["schema_version"], cs.SCHEMA_VERSION)

    def test_history_is_never_overwritten_and_current_follows_the_latest(self):
        self.store.save_intrinsics(intrinsics(rms_px=0.3), now=NOW)
        self.store.save_intrinsics(intrinsics(rms_px=0.2), now=NOW)  # same second
        self.store.save_intrinsics(intrinsics(rms_px=0.25), now=NOW + timedelta(days=1), make_current=False)
        history = self.store.intrinsics_history("23227865")
        self.assertEqual([r.rms_px for r in history], [0.3, 0.2, 0.25])
        self.assertEqual(self.store.current_intrinsics("23227865").rms_px, 0.2)

    def test_cameras_are_kept_apart(self):
        self.store.save_intrinsics(intrinsics("A"), now=NOW)
        self.assertIsNone(self.store.current_intrinsics("B"))

    def test_a_corrupt_current_pointer_reads_as_nothing(self):
        self.store.save_intrinsics(intrinsics(), now=NOW)
        (self.store.intrinsics_dir("23227865") / "current.json").write_text("{not json")
        self.assertIsNone(self.store.current_intrinsics("23227865"))

    def test_setup_round_trip(self):
        self.store.save_setup(setup_for({"A": "x", "B": "y"}), now=NOW)
        loaded = self.store.current_setup()
        self.assertEqual(loaded.serials, ["A", "B"])
        self.assertTrue(loaded.verified)
        self.assertEqual(loaded.camera("B")["intrinsics_id"], "y")


class FingerprintTest(unittest.TestCase):
    def test_differences_are_listed_and_unreadable_nodes_ignored(self):
        live = dict(FP, Width=1280, ReverseX=None)
        self.assertEqual(cs.fingerprint_differences(FP, live), ["Width 720 -> 1280"])
        self.assertEqual(cs.fingerprint_differences(FP, dict(FP)), [])


class AssessCameraTest(unittest.TestCase):
    def test_states(self):
        rec = intrinsics(created_at=NOW.isoformat())
        self.assertEqual(cs.assess_camera("s", "Firefly", None, FP, NOW).state, cs.MISSING)
        self.assertEqual(cs.assess_camera("s", "Firefly", rec, FP, NOW).state, cs.READY)
        mism = cs.assess_camera("s", "Firefly", rec, dict(FP, BinningHorizontal=2), NOW)
        self.assertEqual(mism.state, cs.MISMATCH)
        self.assertIn("BinningHorizontal 1 -> 2", mism.reason)
        self.assertEqual(cs.assess_camera("s", "F", intrinsics(loose=True, created_at=NOW.isoformat()),
                                          FP, NOW).state, cs.LOOSE)
        self.assertEqual(cs.assess_camera("s", "F", rec, FP, NOW, live_rms_px=2.5).state, cs.SUSPECT)

    def test_unknown_settings_and_age_are_notes_not_blocks(self):
        old = intrinsics(created_at=(NOW - timedelta(days=400)).isoformat())
        status = cs.assess_camera("s", "F", old, None, NOW)
        self.assertTrue(status.ready)
        self.assertTrue(any("Preview" in n for n in status.notes))
        self.assertTrue(any("400 days" in n for n in status.notes))


class AssessSessionTest(unittest.TestCase):
    CAMS = [("A", "Firefly #A"), ("B", "Blackfly #B")]

    def statuses(self, **override):
        out = {s: cs.CameraStatus(s, lbl, cs.READY, intrinsics_id=f"id{s}") for s, lbl in self.CAMS}
        out.update(override)
        return out

    def assess(self, setup, **kw):
        return cs.assess_session(self.CAMS, kw.pop("statuses", self.statuses()), setup, **kw)

    def test_ready(self):
        st = self.assess(setup_for({"A": "idA", "B": "idB"}))
        self.assertTrue(st.ready)
        self.assertEqual(st.headline, "3D pose: ready")

    def test_one_camera(self):
        st = cs.assess_session([("A", "Firefly #A")], self.statuses(), None)
        self.assertFalse(st.ready)
        self.assertIn("needs two cameras", st.headline)

    def test_each_setup_problem(self):
        good = {"A": "idA", "B": "idB"}
        cases = [
            (None, cs.NO_SETUP),
            (setup_for({"A": "idA", "C": "idC"}), cs.SETUP_OTHER_CAMERAS),
            (setup_for(good, passed=False), cs.SETUP_FAILED),
            (setup_for({"A": "old", "B": "idB"}), cs.SETUP_STALE),
            (setup_for(good, verified=False), cs.SETUP_UNVERIFIED),
            (setup_for(good, verified=None), cs.SETUP_UNVERIFIED),
        ]
        for setup, state in cases:
            with self.subTest(state=state):
                st = self.assess(setup)
                self.assertFalse(st.ready)
                self.assertEqual(st.setup_state, state)

    def test_camera_reasons_come_first_with_labels(self):
        statuses = self.statuses(B=cs.CameraStatus("B", "Blackfly #B", cs.MISSING, "camera not calibrated"))
        st = self.assess(None, statuses=statuses)
        self.assertEqual(st.reasons[0], "[Blackfly #B] camera not calibrated")
        self.assertIn(cs.NO_SETUP, st.reasons)

    def test_old_setup_needs_the_live_check(self):
        old = setup_for({"A": "idA", "B": "idB"}, created=NOW - timedelta(days=1))
        # Cameras detected after the setup, no live board check yet -> stale.
        self.assertEqual(self.assess(old, setup_valid_since=NOW).setup_state, cs.SETUP_OLD_SESSION)
        # Live check only for one camera -> still stale.
        self.assertEqual(self.assess(old, setup_valid_since=NOW, moved={"A": False}).setup_state,
                         cs.SETUP_OLD_SESSION)
        # Live check says nothing moved -> yesterday's setup is still good.
        self.assertTrue(self.assess(old, setup_valid_since=NOW, moved={"A": False, "B": False}).ready)

    def test_moved_camera(self):
        st = self.assess(setup_for({"A": "idA", "B": "idB"}), moved={"A": False, "B": True})
        self.assertEqual(st.setup_state, cs.SETUP_MOVED)
        self.assertIn("Blackfly #B", st.headline)

    def test_snapshot_is_json_serialisable(self):
        st = self.assess(setup_for({"A": "idA", "B": "idB"}))
        self.assertEqual(json.loads(json.dumps(st.to_dict()))["headline"], "3D pose: ready")


@unittest.skipIf(REAL_PYSPIN, "real PySpin present")
class ControllerFingerprintTest(unittest.TestCase):
    def test_reads_every_node_while_acquiring(self):
        camera = FakeCamera("111", "Cam", 64, 48)
        controller = CameraController(serial="111", system_holder=SharedSystemHolder(lambda: FakeSystem([camera])))
        controller._configure_camera_nodes = lambda: None
        self.addCleanup(controller.stop)
        self.assertIsNone(controller.get_sensor_fingerprint())  # not started
        ok, message = controller.start()
        self.assertTrue(ok, message)
        fp = controller.get_sensor_fingerprint()
        self.assertEqual(set(fp), set(cs.FINGERPRINT_NODES))
        self.assertEqual((fp["Width"], fp["Height"]), (64, 48))
        self.assertEqual(fp["PixelFormat"], "Mono8")
        self.assertIs(fp["ReverseX"], False)


if __name__ == "__main__":
    unittest.main()
