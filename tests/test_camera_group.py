import unittest

from backend.camera_group import CameraGroup, CameraSlot


class FakeController:
    def __init__(self, name, log, *, start=(True, "Preview started."), stop=(True, ""),
                 start_recording=(True, "Recording requested."), raises=()):
        self.name = name
        self.log = log
        self._start = start
        self._stop = stop
        self._start_recording = start_recording
        self.raises = set(raises)
        self.acquiring = False
        self.recording_active = False

    def _maybe_raise(self, what):
        if what in self.raises:
            raise RuntimeError(f"{self.name} {what} blew up")

    def start(self):
        self.log.append(f"{self.name}:start")
        self._maybe_raise("start")
        self.acquiring = self._start[0]
        return self._start

    def stop(self):
        self.log.append(f"{self.name}:stop")
        self._maybe_raise("stop")
        self.acquiring = False
        return self._stop

    def start_recording(self, session_paths, fps=30.0):
        self.log.append(f"{self.name}:start_recording:{session_paths}:{fps}")
        self._maybe_raise("start_recording")
        self.recording_active = self._start_recording[0]
        return self._start_recording

    def stop_recording(self):
        self.log.append(f"{self.name}:stop_recording")
        self._maybe_raise("stop_recording")
        self.recording_active = False

    def notify_sync_pulse_window(self, width_s, label):
        self.log.append(f"{self.name}:sync:{width_s}:{label}")
        self._maybe_raise("notify_sync_pulse_window")

    def notify_label_event(self, label, adl_id, adl_label):
        self.log.append(f"{self.name}:label:{label}:{adl_id}:{adl_label}")
        self._maybe_raise("notify_label_event")


def make_group(log, **overrides):
    """Two cameras: 'a' (primary) and 'b'. overrides = {"a": {...}, "b": {...}}."""
    a = FakeController("a", log, **overrides.get("a", {}))
    b = FakeController("b", log, **overrides.get("b", {}))
    return CameraGroup(
        [
            CameraSlot(a, serial="111", model="Firefly", tag=None, is_primary=True),
            CameraSlot(b, serial="222", model="Blackfly", tag="cam222"),
        ]
    ), a, b


class ConstructionTests(unittest.TestCase):
    def test_needs_at_least_one_camera(self):
        with self.assertRaises(ValueError):
            CameraGroup([])

    def test_rejects_duplicate_serials(self):
        log = []
        with self.assertRaises(ValueError):
            CameraGroup([
                CameraSlot(FakeController("a", log), serial="1"),
                CameraSlot(FakeController("b", log), serial="1"),
            ])

    def test_first_slot_is_primary_when_none_is_marked(self):
        log = []
        group = CameraGroup([
            CameraSlot(FakeController("a", log), serial="1"),
            CameraSlot(FakeController("b", log), serial="2"),
        ])
        self.assertEqual(group.primary.serial, "1")

    def test_slot_lookup_and_label(self):
        group, _, _ = make_group([])
        self.assertEqual(group.slot_for_serial("222").label, "Blackfly #222")
        self.assertIsNone(group.slot_for_serial("nope"))


class StartAllTests(unittest.TestCase):
    def test_starts_every_camera_in_order(self):
        log = []
        group, a, b = make_group(log)
        result = group.start_all()
        self.assertTrue(result.ok)
        self.assertEqual(log, ["a:start", "b:start"])
        self.assertTrue(group.all_acquiring)

    def test_a_failed_second_camera_stops_the_first_again(self):
        log = []
        group, a, b = make_group(log, b={"start": (False, "No camera detected.")})
        result = group.start_all()
        self.assertFalse(result.ok)
        self.assertIn("Blackfly #222", result.message)
        self.assertIn("No camera detected.", result.message)
        self.assertEqual(log, ["a:start", "b:start", "a:stop"])
        self.assertFalse(group.any_acquiring)

    def test_an_exception_while_starting_is_a_failure_not_a_crash(self):
        log = []
        group, a, b = make_group(log, b={"raises": {"start"}})
        result = group.start_all()
        self.assertFalse(result.ok)
        self.assertIn("blew up", result.message)
        self.assertIn("a:stop", log)

    def test_a_failed_first_camera_never_touches_the_second(self):
        log = []
        group, a, b = make_group(log, a={"start": (False, "boom")})
        group.start_all()
        self.assertEqual(log, ["a:start"])


class StopAllTests(unittest.TestCase):
    def test_stop_recording_on_every_camera_happens_before_any_stop(self):
        log = []
        group, a, b = make_group(log)
        group.start_all()
        log.clear()
        group.stop_all()
        self.assertEqual(
            log, ["a:stop_recording", "b:stop_recording", "a:stop", "b:stop"]
        )

    def test_all_stopped_is_ok(self):
        group, _, _ = make_group([])
        group.start_all()
        self.assertTrue(group.stop_all().ok)

    def test_a_deferred_teardown_is_reported_so_the_system_is_not_released(self):
        log = []
        group, a, b = make_group(log, b={"stop": (False, "acquisition thread still alive")})
        group.start_all()
        result = group.stop_all()
        self.assertFalse(result.ok)
        self.assertIn("#222", result.message)
        self.assertIn("still alive", result.message)

    def test_one_camera_raising_in_stop_does_not_skip_the_other(self):
        log = []
        group, a, b = make_group(log, a={"raises": {"stop"}})
        group.start_all()
        log.clear()
        result = group.stop_all()
        self.assertFalse(result.ok)
        self.assertIn("b:stop", log)

    def test_one_camera_raising_in_stop_recording_does_not_skip_the_other(self):
        log = []
        group, a, b = make_group(log, a={"raises": {"stop_recording"}})
        group.start_all()
        log.clear()
        group.stop_recording_all()
        self.assertEqual(log, ["a:stop_recording", "b:stop_recording"])


class RecordingTests(unittest.TestCase):
    def test_each_camera_gets_its_own_paths_and_fps(self):
        log = []
        group, a, b = make_group(log)
        group.start_all()
        log.clear()
        result = group.start_recording_all(
            paths_for=lambda slot: f"paths-{slot.serial}",
            fps_of=lambda slot: 30.0 if slot.serial == "111" else 29.97,
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            log, ["a:start_recording:paths-111:30.0", "b:start_recording:paths-222:29.97"]
        )
        self.assertTrue(group.any_recording)

    def test_a_refusal_rolls_back_the_cameras_already_recording(self):
        log = []
        group, a, b = make_group(log, b={"start_recording": (False, "Cannot open metadata CSV")})
        group.start_all()
        log.clear()
        result = group.start_recording_all(lambda s: "p", lambda s: 30.0)
        self.assertFalse(result.ok)
        self.assertIn("Cannot open metadata CSV", result.message)
        self.assertEqual(log[-1], "a:stop_recording")
        self.assertFalse(group.any_recording)

    def test_an_exception_starting_the_recording_is_a_clean_failure(self):
        log = []
        group, a, b = make_group(log, a={"raises": {"start_recording"}})
        group.start_all()
        result = group.start_recording_all(lambda s: "p", lambda s: 30.0)
        self.assertFalse(result.ok)


class FanOutTests(unittest.TestCase):
    def test_sync_pulse_reaches_every_camera(self):
        log = []
        group, _, _ = make_group(log)
        failures = group.notify_sync_pulse_window(0.1, "record_start")
        self.assertEqual(failures, 0)
        self.assertEqual(log, ["a:sync:0.1:record_start", "b:sync:0.1:record_start"])

    def test_one_camera_failing_a_notification_does_not_block_the_other(self):
        log = []
        group, _, _ = make_group(log, a={"raises": {"notify_sync_pulse_window"}})
        failures = group.notify_sync_pulse_window(0.1, "x")
        self.assertEqual(failures, 1)
        self.assertIn("b:sync:0.1:x", log)

    def test_label_events_reach_every_camera(self):
        log = []
        group, _, _ = make_group(log)
        self.assertEqual(group.notify_label_event("label_start", 3, "Writing"), 0)
        self.assertEqual(
            log, ["a:label:label_start:3:Writing", "b:label:label_start:3:Writing"]
        )

    def test_broadcast_returns_values_and_exceptions_per_camera(self):
        log = []
        group, a, b = make_group(log)
        a.set_thing = lambda v: v * 2

        def bad(v):
            raise ValueError("nope")

        b.set_thing = bad
        results = group.broadcast("set_thing", 21)
        self.assertEqual([r.ok for r in results], [True, False])
        self.assertEqual(results[0].value, 42)
        self.assertIsInstance(results[1].value, ValueError)

    def test_broadcasting_a_missing_method_is_a_per_camera_failure(self):
        group, _, _ = make_group([])
        results = group.broadcast("does_not_exist")
        self.assertTrue(all(not r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
