import unittest

from backend.spinnaker_system import SharedSystemHolder


class FakeSystem:
    def __init__(self, log, name, release_raises=False):
        self.log = log
        self.name = name
        self.release_raises = release_raises

    def ReleaseInstance(self):
        self.log.append(f"release:{self.name}")
        if self.release_raises:
            raise RuntimeError("cameras still referenced")


class Factory:
    def __init__(self, release_raises=False, fail_on=()):
        self.log = []
        self.made = 0
        self.release_raises = release_raises
        self.fail_on = set(fail_on)

    def __call__(self):
        self.made += 1
        if self.made in self.fail_on:
            self.log.append(f"get-failed:{self.made}")
            raise RuntimeError("no system")
        self.log.append(f"get:{self.made}")
        return FakeSystem(self.log, self.made, self.release_raises)


class Owner:
    """Stands in for a CameraController."""

    def __init__(self, name):
        self.name = name


class SharedSystemHolderTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = Owner("a"), Owner("b")

    def test_system_is_created_once_and_shared(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        first = holder.acquire(self.a)
        second = holder.acquire(self.b)
        self.assertIs(first, second)
        self.assertEqual(factory.made, 1)
        self.assertEqual(holder.count, 2)

    def test_release_instance_runs_only_when_the_last_owner_lets_go(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        holder.acquire(self.b)
        self.assertFalse(holder.release(self.a))
        self.assertNotIn("release:1", factory.log)
        self.assertIsNotNone(holder.system)
        self.assertTrue(holder.release(self.b))
        self.assertEqual(factory.log, ["get:1", "release:1"])
        self.assertIsNone(holder.system)
        self.assertEqual(holder.count, 0)

    def test_acquire_after_full_release_makes_a_fresh_system(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        holder.release(self.a)
        holder.acquire(self.a)
        self.assertEqual(factory.made, 2)

    def test_acquire_twice_by_one_owner_is_one_reference(self):
        holder = SharedSystemHolder(Factory())
        holder.acquire(self.a)
        holder.acquire(self.a)
        self.assertEqual(holder.count, 1)
        self.assertTrue(holder.release(self.a))  # one release frees it

    def test_double_release_by_one_owner_never_drops_another_owners_reference(self):
        # The audit's scenario: camera A's teardown was deferred (it keeps its
        # reference); camera B's stop() runs twice (closeEvent after
        # preview-stop). B's second release must NOT free the System under A.
        factory = Factory()
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        holder.acquire(self.b)
        self.assertFalse(holder.release(self.b))
        self.assertFalse(holder.release(self.b))
        self.assertEqual(holder.count, 1)
        self.assertTrue(holder.holds(self.a))
        self.assertNotIn("release:1", factory.log)
        self.assertIsNotNone(holder.system)

    def test_release_without_acquire_is_ignored(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        self.assertFalse(holder.release(self.a))
        self.assertEqual(holder.count, 0)
        self.assertEqual(factory.log, [])

    def test_stranger_release_does_not_touch_an_existing_reference(self):
        holder = SharedSystemHolder(Factory())
        holder.acquire(self.a)
        self.assertFalse(holder.release(self.b))
        self.assertEqual(holder.count, 1)

    def test_failed_get_instance_changes_nothing(self):
        factory = Factory(fail_on={1})
        holder = SharedSystemHolder(factory)
        with self.assertRaises(RuntimeError):
            holder.acquire(self.a)
        self.assertEqual(holder.count, 0)
        self.assertFalse(holder.holds(self.a))
        holder.acquire(self.a)  # a later attempt works
        self.assertEqual(holder.count, 1)

    def test_a_raising_release_instance_is_recorded_not_propagated(self):
        factory = Factory(release_raises=True)
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        self.assertTrue(holder.release(self.a))
        self.assertIn("cameras still referenced", holder.last_release_error)
        self.assertIsNone(holder.system)

    def test_a_controller_that_never_releases_keeps_the_system_alive(self):
        # Deferred teardown: B leaks its handle, so it never calls release().
        factory = Factory()
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        holder.acquire(self.b)  # B will leak
        holder.release(self.a)
        self.assertEqual(holder.count, 1)
        self.assertNotIn("release:1", factory.log)
        self.assertIsNotNone(holder.system)

    def test_restart_if_sole_owner_rebuilds_the_system(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        old = holder.acquire(self.a)
        self.assertTrue(holder.restart_if_sole_owner(self.a))
        self.assertEqual(factory.log, ["get:1", "release:1", "get:2"])
        self.assertIsNot(holder.system, old)
        self.assertEqual(holder.count, 1)

    def test_restart_refuses_while_another_camera_holds_the_system(self):
        factory = Factory()
        holder = SharedSystemHolder(factory)
        system = holder.acquire(self.a)
        holder.acquire(self.b)
        self.assertFalse(holder.restart_if_sole_owner(self.a))
        self.assertIs(holder.system, system)
        self.assertEqual(factory.log, ["get:1"])  # untouched

    def test_restart_refuses_a_caller_that_is_not_the_sole_owner(self):
        # Count is 1, but the one reference belongs to someone else.
        factory = Factory()
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        self.assertFalse(holder.restart_if_sole_owner(self.b))
        self.assertEqual(factory.log, ["get:1"])

    def test_restart_refuses_with_no_owner(self):
        self.assertFalse(SharedSystemHolder(Factory()).restart_if_sole_owner(self.a))

    def test_failed_restart_keeps_the_reference_and_a_retry_succeeds(self):
        factory = Factory(fail_on={2})
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        with self.assertRaises(RuntimeError):
            holder.restart_if_sole_owner(self.a)
        self.assertEqual(holder.count, 1)  # the caller still owns its reference
        self.assertIsNone(holder.system)
        self.assertTrue(holder.restart_if_sole_owner(self.a))  # retry: no double release
        self.assertIsNotNone(holder.system)
        self.assertEqual(factory.log.count("release:1"), 1)
        self.assertTrue(holder.release(self.a))
        self.assertEqual(holder.count, 0)

    def test_acquire_during_the_failed_restart_state_never_hands_out_none(self):
        factory = Factory(fail_on={2})
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        with self.assertRaises(RuntimeError):
            holder.restart_if_sole_owner(self.a)
        system = holder.acquire(self.b)  # another controller starts meanwhile
        self.assertIsNotNone(system)
        self.assertEqual(holder.count, 2)

    def test_release_in_the_failed_restart_state_does_not_raise(self):
        factory = Factory(fail_on={2})
        holder = SharedSystemHolder(factory)
        holder.acquire(self.a)
        with self.assertRaises(RuntimeError):
            holder.restart_if_sole_owner(self.a)
        self.assertTrue(holder.release(self.a))
        self.assertIsNone(holder.last_release_error)


if __name__ == "__main__":
    unittest.main()
