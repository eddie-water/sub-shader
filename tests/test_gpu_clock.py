"""Tests for the GPU clock lock helper (all nvidia-smi calls mocked)."""

import subprocess
from unittest.mock import patch

import pytest

from subshader.utils import gpu_clock


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr="")


class TestQueryClocks:
    def test_parses_clock_pair(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "_run", return_value=completed("2610, 5000\n")):
            assert gpu_clock.query_clocks() == (2610, 5000)

    def test_no_nvidia_smi_returns_none(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value=None):
            assert gpu_clock.query_clocks() is None

    def test_unparseable_output_returns_none(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "_run", return_value=completed("[N/A], [N/A]\n")):
            assert gpu_clock.query_clocks() is None

    def test_failed_command_returns_none(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "_run", return_value=completed(returncode=4)):
            assert gpu_clock.query_clocks() is None


class TestIsLocked:
    def test_clocks_at_floor_is_locked(self):
        with patch.object(gpu_clock, "query_clocks", return_value=(2610, 5001)):
            assert gpu_clock.is_locked()

    def test_driver_snapped_floor_within_tolerance_is_locked(self):
        # Driver snaps a requested 5001 MHz memory floor to the real 5000 MHz
        # P-state; verification must not read that as unlocked.
        with patch.object(gpu_clock, "query_clocks", return_value=(2610, 5000)):
            assert gpu_clock.is_locked()

    def test_idle_clocks_are_unlocked(self):
        with patch.object(gpu_clock, "query_clocks", return_value=(210, 405)):
            assert not gpu_clock.is_locked()

    def test_memory_parked_is_unlocked(self):
        with patch.object(gpu_clock, "query_clocks", return_value=(2610, 810)):
            assert not gpu_clock.is_locked()

    def test_unqueryable_is_unlocked(self):
        with patch.object(gpu_clock, "query_clocks", return_value=None):
            assert not gpu_clock.is_locked()


class TestEnsureLocked:
    def test_already_locked_short_circuits(self):
        with patch.object(gpu_clock, "is_locked", return_value=True), \
             patch.object(gpu_clock, "_run") as run:
            assert gpu_clock.ensure_locked() == "already-locked"
            run.assert_not_called()

    def test_no_nvidia_smi_is_unavailable(self):
        with patch.object(gpu_clock, "is_locked", return_value=False), \
             patch.object(gpu_clock, "_find_nvidia_smi", return_value=None):
            assert gpu_clock.ensure_locked() == "unavailable"

    def test_direct_lock_success(self):
        locked = {"value": False}

        def run_and_lock(cmd, timeout=15):
            locked["value"] = True
            return completed()

        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", side_effect=lambda: locked["value"]), \
             patch.object(gpu_clock, "_run", side_effect=run_and_lock):
            assert gpu_clock.ensure_locked() == "applied"

    def test_permission_denied_falls_back_to_elevation(self):
        locked = {"value": False}

        def elevate(arg_sets):
            locked["value"] = True
            return True

        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", side_effect=lambda: locked["value"]), \
             patch.object(gpu_clock, "_run", return_value=completed(returncode=4)), \
             patch.object(gpu_clock.shutil, "which", return_value="powershell.exe"), \
             patch.object(gpu_clock, "_apply_elevated_windows", side_effect=elevate):
            assert gpu_clock.ensure_locked() == "applied-elevated"

    def test_declined_elevation_fails_gracefully(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", return_value=False), \
             patch.object(gpu_clock, "_run", return_value=completed(returncode=4)), \
             patch.object(gpu_clock.shutil, "which", return_value="powershell.exe"), \
             patch.object(gpu_clock, "_apply_elevated_windows", return_value=False):
            assert gpu_clock.ensure_locked() == "failed"

    def test_no_windows_host_fails_without_elevation_attempt(self):
        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", return_value=False), \
             patch.object(gpu_clock, "_run", return_value=completed(returncode=4)), \
             patch.object(gpu_clock.shutil, "which", return_value=None), \
             patch.object(gpu_clock, "_apply_elevated_windows") as elevated:
            assert gpu_clock.ensure_locked() == "failed"
            elevated.assert_not_called()


class TestRelease:
    def test_direct_release_success(self):
        unlocked = {"value": False}

        def run_and_release(cmd, timeout=15):
            unlocked["value"] = True
            return completed()

        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", side_effect=lambda: not unlocked["value"]), \
             patch.object(gpu_clock, "_run", side_effect=run_and_release):
            assert gpu_clock.release() == "applied"

    def test_release_uses_reset_flags(self):
        recorded = []

        def record(cmd, timeout=15):
            recorded.append(cmd)
            return completed()

        with patch.object(gpu_clock, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(gpu_clock, "is_locked", return_value=False), \
             patch.object(gpu_clock, "_run", side_effect=record):
            gpu_clock.release()
        flags = [arg for cmd in recorded for arg in cmd if arg.startswith("-r")]
        assert flags == ["-rgc", "-rmc"]
