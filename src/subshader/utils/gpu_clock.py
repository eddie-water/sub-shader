"""GPU clock lock - pins SM and memory clock floors for the pipeline's lifetime.

The render loop works ~35 ms out of every 186 ms frame, which the NVIDIA
driver's DVFS reads as an idle GPU: after its post-burst boost window
(~6.5 s) it steps the memory clock down to 810 MHz and the SM clock toward
idle, and the loop's own periodic work never re-arms the boost. The
memory-bound IFFT then runs ~19 ms instead of its hot-clock ~2.8 ms.

Locking clock floors (nvidia-smi -lgc / -lmc) removes that tax
deterministically. On WSL the lock must be applied by the Windows-host
nvidia-smi.exe, which requires elevation - handled here by a single UAC
prompt via PowerShell Start-Process -Verb RunAs. The lock is machine-state:
it survives process exit and is only cleared by `release()` (elevated),
`nvidia-smi -rgc -rmc`, or a reboot. The app therefore locks at startup if
needed and intentionally does NOT auto-release on exit - releasing would
cost a second UAC prompt every run, and a still-locked GPU is merely warm
idle (~30 W), never unreachable.

Floors are hardware-specific, measured on the RTX 4060 Ti dev/demo GPU:
2610 MHz SM is the sustained-boost tier; 5001 MHz memory is the lowest
P-state with full-rate FFT throughput.

CLI: python -m subshader.utils.gpu_clock [status|lock|release]
"""

import shutil
import subprocess

from subshader.utils.logging import get_logger

log = get_logger(__name__)

SM_FLOOR_MHZ = 2610
SM_MAX_MHZ = 3105
MEM_FLOOR_MHZ = 5001
MEM_MAX_MHZ = 9001

# The driver snaps requested floors to real P-states (e.g. 5001 -> 5000 MHz),
# so verification must allow a small shortfall from the requested value.
FLOOR_TOLERANCE_MHZ = 15

# UAC prompt wait: long enough to find the mouse, short enough that an
# unattended launch fails into the unlocked path instead of hanging startup.
ELEVATION_TIMEOUT_S = 60


def _find_nvidia_smi() -> str | None:
    """Locate nvidia-smi: native first, then the Windows-host binary (WSL)."""
    return shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")


def _run(cmd: list[str], timeout: float = 15) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(f"GPU clock command failed to run: {' '.join(cmd)} ({e})")
        return None


def query_clocks() -> tuple[int, int] | None:
    """Return current (sm_mhz, mem_mhz), or None if unqueryable."""
    smi = _find_nvidia_smi()
    if smi is None:
        return None
    result = _run([smi, "--query-gpu=clocks.sm,clocks.mem",
                   "--format=csv,noheader,nounits"])
    if result is None or result.returncode != 0:
        return None
    try:
        sm, mem = (int(v) for v in result.stdout.strip().split(","))
    except ValueError:
        return None
    return sm, mem


def is_locked() -> bool:
    """Heuristic lock check: with floors locked, even an idle GPU reads at or
    above them. (nvidia-smi exposes no direct lock-state field.) A busy GPU
    at boost can read as locked, but re-locking is then harmless anyway."""
    clocks = query_clocks()
    if clocks is None:
        return False
    sm, mem = clocks
    return (sm >= SM_FLOOR_MHZ - FLOOR_TOLERANCE_MHZ
            and mem >= MEM_FLOOR_MHZ - FLOOR_TOLERANCE_MHZ)


def _lock_args() -> list[list[str]]:
    return [
        ["-lgc", f"{SM_FLOOR_MHZ},{SM_MAX_MHZ}"],
        ["-lmc", f"{MEM_FLOOR_MHZ},{MEM_MAX_MHZ}"],
    ]


def _release_args() -> list[list[str]]:
    return [["-rgc"], ["-rmc"]]


def _apply_direct(smi: str, arg_sets: list[list[str]]) -> bool:
    """Run each nvidia-smi invocation unelevated. True only if all succeed."""
    for args in arg_sets:
        result = _run([smi, *args])
        if result is None or result.returncode != 0:
            return False
    return True


def _apply_elevated_windows(arg_sets: list[list[str]]) -> bool:
    """Apply all invocations through one Windows UAC prompt (WSL path).

    A single elevated PowerShell runs every nvidia-smi call, so locking both
    clock domains costs one prompt. Declining or ignoring the prompt leaves
    the clocks untouched.
    """
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return False
    inner = "; ".join("nvidia-smi " + " ".join(args) for args in arg_sets)
    result = _run(
        [powershell, "-Command",
         "Start-Process -FilePath 'powershell' "
         f"-ArgumentList '-WindowStyle','Hidden','-Command','{inner}' "
         "-Verb RunAs -Wait"],
        timeout=ELEVATION_TIMEOUT_S,
    )
    # Start-Process only reports whether the prompt was serviced; the
    # caller verifies the result by re-querying clocks.
    return result is not None and result.returncode == 0


def _apply(arg_sets: list[list[str]], verify) -> str:
    """Shared lock/release flow: direct attempt, elevated fallback, verify.

    Returns one of: "applied", "applied-elevated", "unavailable", "failed".
    """
    smi = _find_nvidia_smi()
    if smi is None:
        log.info("nvidia-smi not found - GPU clock control unavailable")
        return "unavailable"

    if _apply_direct(smi, arg_sets) and verify():
        return "applied"

    # WSL: even the native /usr/lib/wsl/lib/nvidia-smi cannot change clocks -
    # only the Windows-host nvidia-smi run elevated can. PowerShell being
    # reachable is the "there is a Windows host" signal.
    if shutil.which("powershell.exe"):
        log.info("GPU clock change needs elevation - requesting UAC approval "
                 "on the Windows host...")
        if _apply_elevated_windows(arg_sets) and verify():
            return "applied-elevated"

    log.warning("Could not change GPU clocks (permission denied or prompt "
                "declined)")
    return "failed"


def ensure_locked() -> str:
    """Lock clock floors if not already locked. Never raises; the pipeline
    runs unlocked (with the downclock tax) on any failure.

    Returns: "already-locked", "applied", "applied-elevated", "unavailable",
    or "failed".
    """
    if is_locked():
        log.info("GPU clock floors already locked "
                 f"(>= {SM_FLOOR_MHZ}/{MEM_FLOOR_MHZ} MHz)")
        return "already-locked"
    status = _apply(_lock_args(), is_locked)
    if status.startswith("applied"):
        log.info(f"GPU clock floors locked: SM >= {SM_FLOOR_MHZ} MHz, "
                 f"memory >= {MEM_FLOOR_MHZ} MHz ({status})")
    return status


def release() -> str:
    """Reset both clock domains to driver-managed (undoes ensure_locked).

    Returns: "applied", "applied-elevated", "unavailable", or "failed".
    """
    status = _apply(_release_args(), lambda: not is_locked())
    if status.startswith("applied"):
        log.info(f"GPU clocks returned to driver control ({status})")
    return status


def _main() -> None:
    import sys
    verb = sys.argv[1] if len(sys.argv) > 1 else "status"
    if verb == "status":
        clocks = query_clocks()
        if clocks is None:
            print("nvidia-smi unavailable")
        else:
            state = "locked" if is_locked() else "unlocked"
            print(f"SM {clocks[0]} MHz, memory {clocks[1]} MHz ({state})")
    elif verb == "lock":
        print(ensure_locked())
    elif verb == "release":
        print(release())
    else:
        print(f"unknown verb '{verb}' - use status|lock|release")
        sys.exit(2)


if __name__ == "__main__":
    _main()
