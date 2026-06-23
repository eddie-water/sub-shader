"""dot.py — sandbox for the dot product as a frequency-measurement operation.

Backs the intuition for §3.1 of DSP.md. Run it and read the list top to
bottom: reference function on the left, its dot-product score on the right.

Everything here stays real-valued (plain sine references) on a one-second
window. That window pins the frequency bins to the integers — bin spacing is
1 / duration = 1 Hz — and makes the integer-Hz sines a clean orthogonal set.
No complex sinusoids yet; phase is a real wrinkle, surfaced in Experiment 3.

    python research/dot.py
"""
import numpy as np

DURATION_S = 1.0
SAMPLE_RATE_HZ = 64          # Nyquist = 32 Hz, comfortably above the 10 Hz top
N = int(DURATION_S * SAMPLE_RATE_HZ)
BIN_SPACING_HZ = 1.0 / DURATION_S
SWEEP_HZ = range(1, 11)
BAR_FULL_SCALE = 1.0
BAR_WIDTH = 24


def sine(freq_hz, *, amp=1.0, phase=0.0):
    t = np.arange(N) / SAMPLE_RATE_HZ
    return amp * np.sin(2.0 * np.pi * freq_hz * t + phase)


def dot(a, b):
    return float(np.dot(a, b))


def coefficient(signal, freq_hz, *, phase=0.0):
    """Dot product normalized so a unit sine at freq_hz scores 1.0.

    The N/2 divisor is the self-dot of a unit sine over an integer number of
    periods, so the score reads directly as "how much of this amplitude is
    present" once the references are orthogonal.
    """
    return dot(signal, sine(freq_hz, phase=phase)) / (N / 2.0)


def bar(value):
    filled = int(round(abs(value) / BAR_FULL_SCALE * BAR_WIDTH))
    return "#" * min(filled, BAR_WIDTH)


def report_sweep(title, signal):
    print(title)
    print("-" * len(title))
    for freq_hz in SWEEP_HZ:
        score = coefficient(signal, freq_hz)
        print(f"  sin({freq_hz:2d} Hz)  ->  {score:+.3f}  {bar(score)}")
    print()


def experiment_pure_tone():
    report_sweep(
        "1) A pure 1 Hz signal, measured against every 1..10 Hz reference",
        sine(1),
    )
    print("   Only the 1 Hz reference scores. Every other integer reference")
    print("   has its positive and negative accumulations cancel to ~0 — that")
    print("   cancellation IS orthogonality.\n")


def experiment_self_similarity():
    signal = sine(1)
    print("2) Self-similarity — nothing is more similar to a signal than itself")
    print(f"   signal . signal      = {dot(signal, signal):8.3f}   <- maximum (= energy)")
    print(f"   signal . (-signal)   = {dot(signal, -signal):8.3f}   <- flip: same size, negated")
    print(f"   signal . sin(2 Hz)   = {dot(signal, sine(2)):8.3f}   <- orthogonal: ~0")
    print()


def experiment_phase_wrinkle():
    signal = sine(1)
    print("3) The phase wrinkle — a single real sine reference is phase-blind")
    for degrees in (0, 45, 90, 180):
        score = coefficient(signal, 1, phase=np.deg2rad(degrees))
        print(f"   1 Hz signal vs 1 Hz ref shifted {degrees:3d} deg  ->  {score:+.3f}")
    print("   At 90 deg a MATCHING frequency reads as absent. 'Flipped or shifted")
    print("   still scores large' isn't quite true: flip negates, a quarter-cycle")
    print("   shift cancels. This is exactly what later forces in the cosine.\n")


def example_signal():
    """Two clean tones: 2 Hz at full amplitude, 5 Hz at half."""
    return sine(2, amp=1.0) + sine(5, amp=0.5)


def experiment_example_signal():
    signal = example_signal()
    report_sweep(
        "4) Example signal  1.0*sin(2 Hz) + 0.5*sin(5 Hz)  swept 1..10 Hz",
        signal,
    )
    print("   Two bins respond at exactly their amplitudes — 2 Hz -> 1.0, 5 Hz -> 0.5.")
    print(f"   The measurements don't interfere, and self-dot energy = {dot(signal, signal):.1f}")
    print("   = (1.0^2 + 0.5^2) * N/2 — the sum of the squared scores.\n")


def reconstruct_from(signal, refs):
    """Naive reconstruction: score every reference the same way (dot / (N/2))
    and add the scaled references back up. Exact only when refs are orthogonal.
    """
    return sum((dot(signal, ref) / (N / 2.0)) * ref for ref in refs)


def rms_error(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def experiment_reconstruction():
    signal = example_signal()
    print("5) Reconstruction — add the measured pieces back up; do we recover the signal?")
    print()
    reference_sets = [
        ("complete",  "{2, 5}",            "orthogonal, minimal",      [sine(2), sine(5)]),
        ("complete",  "{1..6}",            "orthogonal, with empties", [sine(f) for f in range(1, 7)]),
        ("redundant", "{2, 5, sin2+sin5}", "blend / not independent",  [sine(2), sine(5), sine(2) + sine(5)]),
        ("redundant", "{2, 2, 5}",         "duplicate bin",            [sine(2), sine(2), sine(5)]),
        ("redundant", "{2, 5, 2.3 Hz}",    "off-grid, between bins",   [sine(2), sine(5), sine(2.3)]),
    ]
    for kind, spec, note, refs in reference_sets:
        err = rms_error(signal, reconstruct_from(signal, refs))
        print(f"   {kind:9s} {spec:18s} {note:26s} RMS err = {err:.2e}")
    print()
    print("   Orthogonal sets rebuild it (~0). Every redundant set overshoots:")
    print("   an overlapping reference re-counts energy the others already")
    print("   measured, so the parts no longer sum back to the original.\n")


def main():
    print(f"window: {DURATION_S:g} s @ {SAMPLE_RATE_HZ} Hz  ->  N = {N} samples")
    print(f"bin spacing = 1/duration = {BIN_SPACING_HZ:g} Hz   Nyquist = {SAMPLE_RATE_HZ / 2:g} Hz\n")
    experiment_pure_tone()
    experiment_self_similarity()
    experiment_phase_wrinkle()
    experiment_example_signal()
    experiment_reconstruction()


if __name__ == "__main__":
    main()
