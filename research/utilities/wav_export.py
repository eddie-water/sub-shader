import numpy as np


def export_signal_to_wav(signal: np.ndarray, sample_rate: int, path: str) -> None:
    """Write a 1-D float signal to a 16-bit WAV file."""
    import soundfile as sf
    # Normalize to [-1, 1] if needed
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    print(f"Exported WAV -> {path}  ({len(signal)} samples, {sample_rate} Hz, "
          f"{len(signal) / sample_rate:.2f}s)")
