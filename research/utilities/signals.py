"""Signal registry — single source of truth for test/comparison signals.

Adding a new signal: append a dict to SIGNALS and drop the audio file
in assets/audio/reference/ (or assets/audio/generated/ for synthesized).
"""

SIGNALS = [
    {
        "name": "chirp",
        "label": "Bouncing Chirp",
        "audio": "assets/audio/generated/bouncing_chirp.wav",
        "reference": "assets/images/figures/bouncing_chirp_edison.png",
        "type": "synthetic",
    },
    {
        "name": "polyphonic",
        "label": "MIDI Sine Waves",
        "audio": "assets/audio/reference/midi_sine_waves.wav",
        "reference": "assets/images/figures/midi_sine_wave_edison.png",
        "type": "file",
    },
    {
        "name": "musical",
        "label": "Beltran (4 Bars)",
        "audio": "assets/audio/reference/beltran_sc_rip_4_bar.wav",
        "reference": "assets/images/figures/beltran_sc_rip_4_bar_edison.png",
        "type": "file",
    },
]


def get_signal(name: str) -> dict:
    """Look up a signal by name. Raises ValueError if not found."""
    for s in SIGNALS:
        if s["name"] == name:
            return s
    raise ValueError(f"Unknown signal: {name!r}. Available: {[s['name'] for s in SIGNALS]}")
