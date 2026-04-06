"""Audio module — file I/O and playback facade."""

from .audio_stream import AudioStream

# DEPRECATED aliases — remove after all callers migrated to AudioStream
from .audio_input import AudioInput
from .audio_player import AudioPlayer

__all__ = ["AudioStream", "AudioInput", "AudioPlayer"]
