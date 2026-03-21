"""Tests for exception hierarchy consolidation (QUAL-01, QUAL-03)."""

import pytest
from subshader.exceptions import (
    SubShaderException,
    AudioFileNotFoundError,
    EndOfAudioException,
    WindowCloseException,
    GRACEFUL_EXCEPTIONS,
)


class TestExceptionHierarchy:
    """Verify all exceptions inherit from SubShaderException."""

    def test_audio_file_not_found_inherits_subshader(self):
        assert issubclass(AudioFileNotFoundError, SubShaderException)

    def test_end_of_audio_inherits_subshader(self):
        assert issubclass(EndOfAudioException, SubShaderException)

    def test_window_close_inherits_subshader(self):
        assert issubclass(WindowCloseException, SubShaderException)

    def test_audio_file_not_found_log_level(self):
        assert AudioFileNotFoundError.log_level == "error"

    def test_audio_file_not_found_caught_by_base(self):
        with pytest.raises(SubShaderException):
            raise AudioFileNotFoundError("test")


class TestNoDuplicateExceptions:
    """Verify audio_input.py uses canonical exceptions, not local copies."""

    def test_audio_input_uses_canonical_audio_file_not_found(self):
        from subshader.audio.audio_input import AudioFileNotFoundError as AudioVersion
        from subshader.exceptions import AudioFileNotFoundError as CanonicalVersion
        assert AudioVersion is CanonicalVersion

    def test_audio_input_uses_canonical_end_of_audio(self):
        from subshader.audio.audio_input import EndOfAudioException as AudioVersion
        from subshader.exceptions import EndOfAudioException as CanonicalVersion
        assert AudioVersion is CanonicalVersion


class TestGracefulExceptions:
    """Verify GRACEFUL_EXCEPTIONS scope is correct."""

    def test_runtime_error_not_in_graceful(self):
        assert RuntimeError not in GRACEFUL_EXCEPTIONS

    def test_subshader_exception_in_graceful(self):
        assert SubShaderException in GRACEFUL_EXCEPTIONS

    def test_keyboard_interrupt_in_graceful(self):
        assert KeyboardInterrupt in GRACEFUL_EXCEPTIONS


class TestGpuAvailable:
    """Verify gpu_available() utility exists and returns bool."""

    def test_gpu_available_returns_bool(self):
        from subshader.utils.gpu import gpu_available
        result = gpu_available()
        assert isinstance(result, bool)
