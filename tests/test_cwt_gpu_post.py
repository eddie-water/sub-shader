"""
Tests for GPU-resident CWT post-processing.

GpuCWT keeps the complex coefficient matrix on the device through every
post() stage and downloads only the finished frame. These tests pin the GPU
path to the CPU reference (same math, different array namespace) and prove
the reduceat-free max-pool matches NumPy exactly on non-divisible widths.
"""

import numpy as np
import pytest

from subshader.config import CWTConfig
from subshader.dsp.cwt import CpuCWT, GpuCWT
from subshader.utils.gpu import gpu_available

cp = pytest.importorskip("cupy")

requires_gpu = pytest.mark.skipif(not gpu_available(), reason="No CUDA device available")

SAMPLE_RATE = 44100.0
CHUNK_SIZE = 16384


def _synthetic_chunk(seed: int = 0) -> np.ndarray:
    """Three sustained tones plus a short click and low-level noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(CHUNK_SIZE) / SAMPLE_RATE
    chunk = (
        0.5 * np.sin(2 * np.pi * 110.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1000.0 * t)
        + 0.2 * np.sin(2 * np.pi * 6000.0 * t)
        + 0.01 * rng.standard_normal(CHUNK_SIZE)
    )
    click_start = CHUNK_SIZE * 3 // 4
    chunk[click_start : click_start + 8] += 1.0
    return chunk


def _reference_max_pool(coefs: np.ndarray, target_width: int) -> np.ndarray:
    """The original NumPy formulation: reduceat over linspace block edges."""
    num_samples = coefs.shape[1]
    edges = np.linspace(0, num_samples, target_width + 1).astype(int)
    return np.maximum.reduceat(coefs, edges[:-1], axis=1)


@pytest.fixture(scope="module")
def cwt_config():
    return CWTConfig(chunk_size=CHUNK_SIZE, sample_rate=SAMPLE_RATE)


@pytest.fixture(scope="module")
def cpu_cwt(cwt_config):
    return CpuCWT(cwt_config)


@pytest.fixture(scope="module")
def gpu_cwt(cwt_config):
    if not gpu_available():
        pytest.skip("No CUDA device available")
    dsp = GpuCWT(cwt_config)
    yield dsp
    dsp.cleanup()


@requires_gpu
class TestGpuPostEquivalence:
    """GPU-resident post() must reproduce the CPU reference end to end."""

    def test_process_matches_cpu(self, cpu_cwt, gpu_cwt):
        chunk = _synthetic_chunk()
        cpu_frame = cpu_cwt.process(chunk)
        gpu_frame = gpu_cwt.process(chunk)
        np.testing.assert_allclose(
            gpu_frame, cpu_frame, rtol=1e-3, atol=1e-3 * cpu_frame.max()
        )

    def test_transform_stays_on_device(self, gpu_cwt):
        raw = gpu_cwt.transform(gpu_cwt.pre(_synthetic_chunk()))
        assert isinstance(raw, cp.ndarray)
        assert raw.shape == (gpu_cwt.num_freqs, gpu_cwt.input_n)

    def test_post_accepts_host_input(self, gpu_cwt):
        """A host-side complex matrix (e.g. an instrumented transform) must still post()."""
        raw_gpu = gpu_cwt.transform(gpu_cwt.pre(_synthetic_chunk()))
        from_device = gpu_cwt.post(raw_gpu)
        from_host = gpu_cwt.post(cp.asnumpy(raw_gpu))
        assert isinstance(from_host, np.ndarray)
        assert from_host.shape == from_device.shape
        # np.abs and cp.abs round complex magnitudes differently at the last float32 bit
        np.testing.assert_allclose(from_host, from_device, rtol=1e-5)


@requires_gpu
class TestGpuPostOutputContract:
    """Frame buffer and renderer depend on this exact shape and dtype."""

    def test_output_is_host_float32_with_output_shape(self, gpu_cwt):
        frame = gpu_cwt.process(_synthetic_chunk())
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.float32
        assert frame.shape == gpu_cwt.get_output_shape()

    def test_timing_stages_recorded(self, gpu_cwt):
        gpu_cwt.process(_synthetic_chunk())
        for attr in (
            "_timing_transform_ms",
            "_timing__compute_mag_ms",
            "_timing_discard_unreliable_coefs_ms",
            "_timing_extract_hop_center_ms",
            "_timing_downsample_ms",
            "_timing__download_ms",
        ):
            assert getattr(gpu_cwt, attr) >= 0.0


class TestMaxPoolMatchesReduceat:
    """The gather-based max-pool must equal np.maximum.reduceat bit for bit."""

    NON_DIVISIBLE_CASES = [
        (1000, 64),
        (8191, 64),
        (130, 64),
        (100, 7),
        (65, 64),
        (4097, 64),
    ]

    @pytest.mark.parametrize("num_samples,target_width", NON_DIVISIBLE_CASES)
    def test_cpu_non_divisible(self, cpu_cwt, num_samples, target_width):
        coefs = np.random.default_rng(1).random((12, num_samples))
        out = cpu_cwt.downsample(coefs, target_width)
        np.testing.assert_array_equal(out, _reference_max_pool(coefs, target_width))

    @requires_gpu
    @pytest.mark.parametrize("num_samples,target_width", NON_DIVISIBLE_CASES)
    def test_gpu_non_divisible(self, gpu_cwt, num_samples, target_width):
        coefs = np.random.default_rng(2).random((12, num_samples)).astype(np.float32)
        out = gpu_cwt.downsample(cp.asarray(coefs), target_width)
        assert isinstance(out, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(out), _reference_max_pool(coefs, target_width)
        )

    @requires_gpu
    def test_gpu_divisible(self, gpu_cwt):
        coefs = np.random.default_rng(3).random((12, 8192)).astype(np.float32)
        out = gpu_cwt.downsample(cp.asarray(coefs), 64)
        np.testing.assert_array_equal(cp.asnumpy(out), _reference_max_pool(coefs, 64))

    @requires_gpu
    def test_gpu_spike_survives_non_divisible(self, gpu_cwt):
        coefs = np.zeros((4, 1000), dtype=np.float32)
        coefs[2, 517:520] = 7.0
        out = cp.asnumpy(gpu_cwt.downsample(cp.asarray(coefs), 64))
        assert out.max() == pytest.approx(7.0)
        assert np.count_nonzero(out) == 1
