"""SubShader pipeline — audio visualization orchestrator."""

import numpy as np

from subshader.utils.logging import get_logger
from subshader.utils.gpu import gpu_available
from subshader.utils.timing import timed
from subshader.config import PipelineConfig, CWTConfig, RendererConfig
from subshader.audio import AudioStream
from subshader.dsp.cwt import GpuCWT, CpuCWT
from subshader.renderer import Renderer

log = get_logger(__name__)

# Number of evenly-spaced chunks sampled during the pre-scan intensity survey
PRESCAN_NUM_CHUNKS = 10


class SubShader:
    """Main pipeline: AudioStream -> CWT -> Renderer.

    Orchestrates the three pipeline stages from a single config object.
    The run() method reads like pseudocode: get chunk, process, update.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize all pipeline stages from base pipeline config.

        SubShader constructs module-specific configs internally. AudioStream
        discovers sample_rate and total_samples and writes them back into the
        shared base config. CWTConfig is constructed after AudioStream so that
        the now-populated sample_rate can be copied over.

        Args:
            config: PipelineConfig with file_path, chunk_size, overlap_factor.
                    sample_rate and total_samples are discovered by AudioStream
                    and written back into config.
        """
        log.info("Initializing SubShader pipeline...")

        # AudioStream wraps AudioReader (file I/O) and AudioPlayer (playback).
        # Discovers sample_rate from the file and writes it back into config.
        self.audio = AudioStream(config)

        # Construct CWTConfig with all discovered runtime values.
        # sample_rate is now populated by AudioStream. Wavelet-specific params are
        # carried over from the incoming config when present (it may already be a
        # CWTConfig), falling back to CWTConfig defaults for a base PipelineConfig —
        # otherwise num_octaves/notes_per_octave/etc. would be silently dropped.
        cwt_config = CWTConfig(
            file_path=config.file_path,
            chunk_size=config.chunk_size,
            overlap_factor=config.overlap_factor,
            sample_rate=config.sample_rate,
            total_samples=config.total_samples,
            notes_per_octave=getattr(config, "notes_per_octave", CWTConfig.notes_per_octave),
            num_octaves=getattr(config, "num_octaves", CWTConfig.num_octaves),
            root_note_hz=getattr(config, "root_note_hz", CWTConfig.root_note_hz),
            target_width=getattr(config, "target_width", CWTConfig.target_width),
            num_cycles=getattr(config, "num_cycles", CWTConfig.num_cycles),
            num_fwhm_cycles=getattr(config, "num_fwhm_cycles", CWTConfig.num_fwhm_cycles),
        )

        # Select CWT backend based on GPU availability
        if gpu_available():
            self.dsp = GpuCWT(cwt_config)
        else:
            log.warning("GPU unavailable — running CpuCWT. Expect slower performance.")
            self.dsp = CpuCWT(cwt_config)

        # Renderer creates the GLFW window and GPU rendering pipeline.
        # frame_shape comes from the CWT output so the texture matches exactly.
        # RendererConfig carries display-specific params (num_frames, color_norm).
        renderer_config = RendererConfig(
            file_path=config.file_path,
            chunk_size=config.chunk_size,
            overlap_factor=config.overlap_factor,
            sample_rate=config.sample_rate,
        )
        self.renderer = Renderer(
            frame_shape=self.dsp.get_output_shape(),
            config=renderer_config,
        )

        # Pre-scan the audio file to determine a fixed intensity reference.
        # Must happen after Renderer is constructed (renderer_config carries the percentile).
        self._fixed_intensity_max = self._prescan_intensity(renderer_config)
        self.renderer.set_fixed_intensity_max(self._fixed_intensity_max)

        log.info("SubShader pipeline initialized")

    @timed
    def _prescan_intensity(self, renderer_config: RendererConfig) -> float:
        """Sample evenly-spaced CWT frames to compute a fixed intensity reference.

        Reads PRESCAN_NUM_CHUNKS chunks spread across the audio file, runs each
        through the CWT, and takes the percentile-based max across all samples.
        The reader is reset to position 0 after scanning so playback starts from
        the beginning.

        Args:
            renderer_config: Carries color_norm.global_intensity_percentile.

        Returns:
            float: Fixed intensity_max to use for the entire playback session.
        """
        reader = self.audio._reader
        total_samples = reader.total_samples
        chunk_size = reader.chunk_size
        percentile = renderer_config.color_norm.global_intensity_percentile

        # Compute evenly-spaced seek positions (hop to each sample point)
        step = max(1, (total_samples - chunk_size) // PRESCAN_NUM_CHUNKS)
        seek_positions = [i * step for i in range(PRESCAN_NUM_CHUNKS) if i * step + chunk_size <= total_samples]

        intensity_max = 0.0
        for pos in seek_positions:
            reader.file_pos = pos
            chunk = reader.get_chunk()
            if chunk is None:
                continue
            coefs = self.dsp.process(chunk)
            frame_value = float(np.percentile(np.abs(coefs), percentile))
            intensity_max = max(intensity_max, frame_value)

        # Reset reader to start of file for normal playback
        reader.file_pos = 0

        # Floor to avoid division-by-zero in shader
        intensity_max = max(intensity_max, 1e-8)

        log.info(f"Pre-scan complete: fixed intensity_max = {intensity_max:.4f}")
        return intensity_max

    def run(self) -> None:
        """Main visualization loop.

        Drives the audio-clock-synchronized render loop. The audio device clock
        is the single source of truth (D-06): next_chunk() blocks until the
        audio clock has advanced, then returns the aligned chunk.
        """
        self.audio.start()
        while not self.renderer.should_close():
            chunk = self.audio.next_chunk()
            if chunk is None:
                # End of file — audio will loop; next_chunk() handles the reset
                continue
            coefs = self.dsp.process(chunk)
            self.renderer.update(coefs)

    def cleanup(self) -> None:
        """Release all resources. Safe to call multiple times."""
        if hasattr(self, 'audio') and self.audio is not None:
            try:
                self.audio.cleanup()
            except Exception as e:
                log.warning(f"Error during audio cleanup: {e}")
            finally:
                self.audio = None

        if hasattr(self, 'dsp') and self.dsp is not None:
            try:
                self.dsp.cleanup()
            except Exception as e:
                log.warning(f"Error during DSP cleanup: {e}")
            finally:
                self.dsp = None

        if hasattr(self, 'renderer') and self.renderer is not None:
            try:
                self.renderer.cleanup()
            except Exception as e:
                log.warning(f"Error during renderer cleanup: {e}")
            finally:
                self.renderer = None

        log.info("SubShader pipeline cleaned up")
