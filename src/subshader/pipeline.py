"""SubShader pipeline — audio visualization orchestrator."""

from subshader.utils.logging import get_logger
from subshader.utils.gpu import gpu_available
from subshader.config import RendererConfig
from subshader.audio import AudioStream
from subshader.dsp.cwt import GpuCWT, CpuCWT
from subshader.renderer import Renderer

log = get_logger(__name__)


class SubShader:
    """Main pipeline: AudioStream -> CWT -> Renderer.

    Orchestrates the three pipeline stages from a single config object.
    The run() method reads like pseudocode: get chunk, process, update.
    """

    def __init__(self, config) -> None:
        """Initialize all pipeline stages from config.

        Args:
            config: CWTConfig or PipelineConfig with file_path, chunk_size,
                    overlap_factor. sample_rate and total_samples are
                    discovered by AudioStream and written back into config.
        """
        log.info("Initializing SubShader pipeline...")

        # AudioStream wraps AudioReader (file I/O) and AudioPlayer (playback).
        # Discovers sample_rate from the file and writes it back into config.
        self.audio = AudioStream(config)

        # Select CWT backend based on GPU availability
        if gpu_available():
            self.dsp = GpuCWT(config)
        else:
            log.warning("GPU unavailable — running CpuCWT. Expect slower performance.")
            self.dsp = CpuCWT(config)

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
            file_path=config.file_path,
            frame_shape=self.dsp.get_output_shape(),
            config=renderer_config,
        )

        log.info("SubShader pipeline initialized")

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
