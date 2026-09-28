#!/usr/bin/env python3
"""Offline (non-realtime) SubShader demo video renderer.

Runs the real SubShader pipeline over a wav file with no audio device
attached, captures every rendered frame from an offscreen framebuffer, and
assembles an mp4 with the original wav muxed in as the soundtrack.

Timestamp model
----------------
The pipeline emits exactly one frame per hop, so "frames at their true audio
timestamps" reduces to an exact rational framerate:

    fps = sample_rate / hop_size   (e.g. 44100 / 8192 = 5.3833...)

That rational literal (``-framerate 44100/8192``) is passed to ffmpeg
directly - never a rounded decimal, which would drift linearly over the
length of the clip. Frame ``i`` lands at ``t = i * hop_size / sample_rate``,
which reproduces the live app exactly: the live render loop fetches chunk
``i`` when the playback clock reaches sample ``i * hop_size``.

No audio device is ever opened. The pipeline is constructed and driven by
hand via ``pipeline.audio.get_chunk()`` - the realtime run loop and the
audio device start-up are never invoked.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from subshader.utils.logging import logger_init, get_logger
from subshader.config import CWTConfig
from subshader.pipeline import SubShader

try:
    from PIL import Image
except ImportError as e:
    raise RuntimeError(
        "Pillow is required for frame capture but is not installed. "
        "Run: pip install pillow"
    ) from e

log = get_logger(__name__)

DEFAULT_AUDIO_FILE = "assets/audio/reference/beltran_sc_rip_4_bar.wav"
DEFAULT_OUTPUT = "assets/video/subshader_demo.mp4"
PROGRESS_LOG_EVERY = 25
TARGET_SIZE_MB = 10


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the offline demo renderer."""
    parser = argparse.ArgumentParser(
        prog="render_demo",
        description=(
            "Offline (non-realtime) SubShader demo video renderer. Runs the "
            "real pipeline over a wav file with no audio device, captures "
            "every rendered frame, and assembles an mp4 with the wav muxed "
            "in as the soundtrack."
        ),
        epilog=(
            "File-size knobs (target: ~10 MB for a GitHub README embed):\n"
            "  --crf      higher = smaller/lower quality (default 23)\n"
            "  --height   lower = smaller (width scales to keep 16:9 by default)\n"
            "  --duration shorter clips are smaller\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file", nargs="?", default=DEFAULT_AUDIO_FILE,
        help=f"Path to the input wav file (default: {DEFAULT_AUDIO_FILE})",
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT,
        help=f"Output mp4 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--start", type=float, default=None,
                        help="Trim start offset in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Trim duration in seconds")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output frame width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Output frame height in pixels (default: 720)")
    parser.add_argument("--crf", type=int, default=23,
                        help="x264 constant rate factor, higher = smaller file (default: 23)")
    parser.add_argument(
        "--out-fps", type=int, default=30,
        help="Re-time output to this constant fps by duplicating frames; "
             "0 keeps the native rational sample_rate/hop_size rate (default: 30)",
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=None,
        help="Directory to write captured PNGs into (default: a fresh tempdir)",
    )
    parser.add_argument("--keep-frames", action="store_true",
                        help="Do not delete the frames directory after encoding")
    parser.add_argument("--frames-only", action="store_true",
                        help="Capture PNG frames and skip mp4 encoding")

    args = parser.parse_args()

    if args.width % 2 != 0:
        parser.error(f"--width must be even, got {args.width}")
    if args.height % 2 != 0:
        parser.error(f"--height must be even, got {args.height}")

    return args


def trim_wav(src: Path, start: float | None, duration: float | None, workdir: Path) -> Path:
    """Trim a wav file with ffmpeg, or return it unchanged if no trim requested.

    Args:
        src: Source wav path.
        start: Trim start offset in seconds, or None.
        duration: Trim duration in seconds, or None.
        workdir: Directory to write the trimmed wav into.

    Returns:
        Path: `src` unchanged when both start and duration are None, otherwise
            the path to the newly written trimmed wav.

    Raises:
        RuntimeError: If ffmpeg is not on PATH.
    """
    if start is None and duration is None:
        return src

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim audio but was not found on PATH")

    trimmed_path = workdir / "trimmed.wav"
    argv = ["ffmpeg", "-y"]
    if start is not None:
        argv += ["-ss", str(start)]
    if duration is not None:
        argv += ["-t", str(duration)]
    argv += ["-i", str(src), "-c:a", "pcm_s16le", str(trimmed_path)]

    log.info(f"Trimming wav: start={start} duration={duration}")
    subprocess.run(argv, check=True)
    return trimmed_path


def capture_frames(wav_path: Path, frames_dir: Path, width: int, height: int) -> tuple[int, float, int]:
    """Drive the real SubShader pipeline offline and capture every frame as a PNG.

    Constructs the pipeline with no audio device, pulls chunks by hand via
    `pipeline.audio.get_chunk()`, and captures GPU shader output into an
    offscreen framebuffer sized independently of the (still visible) GLFW
    window.

    Args:
        wav_path: Path to the (possibly already trimmed) wav file to render.
        frames_dir: Directory to write frame_{i:06d}.png into. Must exist.
        width: Offscreen framebuffer width in pixels.
        height: Offscreen framebuffer height in pixels.

    Returns:
        tuple[int, float, int]: (frame_count, sample_rate, hop_size) - sample_rate
            and hop_size are read off the config after SubShader construction,
            since AudioReader writes the discovered sample_rate back into it.
    """
    config = CWTConfig(file_path=str(wav_path))
    errors = config.validate()
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    pipeline = SubShader(config)
    try:
        ctx = pipeline.renderer.gl_context.ctx
        fbo = ctx.simple_framebuffer((width, height))

        sample_rate = config.sample_rate
        hop_size = config.hop_size

        frame_count = 0
        while True:
            if pipeline.renderer.should_close():
                log.warning("Renderer window closed - capture stopped early")
                break

            chunk = pipeline.audio.get_chunk()
            if chunk is None:
                break

            coefs = pipeline.dsp.process(chunk)
            pipeline.renderer.update(coefs)

            fbo.use()
            fbo.clear(0.0, 0.0, 0.0)
            pipeline.renderer.gpu_renderer.render_graphic()
            pixels = fbo.read(components=3, dtype="f1")
            ctx.screen.use()

            arr = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
            arr = np.ascontiguousarray(arr[::-1])
            Image.fromarray(arr, mode="RGB").save(frames_dir / f"frame_{frame_count:06d}.png")

            if frame_count % PROGRESS_LOG_EVERY == 0:
                elapsed_s = frame_count * hop_size / sample_rate
                log.info(f"Captured frame {frame_count} (audio t={elapsed_s:.2f}s)")

            frame_count += 1

        return frame_count, sample_rate, hop_size
    finally:
        pipeline.cleanup()


def assemble_video(
    frames_dir: Path,
    wav_path: Path,
    output: Path,
    sample_rate: float,
    hop_size: int,
    crf: int,
    out_fps: int,
) -> Path:
    """Assemble captured PNG frames + the wav soundtrack into an mp4 via ffmpeg.

    The rational `-framerate` on the image input is the sync source of truth;
    `-r out_fps` (when requested) only re-times the output to a constant fps
    by duplicating frames - it does not change frame timestamps.

    Args:
        frames_dir: Directory containing frame_%06d.png sequence.
        wav_path: Wav file to mux in as the soundtrack (same file rendered).
        output: Destination mp4 path.
        sample_rate: Sample rate discovered from the rendered wav.
        hop_size: Pipeline hop size in samples.
        crf: x264 constant rate factor.
        out_fps: Constant output fps, or 0 to keep the native rational rate.

    Returns:
        Path: The output mp4 path.

    Raises:
        RuntimeError: If the ffmpeg invocation fails.
    """
    framerate = f"{int(round(sample_rate))}/{hop_size}"
    argv = [
        "ffmpeg", "-y",
        "-framerate", framerate,
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(wav_path),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k",
    ]
    if out_fps > 0:
        argv += ["-r", str(out_fps)]
    argv += ["-shortest", str(output)]

    log.info(f"Encoding video: framerate={framerate} crf={crf} out_fps={out_fps}")
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encode failed: {' '.join(argv)}") from e

    return output


def main() -> int:
    logger_init(log_level="INFO", console_output=True, file_output=False)
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    user_supplied_frames_dir = args.frames_dir is not None
    if user_supplied_frames_dir:
        frames_dir = args.frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix="render_demo_"))

    try:
        audio_path = Path(args.audio_file)
        rendered_wav = trim_wav(audio_path, args.start, args.duration, frames_dir)

        frame_count, sample_rate, hop_size = capture_frames(
            rendered_wav, frames_dir, args.width, args.height
        )

        framerate = f"{int(round(sample_rate))}/{hop_size}"
        log.info(f"Captured {frame_count} frames at exact rate {framerate} fps")

        if args.frames_only:
            log.info(f"--frames-only: PNG frames left at {frames_dir}")
            return 0

        assemble_video(
            frames_dir, rendered_wav, output_path,
            sample_rate, hop_size, args.crf, args.out_fps,
        )

        duration_s = frame_count * hop_size / sample_rate
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info(
            f"Wrote {output_path} | frames={frame_count} | fps={framerate} | "
            f"duration={duration_s:.3f}s | size={size_mb:.2f} MB"
        )
        if size_mb > TARGET_SIZE_MB:
            log.warning(
                f"Output is {size_mb:.2f} MB, over the {TARGET_SIZE_MB} MB embed target - "
                f"raise --crf, lower --height, or shorten with --duration"
            )

        return 0
    finally:
        if not args.keep_frames and not user_supplied_frames_dir:
            shutil.rmtree(frames_dir, ignore_errors=True)
        elif args.keep_frames:
            log.info(f"--keep-frames: frames retained at {frames_dir}")


if __name__ == "__main__":
    sys.exit(main())
