#!/usr/bin/env python3
"""SubShader — real-time audio visualization."""

import argparse
from subshader.utils.logging import logger_init
from subshader.config import CWTConfig
from subshader.pipeline import SubShader
from subshader.exceptions import GRACEFUL_EXCEPTIONS, reporter


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(prog="subshader",
                                     description="SubShader real-time audio visualizer")
    parser.add_argument("audio_file", nargs="?", default=None,
                        help="Path to WAV audio file (uses default if not provided)")
    return parser.parse_args()


def main():
    """Main entry point for the SubShader application."""
    logger_init(log_level="INFO", console_output=True, file_output=True)
    args = parse_args()

    config = CWTConfig()
    if args.audio_file:
        config.file_path = args.audio_file

    pipeline = SubShader(config)
    try:
        pipeline.run()
    except GRACEFUL_EXCEPTIONS as e:
        reporter.report(e)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
