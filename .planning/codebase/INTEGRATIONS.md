# External Integrations

**Analysis Date:** 2026-03-21

## APIs & External Services

**No remote APIs detected** - This is a standalone desktop application with no cloud service integrations or remote API calls.

**Benchmarking & Comparison:**
- STFT (Short-Time Fourier Transform) - Included for performance comparison via `scipy.signal.stft`
- PyWavelets CWT - External library comparison; used in research benchmarking
- ANTS (Analyzing Neural Time-Series) - CWT implementation from educational course used for algorithm validation

## Data Storage

**Audio Files:**
- Local filesystem only - No remote audio storage or streaming
- Supported format: WAV (via soundfile)
- Default location: `assets/audio/daw/a2a3_a4_minor_scale.wav` (configured in `src/subshader/__main__.py`)
- Configurable path: `AudioConfig.file_path` in `src/subshader/config.py`

**Processed Results:**
- Circular buffer in memory only (see `CircularFrameBuffer` in `src/subshader/viz/plotter.py`)
- Benchmark results saved to CSV: `research/benchmark_results.csv`
- No persistent database or cloud storage

**Logs:**
- File-based logging to local `logs/` directory
- Configured via `src/subshader/utils/logging.py` with console and file output options
- Log level control: `logger_init(log_level="INFO", console_output=False, file_output=True)`

**Research Data:**
- Benchmark artifacts in `research/` directory (images, timing data)
- No integration with data platforms or analytics services

## Authentication & Identity

**Not applicable** - No user authentication system or identity provider.

- No login mechanisms
- No API keys or credentials required
- Single-user desktop application

## Monitoring & Observability

**Error Tracking:**
- Custom exception handling in `src/subshader/exceptions.py`
- Exception reporter: `exceptions.reporter.report(e)` for graceful error handling
- No remote error tracking service (Sentry, etc.)

**Performance Monitoring:**
- Built-in loop timer: `LoopTimer` in `src/subshader/utils/loop_timer.py`
- Frame rate counter: `FrameCounterPyQT5` in `src/subshader/utils/frame_counter_pyqt5.py`
- Diagnostics utilities: `src/subshader/utils/gl_diagnostics.py` for OpenGL diagnostics
- Benchmark framework: `research/benchmark.py` for comparative performance analysis

**Logs:**
- File-based logging (no remote logging service)
- Location: `logs/` directory
- Controlled via `logger_init()` in `src/subshader/utils/logging.py`

## CI/CD & Deployment

**Hosting:**
- Not detected - Desktop application
- Local execution only (runs on user's machine)

**CI Pipeline:**
- Not detected
- No GitHub Actions, GitLab CI, or equivalent automation
- No deployment scripts or pipeline configuration

**Build & Install:**
- Manual installation via setuptools/pip
- No automated builds or releases detected

## Environment Configuration

**Required env vars:**
- `DISPLAY` - Graphics display server (WSL2 specific, auto-configured to `:0` if not set)
- No other variables required for basic operation

**Optional env vars:**
- `SUBSHADER_DEBUG` - Enable OpenGL debug output (set to `'1'` for debug mode)

**Secrets location:**
- No secrets management system
- No `.env` file patterns detected (`.vscode/settings.json` references `${workspaceFolder}/.env` but file doesn't exist)
- No credentials, API keys, or sensitive configuration required

**Platform-specific configuration:**
- WSL detection and graphics setup in `src/subshader/utils/os_env_setup.py`
- Automatic DISPLAY and OpenGL configuration for WSL environments
- Fallback display dimensions: 1920x1080 if system detection fails (see `_get_system_display_size()` in `src/subshader/config.py`)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

**Input Events:**
- Window close detection via GLFW callback (see `src/subshader/viz/plotter.py`)
- No network webhooks or external event subscriptions

---

*Integration audit: 2026-03-21*
