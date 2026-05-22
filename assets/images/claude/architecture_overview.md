# SubShader High-Level Architecture

```mermaid
flowchart TB
    %% ── Initialization Phase ─────────────────────────────────────
    subgraph INIT ["Initialization"]
        direction TB
        CONFIG["<b>Config</b><br/>Load & validate settings"]
        CONFIG --> AI["<b>AudioInput</b><br/>Open audio file"]
        CONFIG --> AP["<b>AudioPlayer</b><br/>Prepare playback stream"]
        CONFIG --> CWT["<b>Wavelet</b><br/>Build frequency scale<br/>& GPU/CPU kernels"]
        CONFIG --> VIZ["<b>ShaderPlot</b><br/>Create window, shaders,<br/>& frame buffer"]
    end

    %% ── Runtime Phase ────────────────────────────────────────────
    INIT --> START["Begin playback"]

    subgraph RUNTIME ["Runtime  ·  Audio-Clock-Driven Loop"]
        direction TB
        START --> POLL["Poll audio clock position"]
        POLL --> READY{"New chunk<br/>ready?"}
        READY -- No --> POLL
        READY -- Yes --> READ["Read audio chunk"]
        READ --> DSP["Compute CWT"]
        DSP --> RENDER["Render to screen"]
        RENDER --> POLL
    end

    %% ── Shutdown ─────────────────────────────────────────────────
    RUNTIME -- "Window closed" --> CLEANUP["Release all resources"]

    %% ── Styling ──────────────────────────────────────────────────
    classDef config fill:#8b5cf6,stroke:#7c3aed,color:#fff
    classDef audio fill:#f59e0b,stroke:#d97706,color:#fff
    classDef dsp fill:#4a9eff,stroke:#2563eb,color:#fff
    classDef viz fill:#10b981,stroke:#059669,color:#fff
    classDef loop fill:#334155,stroke:#64748b,color:#fff
    classDef decision fill:#f472b6,stroke:#ec4899,color:#fff

    class CONFIG config
    class AI,AP,READ audio
    class CWT,DSP dsp
    class VIZ,RENDER viz
    class POLL,START,CLEANUP loop
    class READY decision
```

## Modules

| Module | Role |
|--------|------|
| **Config** | Centralized settings — audio, wavelet, and visualization parameters with validation |
| **AudioInput** | Reads overlapping chunks from a WAV file on demand |
| **AudioPlayer** | Plays audio via sounddevice; its device clock is the timing master |
| **Wavelet** | Continuous Wavelet Transform — GPU-accelerated (CuPy) with NumPy fallback |
| **ShaderPlot** | GLFW window + ModernGL renderer with circular frame buffer and GLSL shaders |
