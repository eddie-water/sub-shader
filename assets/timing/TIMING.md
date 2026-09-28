# SubShader Timing

✅ **Real-time** - 5.9 ms of work per 186 ms frame budget, a 32× margin. The worst frame observed took 11 ms - still 16.8× under the deadline.

## Start Up

**Start up pays 799 ms once - allocations, kernel builds, GPU warmup - so every frame of the Process Loop stays lean.**

![Start up - one-time construction, CPU and GPU lanes](../images/drawio/vertical_startup_v13_black.png)

![Start up per-stage cascade, ending at the first Process Loop deadline window - rows α–κ match the flowchart blocks](timing_startup_hybrid_v13.png)

## Process Loop

**One frame of audio propagates end to end in 5.9 ms, against a 186 ms deadline.**

![Process Loop - the per-frame pipeline, CPU and GPU lanes](../images/drawio/vertical_runtime_v13_black.png)

![Process Loop per-stage cascade - rows λ–ω match the flowchart blocks](timing_runtime_hybrid_v13.png)

## Why This Method

**The GPU CWT keeps wavelet accuracy at real-time speed - the textbook options give up one or the other.**

![Fourier vs Wavelet - time per frame (log scale) and frequency resolution](timing_methods_v10.png)

![Compute per frame for each backend, chunk size, and resolution](timing_config_v10.png)

**Test parameters**

| Number of Samples | Sampling Rate | Lowest Frequency | Highest Frequency | Frequency Resolution | Number of Frequencies |
| --- | --- | --- | --- | --- | --- |
| 16,384 | 44.1 kHz | 27.5 Hz | 21.1 kHz | 12 per octave | 116 |
