# Model-specific video rollout

Generic ODE/SDE/CPS transitions and owned trajectory recording are now used by
the H3, Cosmos3 and LTX custom denoising loops. The HTTP interface exposes the
legacy video trajectory, per-step scores, and conditioning snapshot.

## Model adapters

- H3 uses a request-local adapter instead of a temporary scheduler. It maps
  native velocity to `-v`, adds a batch axis to packed rows, and owns its video
  snapshots before each in-place update. Native audio updates still execute.
- Cosmos3 uses the same recorder and drops the legacy mixin collection hooks.
  It retains video-only rollout admission; serving UniPC keeps its native call
  signature. Rollout resources are released even when denoising fails.
- LTX supports single-stage Euler with standard CFG. Its video gather removes
  SP padding, while its existing audio scheduler continues deterministic
  updates. Two-stage, res2s and custom guider rollout remain unsupported.

## Output contract

Internal named trajectories use `[B, K, ...]`. The legacy `dit_trajectory`
response retains sparse timesteps and full sigmas. It also exports the original
boundary indices and optional model timesteps; H3's model clock is `1 - sigma`
while its rollout clock is `sigma * 1000`. Packed H3 conditioning is explicitly
marked as a single sample so token axes survive response slicing.

Only video is recorded in model integrations at this layer. H3/LTX still need
their native audio computation during sampling; joint audio/video capture and
the public `stream_trajectories` response are introduced by the next layer.

## Remaining boundaries

Scheduler session state still lives on the request. Conditioning dictionaries
and legacy video scores remain compatibility boundaries. The negative squared
residual score is not a normalized Gaussian likelihood. Audio policy scoring,
decoded audio transport, conditioned-coordinate masks and trainer replay are
separate work. Full checkpoint inference and training replay are not validated
by the focused unit and sequence-parallel tests.
