# Generic rollout contracts

The generic denoising loop uses explicit stream specifications and an owned,
loop-local recorder. Existing video rollout requests keep their legacy response
format and scoring convention.

## Responsibilities

- `flow_transition.py` contains request-independent ODE/SDE/CPS math. Callers
  supply predictions, sigma pairs and noise; the result contains the next state,
  transition mean, noise scale, and per-sample score sum/count.
- `rollout_recorder.py` owns cloned snapshots, validates shapes and schedules,
  and gathers retained boundaries. Stream specifications are generic; the
  serving integration currently records video. Capture never draws noise or
  advances a scheduler, and aborting performs no collectives.
- `RolloutDenoisingMixin` and `SchedulerRLMixin` bridge the existing request,
  scheduler session, conditioning and debug interfaces. The existing Cosmos
  collection hooks remain until its custom loop is migrated in the next layer.

## State and compatibility

Internal trajectories use `[B, K, ...]` and retain their original boundary
indices. Named streams carry full `N+1` schedules. The legacy video projection
keeps retained timesteps and full sigmas, including for sparse capture.

Global rollout noise is prepared before SP sharding. Per-sample generators draw
one full sample each; score sum/count already represent that full state and
must not be all-reduced again. The existing negative squared residual score is
preserved; it is not a normalized Gaussian log density.

Model-specific video adapters and the public audio/video stream response are
subsequent layers. This layer adds neither an audio policy nor trainer replay.
