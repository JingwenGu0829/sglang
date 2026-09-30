# Rollout contracts: initial multi-stream integration

A joint audio/video model needs both states to replay a video policy step. The
first integration captures that joint state while retaining a video-only policy:
audio follows its native deterministic transition. It does not yet implement
audio-policy likelihoods or a trainer-side replay adapter.

## Responsibilities

- `flow_transition.py` implements request-independent ODE/SDE/CPS math. Callers
  supply predictions, sigma pairs and random noise; it returns the next state,
  transition mean, noise scale, and per-sample score sum/count.
- `rollout_recorder.py` owns cloned snapshots and validates their shape, clocks,
  joint boundaries and lifecycle. It never calls a stage or scheduler. Each
  stream supplies its own gather operation; aborting performs no collectives.
- Model integrations resolve layouts, prediction conventions, schedules and
  updates. H3's adapter supplies `-velocity` to flow math and adds an explicit
  batch axis to packed rows. LTX uses separate audio/video schedulers and gather
  operations, including removal of sequence-parallel padding.
- `RolloutDenoisingMixin` and `SchedulerRLMixin` remain compatibility boundaries
  for existing requests, conditioning dictionaries, scores and debug outputs.
  The scheduler still stores its legacy session on the request. Moving that
  session and replacing conditioning dictionaries with family-specific schemas
  are separate migration steps; the recorder and H3 adapter do not depend on it.

## State and wire format

Internal named trajectories have shape `[B, K, ...]`. All streams are captured
before either stream updates at a joint boundary, plus an optional final state.
Each stream has:

- `latents`: owned snapshots at the retained boundaries;
- `latent_step_indices`: their original indices in `[0, N]`;
- `sigmas` and `timesteps`: full `N+1` schedules, even for sparse capture;
- `model_timesteps`: optional actual `N` model inputs when the model clock differs
  from the rollout clock (H3 uses `1 - sigma`, versus exported `sigma * 1000`).

Streams may have different tensor shapes and sigma schedules, but must share
`B` and the number of joint boundaries. Asynchronous modality schedules need an
explicit execution plan and are outside this initial contract.

The HTTP response adds `stream_trajectories["video"]` and, for H3/LTX,
`stream_trajectories["audio"]`. Responses are per sample, so serialized latents
have shape `[K, ...]`. The legacy `dit_trajectory` remains the video projection,
with **retained** timesteps and full sigmas, preserving its previous sparse
behavior. H3 conditioning is explicitly marked as a single packed sample so
one-token axes are not mistaken for batch axes during response slicing.

The existing `rollout_return_dit_trajectory` flag enables all available replay
streams. Scores and legacy debug tensors are still video-only. Capture does
not draw random numbers or change scheduler advancement.

## Supported boundaries and remaining work

- Generic flow schedulers retain their existing rollout behavior. Global noise
  shape is resolved before SP sharding; per-sample generators produce one full
  sample each. Score sum/count already represent the global state and must not
  be all-reduced again.
- Cosmos rollout supports its existing unconditioned video path. Serving UniPC
  keeps its native call signature. Conditioned/action/sound rollout is rejected.
- H3 preserves its native serving loop and native audio updates. Rollout requires
  one generator for the single packed sample. Generic decoded trajectory flags
  remain unsupported.
- LTX rollout currently requires single-stage Euler with standard CFG. Two-stage,
  res2s and custom guider paths are rejected because they bypass the scheduler
  transition adapter. Checkpoint inference, full training replay, and conditioned
  coordinate masks have not been validated by this iteration.

`rollout_log_prob_no_const=True` retains the existing negative squared residual
score; it is **not** a normalized log density. Deterministic steps have zero
legacy score, not a Gaussian density. A true-density request for a degenerate
stochastic transition fails instead of producing NaNs. Trainers must explicitly
choose scoring semantics; changing their reduction or silently adding an audio
score would change the optimization objective.

The next migration should introduce a validated per-stream transition plan,
family-specific replay conditioning, generated-coordinate masks, independent
policy RNG streams, and a train-side function that scores an observed next state.
Decoded audio transport and a Miles joint-state replay adapter are still needed
for audio rewards and audio-policy training.
