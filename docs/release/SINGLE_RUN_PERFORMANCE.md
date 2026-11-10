# Single-episode runtime profile

This local development branch builds on `experiment/r5-control-revalidation`.
It targets the latency of **one experiment**. Existing multi-GPU CLI parallelism
is unchanged. It is not a new paper baseline, and no GPU speedup is claimed from
the CPU checks below.

## Runtime modes

The existing GPT task, batch and supervised bottle launchers use the profile.
Select the mode in the launching shell; the child process inherits it:

```sh
export GPT_BENCHMARK_RUNTIME_MODE=optimized
```

| Mode | Behavior |
| --- | --- |
| `optimized` (branch default) | Timings, background video, scoped joint read reuse, identical image encoding cache |
| `timing` | Timings with synchronous recording and uncached reads/encoding, for comparison |
| `off` | The unchanged episode runtime from the parent control-validation branch |

Dry runs and caller-supplied non-GPT policies retain their existing dispatch.
Each experiment runs in its own process. A new batch records its mode and
preserves it when resumed; changing a shell variable cannot silently switch the
mode of an existing batch. Use a fresh output directory for each comparison.

## What changes

- **Timing:** `timing.json` and `result.json.runtime_performance` record runner
  startup, model decision/transport, image preparation, planning, physics stepping,
  controller callbacks, render capture, robot reads, evidence writes and video
  processing. Intermediate checkpoints retain completed measurements on errors.
  Model transport includes bridge/network/provider wait; it is not pure inference
  time. CLI process timing is available only for a directly hosted CLI client.
- **Video:** one encoder thread per camera receives owned RGB bytes through a
  four-frame queue. Every accepted frame retains its order. When the queue is
  full the producer waits; frames are never dropped. MP4, preview sampling,
  first/latest images and GIF contents use the original recorder. GIF finalization
  runs in the encoder thread on close. The runner waits for all encoders before
  reporting completion; encoding errors invalidate the run as recording errors.
- **Joint reads:** the native R5 `before_physics_step` callback may read the same
  articulation positions several times. Reuse those arrays only inside that one
  callback. Return independent copies; clear on callback exit/error. Physics,
  settling samples, collision checks, feedback and controller targets retain
  their original frequency and code. Reads outside the callback always hit Isaac.
- **Image encoding:** repeated identical RGB observations can reuse prepared PNG
  bytes. Keys include pixels, dimensions, calibration, robot state, annotation
  geometry and resize setting. The episode-local cache is bounded to six entries
  and 32 MiB of retained encoded payload/metadata. Timestamp labels, full history,
  depth evidence, schema, prompts and model settings are still built normally.

No simulator method runs on the encoder threads. Camera renders are deliberately
not cached: freshness requires a separate RTX check. The model process/session,
reasoning effort, history length, image resolution, physics rate, control profile,
scoring, and the 150-generation / 18,000-second defaults are unchanged. Faster
execution can still allow more decisions before the existing wall deadline.

## Inspect and compare

Offline profile inspection (no simulator or model):

```sh
python tools/control_revalidation/run.py check \
  --experiment-config configs/experiments/l3_01_preloosen_270.json
```

After an explicitly launched experiment:

```sh
python tools/runtime_timing.py /path/to/episode
```

Compare `timing` and `optimized` using the same task/configuration, seed,
hardware and recorded action sequence before drawing performance conclusions.
Keep live model runs separate because model latency and decisions vary.
Inclusive timers nest; exclusive timers subtract same-thread children. Worker
threads overlap, so summing phases does **not** give episode wall time. Video
enqueue/drain time measures waiting on the caller; encoder time measures work on
the worker. Timing starts at runner entry, excludes earlier model/client assembly
and ends before final backend teardown. An external startup-inclusive supervisor
budget continues to apply unchanged.

Offline regressions compare the full outgoing model payload, episode events and
results across all modes, and verify video ordering, GIF/image equivalence,
bounded queues/caches, per-step invalidation, error propagation and hook cleanup.
These checks establish implementation behavior, not physical task success or a
measured end-to-end GPU speedup.
