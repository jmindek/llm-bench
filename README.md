# llm-bench

Client-side benchmark scripts for local LLM providers (omlx, OptiQ, etc.).

## Why client-side?

Server-reported metrics are unreliable:
- omlx's per-request `tok/s` falls as generations get longer (fixed per-request cost)
- Server logs under-report vs wall-clock
- Network and harness overhead is excluded

**Always measure from the client.**

## Scripts

### `bench_stream.py` — TTFT + sustained decode throughput

Measures warm TTFT (mean of 3 short requests) and sustained tokens-per-second.

```bash
python3 scripts/bench_stream.py <URL> <MODEL> <API_KEY> [N_DECODE]
```

### `bench_pptg.py` — prefill + generation at context length

Measures prefill (pp) and generation (tg) rates at a given context length. Uses a fresh prompt each run to defeat prefix caches.

```bash
python3 scripts/bench_pptg.py <URL> <MODEL> <API_KEY> <CTX_TOKENS>
```

## Key Metrics

| Term | Meaning |
|------|---------|
| TTFT | Time To First Token |
| tps | decode tokens/sec (sustained) |
| pp | prefill tokens/sec (prompt processing) |
| tg | generation tokens/sec (decode) |

## Pitfalls

- **Cache hits skew numbers**: identical prompts reuse prefix cache → inflated pp. Vary prompts between cold runs.
- **MTP warm-up**: first ~30 tokens run slower. Use ≥400-token generations.
- **Thermal throttling**: back-to-back long runs throttle GPU on all hardware. Cooldown or interleave.
- **ANE prefill** (Apple Silicon only): ~27s overhead on first load (compiles fixed-shape programs). Load-time only, not TTFT.
