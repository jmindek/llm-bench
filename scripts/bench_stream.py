#!/usr/bin/env python3
"""Benchmark TTFT and sustained decode throughput (tps) for local LLM providers.

Measures from the CLIENT side — not the server — for accurate end-to-end metrics.

Usage:
    python3 bench_stream.py <URL> <MODEL> <API_KEY> [N_DECODE]

Examples:
    python3 bench_stream.py http://localhost:8000/v1/chat/completions qwen3.6-35b YOUR_API_KEY
    python3 bench_stream.py http://localhost:8080/v1/chat/completions qwen3.6-35b YOUR_API_KEY 512

Outputs:
    warm_ttft=<mean>s tps=<rate> tokens=<count> total=<time>s
"""
import json
import random
import sys
import time
import urllib.request

URL = sys.argv[1]
MODEL = sys.argv[2]
API_KEY = sys.argv[3]
N_DECODE = int(sys.argv[4]) if len(sys.argv) > 4 else 400

# Pool of prompts — no reuse, forces cold prefill every time
PROMPTS = [
    "Write a detailed paragraph about why the sky is blue, covering Rayleigh scattering.",
    "Explain how photosynthesis works in plants, mentioning chloroplasts and light reactions.",
    "Describe the process of oceanic crust formation at mid-ocean ridges.",
    "Summarize the key events leading to the fall of the Roman Empire.",
    "Explain the basic principles of supply and demand in economics.",
    "Describe how a digital camera captures and processes an image.",
    "Explain the difference between supervised and unsupervised machine learning.",
    "Describe the structure and function of the human nervous system.",
]


def random_prompt():
    """Pick a prompt, ensuring no two adjacent calls return the same one."""
    return random.choice(PROMPTS)


def count_tokens(text):
    """Rough token count: ~4 chars per token (gpt-4 style)."""
    return max(1, len(text) // 4)


def stream_once(max_tokens):
    """Stream a single request. Returns (ttft, decoded_tokens, total_time)."""
    prompt = random_prompt()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(URL, data=body, headers=headers)
    start = time.perf_counter()
    ttft = None
    ct_usage = 0
    ct_delta = 0
    first_delta = None
    last_delta = None

    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                c = json.loads(p)
            except json.JSONDecodeError:
                continue

            u = c.get("usage")
            if u and u.get("completion_tokens"):
                ct_usage = u["completion_tokens"]

            ch = c.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                t = (d.get("content") or d.get("reasoning_content")
                     or d.get("reasoning") or "")
                if t:
                    ct_delta += count_tokens(t)
                    now = time.perf_counter()
                    if first_delta is None:
                        first_delta = now
                    last_delta = now

    ttft = first_delta - start if first_delta else None
    total = time.perf_counter() - start

    # Prefer usage count; fall back to delta count
    ct = ct_usage if ct_usage > 0 else ct_delta
    return ttft, ct, total, first_delta, last_delta


# Warmup: cold load + verify model responds
for _ in range(3):
    stream_once(8)

# Warm TTFT: 3 short requests with unique prompts, take mean
ttfts = [stream_once(32)[0] for _ in range(3)]
ttft_mean = sum(t for t in ttfts if t is not None) / max(1, sum(1 for t in ttfts if t is not None))

# Sustained decode: single request
ttft, n, total, first_delta, last_delta = stream_once(N_DECODE)
if last_delta and first_delta is not None:
    dec = last_delta - first_delta
else:
    dec = total - (ttft or 0)
tps = n / dec if dec > 0 else float("nan")

print(f"warm_ttft={ttft_mean:.3f}s tps={tps:.1f} tokens={n} total={total:.2f}s")
