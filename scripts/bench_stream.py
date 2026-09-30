#!/usr/bin/env python3
"""Benchmark TTFT and sustained decode throughput (tps) for local LLM providers.

Measures from the CLIENT side — not the server — for accurate end-to-end metrics.

Usage:
    python3 bench_stream.py <URL> <MODEL> <API_KEY> [N_DECODE]

Examples:
    python3 bench_stream.py http://localhost:8000/v1/chat/completions qwen3.6-35b sk-555-omlx
    python3 bench_stream.py http://localhost:8080/v1/chat/completions qwen3.6-35b sk-optiq-xxx 512

Outputs:
    warm_ttft=<mean>s tps=<rate> tokens=<count> total=<time>s
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1]
MODEL = sys.argv[2]
API_KEY = sys.argv[3]
N_DECODE = int(sys.argv[4]) if len(sys.argv) > 4 else 400

PROMPT = "Write a detailed paragraph about why the sky is blue, covering Rayleigh scattering."


def stream_once(max_tokens):
    """Stream a single request and return (ttft, completion_tokens, total_time)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
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
    ct = 0

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
                ct = u["completion_tokens"]

            ch = c.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                t = (d.get("content") or d.get("reasoning_content")
                     or d.get("reasoning") or "")
                if t and ttft is None:
                    ttft = time.perf_counter() - start

    total = time.perf_counter() - start
    return ttft, ct, total


# Warmup: cold load + verify model is loaded
for _ in range(3):
    stream_once(8)

# Warm TTFT: 3 short requests, take mean
ttfts = [stream_once(32)[0] for _ in range(3)]
ttft_mean = sum(ttfts) / len(ttfts)

# Sustained decode: measure tps
ttft, n, total = stream_once(N_DECODE)
dec = total - (ttft or 0)
tps = n / dec if dec > 0 else float("nan")

print(f"warm_ttft={ttft_mean:.3f}s tps={tps:.1f} tokens={n} total={total:.2f}s")
