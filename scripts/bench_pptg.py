#!/usr/bin/env python3
"""Benchmark prefill (pp) and generation (tg) throughput at a given context length.

Measures from the CLIENT side — not the server — for accurate end-to-end metrics.

Usage:
    python3 bench_pptg.py <URL> <MODEL> <API_KEY> <CTX_TOKENS>

Examples:
    python3 bench_pptg.py http://localhost:8000/v1/chat/completions qwen3.6-35b sk-555-omlx 16000
    python3 bench_pptg.py http://localhost:8080/v1/chat/completions qwen3.6-35b sk-optiq-xxx 32000

Outputs:
    pp=<rate> tok/s tg=<rate> tok/s tokens=<count> total=<time>s

IMPORTANT: Uses a FRESH prompt each cold run to defeat prefix caches.
A cache hit shows ttft ~0.4s and pp inflated to 20k+ tok/s — that's NOT real prefill.
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1]
MODEL = sys.argv[2]
API_KEY = sys.argv[3]
CTX = int(sys.argv[4])

# Build a filler prompt that approximates the target token count
# ~6 tokens per "Word. " unit
filler = "Word. " * (CTX // 6)
PROMPT = f"Here is a long context: {filler}"


def stream_once(prompt, max_tokens):
    """Stream a single request and return (ttft, prompt_tokens, completion_tokens, total_time)."""
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
    pt = 0
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
            if u:
                if u.get("prompt_tokens"):
                    pt = u["prompt_tokens"]
                if u.get("completion_tokens"):
                    ct = u["completion_tokens"]

            ch = c.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                t = (d.get("content") or d.get("reasoning_content")
                     or d.get("reasoning") or "")
                if t and ttft is None:
                    ttft = time.perf_counter() - start

    total = time.perf_counter() - start
    return ttft, pt, ct, total


# Cold run: fresh prompt to defeat prefix cache
ttft, pt, ct, total = stream_once(PROMPT, 400)

# Calculate rates
pp = pt / ttft if ttft and ttft > 0 else 0
dec = total - (ttft or 0)
tg = ct / dec if dec > 0 else 0

print(f"pp={pp:.0f} tok/s tg={tg:.0f} tok/s prompt_tokens={pt} completion_tokens={ct} total={total:.2f}s")
