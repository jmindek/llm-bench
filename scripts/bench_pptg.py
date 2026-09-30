#!/usr/bin/env python3
"""Benchmark prefill (pp) and generation (tg) throughput at a given context length.

Measures from the CLIENT side — not the server — for accurate end-to-end metrics.

Usage:
    python3 bench_pptg.py <URL> <MODEL> <API_KEY> <CTX_TOKENS>

Examples:
    python3 bench_pptg.py http://localhost:8000/v1/chat/completions qwen3.6-35b YOUR_API_KEY 16000
    python3 bench_pptg.py http://localhost:8080/v1/chat/completions qwen3.6-35b YOUR_API_KEY 32000

Outputs:
    pp=<rate> tok/s tg=<rate> tok/s prompt_tokens=<count> completion_tokens=<count> total=<time>s

IMPORTANT: Uses a FRESH prompt each cold run to defeat prefix caches.
A cache hit shows ttft ~0.4s and pp inflated to 20k+ tok/s — that's NOT real prefill.
"""
import json
import random
import sys
import time
import urllib.request

URL = sys.argv[1]
MODEL = sys.argv[2]
API_KEY = sys.argv[3]
CTX = int(sys.argv[4])


def build_prompt(ctx_tokens):
    """Build a filler prompt approximating ctx_tokens, with random suffix to defeat cache."""
    filler = "Word. " * (ctx_tokens // 6)
    suffix = f"-seed-{random.randint(0, 100000)}"
    return f"Here is a long context: {filler}{suffix}"


def count_tokens(text):
    """Rough token count: ~4 chars per token (gpt-4 style)."""
    return max(1, len(text) // 4)


def stream_once(prompt, max_tokens):
    """Stream a single request. Returns (ttft, prompt_tokens, completion_tokens,
    total_time, first_delta, last_delta)."""
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
    pt_usage = 0
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
            if u:
                if u.get("prompt_tokens"):
                    pt_usage = u["prompt_tokens"]
                if u.get("completion_tokens"):
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

    # Prefer usage counts; ct falls back to delta count
    # pt: use CTX as estimate when usage chunk is missing
    ct = ct_usage if ct_usage > 0 else ct_delta
    pt = pt_usage if pt_usage > 0 else CTX
    return ttft, pt, ct, total, first_delta, last_delta


# Cold run: fresh prompt to defeat prefix cache
prompt = build_prompt(CTX)
ttft, pt, ct, total, first_delta, last_delta = stream_once(prompt, 400)

# Calculate rates
pp = pt / ttft if ttft and ttft > 0 else 0
if last_delta and first_delta is not None:
    dec = last_delta - first_delta
else:
    dec = total - (ttft or 0)
tg = ct / dec if dec > 0 else 0

print(f"pp={pp:.0f} tok/s tg={tg:.0f} tok/s prompt_tokens={pt} completion_tokens={ct} total={total:.2f}s")
