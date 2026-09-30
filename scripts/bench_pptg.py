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
import argparse
import json
import random
import time
import urllib.request

def parse_args(argv=None):
    """Parse command-line arguments. Returns (URL, MODEL, API_KEY, CTX)."""
    parser = argparse.ArgumentParser(description="Benchmark prefill and generation throughput.")
    parser.add_argument("url", help="API endpoint URL")
    parser.add_argument("model", help="Model name")
    parser.add_argument("api_key", help="API key")
    parser.add_argument("ctx", type=int, help="Target context length in tokens")
    args = parser.parse_args(argv)
    return args.url, args.model, args.api_key, args.ctx


def build_prompt(ctx_tokens):
    """Build a filler prompt approximating ctx_tokens, with random suffix to defeat cache."""
    filler = "Word. " * (ctx_tokens // 6)
    suffix = f"-seed-{random.randint(0, 100000)}"
    return f"Here is a long context: {filler}{suffix}"


def count_tokens(text):
    """Rough token count: ~4 chars per token (gpt-4 style)."""
    return max(1, len(text) // 4)


def build_request_body(model, prompt):
    """Build the JSON request body for a streaming request."""
    return json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()


def build_headers(api_key):
    """Build request headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def stream_once(prompt, max_tokens, url, model, api_key):
    """Stream a single request. Returns (ttft, prompt_tokens, completion_tokens,
    total_time, first_delta, last_delta)."""
    req = urllib.request.Request(url, data=build_request_body(model, prompt), headers=build_headers(api_key))
    start = time.perf_counter()
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
    ct = ct_usage if ct_usage > 0 else ct_delta
    pt = pt_usage if pt_usage > 0 else CTX
    return ttft, pt, ct, total, first_delta, last_delta


def calc_rates(ttft, pt, ct, first_delta, last_delta, total):
    """Calculate prefill and generation throughput rates.
    
    Returns (pp, tg, decode_time).
    """
    pp = pt / ttft if ttft and ttft > 0 else 0.0
    if last_delta and first_delta is not None:
        dec = last_delta - first_delta
    else:
        dec = total - (ttft or 0)
    tg = ct / dec if dec > 0 else 0.0
    return pp, tg, dec


def main():
    """Run the benchmark and print results."""
    URL, MODEL, API_KEY, CTX = parse_args()
    prompt = build_prompt(CTX)
    ttft, pt, ct, total, first_delta, last_delta = stream_once(prompt, 400, URL, MODEL, API_KEY)
    pp, tg, dec = calc_rates(ttft, pt, ct, first_delta, last_delta, total)

    print(f"pp={pp:.0f} tok/s tg={tg:.0f} tok/s prompt_tokens={pt} completion_tokens={ct} total={total:.2f}s")


if __name__ == "__main__":
    main()
