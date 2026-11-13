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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
import requests


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


def stream_once(prompt, max_tokens, url, model, api_key):
    """Stream a single request with prompt. Returns (ttft, prompt_tokens, completion_tokens,
    total_time, first_delta, last_delta)."""
    last_error = None
    for attempt in range(1, common.RETRY_COUNT + 1):
        try:
            body = common.build_request_body(model, prompt, max_tokens)
            headers = common.build_headers(api_key)
            start = time.perf_counter()
            pt_usage = 0
            ct_usage = 0
            ct_delta = 0
            first_delta = None
            last_delta = None

            with requests.post(url, data=json.dumps(body), headers=headers, stream=True, timeout=common.DEFAULT_TIMEOUT) as resp:
                for raw in resp.iter_lines():
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
                        t = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                        if t:
                            ct_delta += common.count_tokens(t)
                            now = time.perf_counter()
                            if first_delta is None:
                                first_delta = now
                            last_delta = now

            ttft = first_delta - start if first_delta else None
            total = time.perf_counter() - start
            ct = ct_usage if ct_usage > 0 else ct_delta
            pt = pt_usage if pt_usage > 0 else common.count_tokens(prompt)
            return ttft, pt, ct, total, first_delta, last_delta
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < common.RETRY_COUNT:
                wait = 2**attempt
                print(f"retry {attempt} in {wait}s after: {e}", flush=True)
                time.sleep(wait)

    raise RuntimeError(f"request failed after {common.RETRY_COUNT} attempts: {last_error}")


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
    ttft, pt, ct, total, first_delta, last_delta = stream_once(prompt, common.MAX_TOKENS, URL, MODEL, API_KEY)
    pp, tg, _dec = calc_rates(ttft, pt, ct, first_delta, last_delta, total)

    print(f"pp={pp:.0f} tok/s tg={tg:.0f} tok/s prompt_tokens={pt} completion_tokens={ct} total={total:.2f}s")


if __name__ == "__main__":
    main()
