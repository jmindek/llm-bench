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
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common


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
    ttft, pt, ct, total, first_delta, last_delta = common.stream_once(common.MAX_TOKENS, URL, MODEL, API_KEY, prompt)
    pp, tg, _dec = calc_rates(ttft, pt, ct, first_delta, last_delta, total)

    print(f"pp={pp:.0f} tok/s tg={tg:.0f} tok/s prompt_tokens={pt} completion_tokens={ct} total={total:.2f}s")


if __name__ == "__main__":
    main()
