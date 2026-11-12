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

import argparse
import sys

sys.path.insert(0, "scripts")
import common


def parse_args(argv=None):
    """Parse command-line arguments. Returns (URL, MODEL, API_KEY, N_DECODE)."""
    parser = argparse.ArgumentParser(description="Benchmark TTFT and sustained decode throughput.")
    parser.add_argument("url", help="API endpoint URL")
    parser.add_argument("model", help="Model name")
    parser.add_argument("api_key", help="API key")
    parser.add_argument("n_decode", type=int, default=common.MAX_TOKENS, help="Number of tokens to decode (default: 400)")
    args = parser.parse_args(argv)
    return args.url, args.model, args.api_key, args.n_decode


def calc_tps(n, dec):
    """Calculate tokens-per-second, returning NaN if decode time is invalid."""
    return n / dec if dec > 0 else float("nan")


def run_warmup(url, model, api_key):
    """Run 3 warmup requests to cold-load the model."""
    for _ in range(common.WARMUP_REQUESTS):
        common.stream_once(common.WARMUP_TOKENS, url, model, api_key)


def measure_warm_ttft(url, model, api_key):
    """Run 3 short requests and return mean TTFT."""
    ttfts = []
    for _ in range(common.TTFT_MEASURE_REQUESTS):
        t, _, _, _, _ = common.stream_once(common.TTFT_MEASURE_TOKENS, url, model, api_key)
        if t is not None:
            ttfts.append(t)
    return sum(ttfts) / len(ttfts) if ttfts else 0.0


def main():
    """Run the full benchmark and print results."""
    URL, MODEL, API_KEY, N_DECODE = parse_args()
    run_warmup(URL, MODEL, API_KEY)
    ttft_mean = measure_warm_ttft(URL, MODEL, API_KEY)

    ttft, n, total, first_delta, last_delta = common.stream_once(N_DECODE, URL, MODEL, API_KEY)
    if last_delta and first_delta is not None:
        dec = last_delta - first_delta
    else:
        dec = total - (ttft or 0)
    tps = calc_tps(n, dec)

    print(f"warm_ttft={ttft_mean:.3f}s tps={tps:.1f} tokens={n} total={total:.2f}s")


if __name__ == "__main__":
    main()