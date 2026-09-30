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
import json
import random
import time
import urllib.request

def parse_args(argv=None):
    """Parse command-line arguments. Returns (URL, MODEL, API_KEY, N_DECODE)."""
    parser = argparse.ArgumentParser(description="Benchmark TTFT and sustained decode throughput.")
    parser.add_argument("url", help="API endpoint URL")
    parser.add_argument("model", help="Model name")
    parser.add_argument("api_key", help="API key")
    parser.add_argument("n_decode", type=int, default=400, help="Number of tokens to decode (default: 400)")
    args = parser.parse_args(argv)
    return args.url, args.model, args.api_key, args.n_decode

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


_last_prompt = None


def random_prompt():
    """Pick a prompt, ensuring no two adjacent calls return the same one."""
    global _last_prompt
    if len(PROMPTS) == 1:
        return PROMPTS[0]
    choices = [p for p in PROMPTS if p != _last_prompt]
    _last_prompt = random.choice(choices)
    return _last_prompt


def count_tokens(text):
    """Rough token count: ~4 chars per token (gpt-4 style)."""
    return max(1, len(text) // 4)


def build_request_body(prompt):
    """Build the JSON request body for a streaming request."""
    return json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()


def build_headers():
    """Build request headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def stream_once(max_tokens):
    """Stream a single request. Returns (ttft, decoded_tokens, total_time, first_delta, last_delta)."""
    prompt = random_prompt()
    req = urllib.request.Request(URL, data=build_request_body(prompt), headers=build_headers())
    start = time.perf_counter()
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
    ct = ct_usage if ct_usage > 0 else ct_delta
    return ttft, ct, total, first_delta, last_delta


def calc_tps(n, dec):
    """Calculate tokens-per-second, returning NaN if decode time is invalid."""
    return n / dec if dec > 0 else float("nan")


def run_warmup():
    """Run 3 warmup requests to cold-load the model."""
    for _ in range(3):
        stream_once(8)


def measure_warm_ttft():
    """Run 3 short requests and return mean TTFT."""
    ttfts = []
    for _ in range(3):
        t, _, _, _, _ = stream_once(32)
        if t is not None:
            ttfts.append(t)
    return sum(ttfts) / len(ttfts) if ttfts else 0.0


def main():
    """Run the full benchmark and print results."""
    URL, MODEL, API_KEY, N_DECODE = parse_args()
    run_warmup()
    ttft_mean = measure_warm_ttft()

    ttft, n, total, first_delta, last_delta = stream_once(N_DECODE)
    if last_delta and first_delta is not None:
        dec = last_delta - first_delta
    else:
        dec = total - (ttft or 0)
    tps = calc_tps(n, dec)

    print(f"warm_ttft={ttft_mean:.3f}s tps={tps:.1f} tokens={n} total={total:.2f}s")


if __name__ == "__main__":
    main()
