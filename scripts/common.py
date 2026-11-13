"""Shared utilities for LLM benchmarking scripts."""

import json
import os
import time

import requests

# Magic numbers extracted to named constants
DEFAULT_TIMEOUT = 900
MAX_TOKENS = 400
WARMUP_REQUESTS = 3
WARMUP_TOKENS = 8
TTFT_MEASURE_TOKENS = 32
TTFT_MEASURE_REQUESTS = 3
RETRY_COUNT = 3


def count_tokens(text):
    """Rough token count: ~4 chars per token (gpt-4 style)."""
    return max(1, len(text) // 4)


def build_request_body(model, prompt, max_tokens=MAX_TOKENS):
    """Build the JSON request body for a streaming request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def build_headers(api_key):
    """Build request headers with optional auth."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def stream_once(max_tokens, url, model, api_key, prompt=None):
    """Stream a single request with retry logic.

    Returns (ttft, decoded_tokens, total_time, first_delta, last_delta).
    Raises RuntimeError after all retries are exhausted.
    """
    last_error = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            if prompt is None:
                prompt = random_prompt()
            start = time.perf_counter()
            ct_usage = 0
            ct_delta = 0
            first_delta = None
            last_delta = None

            with requests.post(url, data=json.dumps(build_request_body(model, prompt, max_tokens)), headers=build_headers(api_key), stream=True, timeout=DEFAULT_TIMEOUT) as resp:
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
                    if u and u.get("completion_tokens"):
                        ct_usage = u["completion_tokens"]

                    ch = c.get("choices") or []
                    if ch:
                        d = ch[0].get("delta") or {}
                        t = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
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
        except (requests.exceptions.RequestException) as e:
            last_error = e
            if attempt < RETRY_COUNT:
                wait = 2**attempt
                print(f"retry {attempt} in {wait}s after: {e}", flush=True)
                time.sleep(wait)

    raise RuntimeError(f"request failed after {RETRY_COUNT} attempts: {last_error}")


# Pool of prompts — no reuse, forces cold prefill every time
# Loaded from prompts.json, falls back to embedded defaults if missing
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts.json")

DEFAULT_PROMPTS = [
    "Write a detailed paragraph about why the sky is blue, covering Rayleigh scattering.",
    "Explain how photosynthesis works in plants, mentioning chloroplasts and light reactions.",
    "Describe the process of oceanic crust formation at mid-ocean ridges.",
    "Summarize the key events leading to the fall of the Roman Empire.",
    "Explain the basic principles of supply and demand in economics.",
    "Describe how a digital camera captures and processes an image.",
    "Explain the difference between supervised and unsupervised machine learning.",
    "Describe the structure and function of the human nervous system.",
]

try:
    with open(PROMPTS_PATH) as f:
        PROMPTS = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    PROMPTS = DEFAULT_PROMPTS

_last_prompt = None


def random_prompt():
    """Pick a prompt, ensuring no two adjacent calls return the same one."""
    global _last_prompt
    if len(PROMPTS) == 1:
        return PROMPTS[0]
    choices = [p for p in PROMPTS if p != _last_prompt]
    _last_prompt = __import__("random").choice(choices)
    return _last_prompt
