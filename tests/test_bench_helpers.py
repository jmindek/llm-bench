"""Unit tests for bench_stream and bench_pptg helper functions."""
import json
import sys
import textwrap
from unittest.mock import patch, mock_open

import pytest

# Import module-level constants and functions by loading the scripts as modules
sys.path.insert(0, "scripts")

import bench_stream
import bench_pptg


# ── bench_stream tests ──────────────────────────────────────────────────────


class TestRandomPrompt:
    def test_returns_from_pool(self):
        for _ in range(100):
            prompt = bench_stream.random_prompt()
            assert prompt in bench_stream.PROMPTS

    def test_no_adjacent_duplicates(self):
        prev = None
        for _ in range(200):
            prompt = bench_stream.random_prompt()
            if prev is not None:
                assert prompt != prev, "adjacent duplicate detected"
            prev = prompt


class TestCountTokens:
    def test_basic(self):
        assert bench_stream.count_tokens("hello world") == 2

    def test_empty(self):
        assert bench_stream.count_tokens("") == 1  # min(1, ...)

    def test_approx_4_chars_per_token(self):
        text = "a" * 40
        assert bench_stream.count_tokens(text) == 10


class TestCalcTps:
    def test_normal(self):
        assert bench_stream.calc_tps(400, 4.0) == 100.0

    def test_zero_decode(self):
        import math
        assert math.isnan(bench_stream.calc_tps(400, 0))

    def test_negative_decode(self):
        import math
        assert math.isnan(bench_stream.calc_tps(400, -1.0))


# ── bench_pptg tests ────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_filler(self):
        prompt = bench_pptg.build_prompt(1000)
        assert "Word. " in prompt

    def test_contains_seed(self):
        prompt = bench_pptg.build_prompt(1000)
        assert "-seed-" in prompt

    def test_approx_target_length(self):
        """Filler should be roughly ctx_tokens words."""
        prompt = bench_pptg.build_prompt(600)
        filler_count = prompt.count("Word. ")
        assert filler_count == 100  # 600 // 6


class TestCountTokensPptg:
    def test_basic(self):
        assert bench_pptg.count_tokens("hello world") == 2

    def test_empty(self):
        assert bench_pptg.count_tokens("") == 1


class TestCalcRates:
    def test_normal(self):
        pp, tg, dec = bench_pptg.calc_rates(
            ttft=1.0, pt=1000, ct=100,
            first_delta=1.0, last_delta=2.0, total=3.0
        )
        assert pp == 1000.0
        assert tg == 100.0
        assert dec == 1.0

    def test_no_delta_times(self):
        pp, tg, dec = bench_pptg.calc_rates(
            ttft=1.0, pt=1000, ct=100,
            first_delta=None, last_delta=None, total=3.0
        )
        assert pp == 1000.0
        assert tg == 50.0  # 100 / (3.0 - 1.0)
        assert dec == 2.0

    def test_zero_decode(self):
        pp, tg, dec = bench_pptg.calc_rates(
            ttft=1.0, pt=1000, ct=100,
            first_delta=1.0, last_delta=1.0, total=2.0
        )
        assert tg == 0.0

    def test_no_ttft(self):
        pp, tg, dec = bench_pptg.calc_rates(
            ttft=None, pt=1000, ct=100,
            first_delta=None, last_delta=None, total=3.0
        )
        assert pp == 0.0
