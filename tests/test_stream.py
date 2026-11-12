"""Integration tests for streaming functions."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "scripts")

import bench_pptg
import common


@pytest.fixture
def stream_lines():
    """Fixture for standard streaming response lines."""
    lines = []
    for _i in range(10):
        payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
        lines.append(f"data: {payload}\n")
    payload = json.dumps({"usage": {"completion_tokens": 10}})
    lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


@pytest.fixture
def stream_lines_no_usage():
    """Fixture for streaming response without usage block."""
    lines = []
    for _i in range(10):
        payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
        lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


@pytest.fixture
def stream_lines_bad_json():
    """Fixture for streaming response with bad JSON line."""
    return [
        'data: {"choices": [{"delta": {"content": "x"}}]}\n',
        "data: {bad json}\n",
        "data: [DONE]\n",
    ]


@pytest.fixture
def pptg_stream_lines():
    """Fixture for PPTG streaming response lines."""
    lines = []
    for _i in range(5):
        payload = json.dumps({"choices": [{"delta": {"content": "w"}}]})
        lines.append(f"data: {payload}\n")
    payload = json.dumps({"usage": {"prompt_tokens": 1000, "completion_tokens": 5}})
    lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


@pytest.fixture
def pptg_stream_lines_no_usage():
    """Fixture for PPTG streaming response without usage."""
    lines = []
    for _i in range(5):
        payload = json.dumps({"choices": [{"delta": {"content": "w"}}]})
        lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


@pytest.fixture
def mock_response_factory():
    """Factory for mock HTTP responses."""

    def make(lines):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)
        mock_resp.iter_lines = MagicMock(return_value=iter([x.encode("utf-8") for x in lines]))
        return mock_resp

    return make


@patch("common.requests.post")
def test_stream_once(mock_post, stream_lines, mock_response_factory):
    mock_post.return_value = mock_response_factory(stream_lines)
    ttft, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ttft is not None
    assert ct == 10


@patch("common.requests.post")
def test_stream_once_no_usage(mock_post, stream_lines_no_usage, mock_response_factory):
    mock_post.return_value = mock_response_factory(stream_lines_no_usage)
    _ttft, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ct == 10


@patch("common.requests.post")
def test_stream_once_json_decode_error(mock_post, stream_lines_bad_json, mock_response_factory):
    mock_post.return_value = mock_response_factory(stream_lines_bad_json)
    _ttft, ct, _total, _first, _last = common.stream_once(1, "http://localhost", "test", "key")
    assert ct == 1


@patch("bench_pptg.requests.post")
def test_pptg_stream_once(mock_post, pptg_stream_lines, mock_response_factory):
    mock_post.return_value = mock_response_factory(pptg_stream_lines)
    ttft, pt, ct, _total, _first, _last = bench_pptg.stream_once("test", 5, "http://localhost", "test", "key")
    assert ttft is not None
    assert pt == 1000
    assert ct == 5


@patch("bench_pptg.requests.post")
def test_pptg_stream_once_no_usage(mock_post, pptg_stream_lines_no_usage, mock_response_factory):
    mock_post.return_value = mock_response_factory(pptg_stream_lines_no_usage)
    _ttft, pt, ct, _total, _first, _last = bench_pptg.stream_once("test", 5, "http://localhost", "test", "key")
    assert pt == 1
    assert ct == 5