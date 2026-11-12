"""Integration tests for streaming functions with mocked HTTP."""

import json
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "scripts")

import bench_pptg
import bench_stream


def make_streaming_lines(tokens=400, usage=True):
    lines = []
    for _i in range(tokens):
        payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
        lines.append(f"data: {payload}\n")
    if usage:
        payload = json.dumps({"usage": {"completion_tokens": tokens}})
        lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


def make_pptg_streaming_lines(prompt_tokens=1000, completion_tokens=5):
    lines = []
    for _i in range(completion_tokens):
        payload = json.dumps({"choices": [{"delta": {"content": "w"}}]})
        lines.append(f"data: {payload}\n")
    payload = json.dumps({"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})
    lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    return lines


def mock_response(lines):
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=None)
    mock_resp.__iter__ = MagicMock(return_value=iter([x.encode("utf-8") for x in lines]))
    return mock_resp


@patch("bench_stream.urllib.request.Request")
@patch("bench_stream.urllib.request.urlopen")
def test_stream_once(mock_urlopen, mock_request):
    lines = make_streaming_lines(10)
    mock_urlopen.return_value = mock_response(lines)

    ttft, ct, _total, _first, _last = bench_stream.stream_once(10, "http://localhost", "test", "key")
    assert ttft is not None
    assert ct == 10


@patch("bench_stream.urllib.request.Request")
@patch("bench_stream.urllib.request.urlopen")
def test_stream_once_no_usage(mock_urlopen, mock_request):
    lines = make_streaming_lines(10, usage=False)
    mock_urlopen.return_value = mock_response(lines)

    _ttft, ct, _total, _first, _last = bench_stream.stream_once(10, "http://localhost", "test", "key")
    assert ct == 10


@patch("bench_stream.urllib.request.Request")
@patch("bench_stream.urllib.request.urlopen")
def test_stream_once_json_decode_error(mock_urlopen, mock_request):
    lines = [
        'data: {"choices": [{"delta": {"content": "x"}}]}\n',
        "data: {bad json}\n",
        "data: [DONE]\n",
    ]
    mock_urlopen.return_value = mock_response(lines)

    _ttft, ct, _total, _first, _last = bench_stream.stream_once(1, "http://localhost", "test", "key")
    assert ct == 1


@patch("bench_pptg.urllib.request.Request")
@patch("bench_pptg.urllib.request.urlopen")
def test_pptg_stream_once(mock_urlopen, mock_request):
    lines = make_pptg_streaming_lines(1000, 5)
    mock_urlopen.return_value = mock_response(lines)

    ttft, pt, ct, _total, _first, _last = bench_pptg.stream_once("test", 5, "http://localhost", "test", "key")
    assert ttft is not None
    assert pt == 1000
    assert ct == 5


@patch("bench_pptg.urllib.request.Request")
@patch("bench_pptg.urllib.request.urlopen")
def test_pptg_stream_once_no_usage(mock_urlopen, mock_request):
    lines = []
    for _i in range(5):
        payload = json.dumps({"choices": [{"delta": {"content": "w"}}]})
        lines.append(f"data: {payload}\n")
    lines.append("data: [DONE]\n")
    mock_urlopen.return_value = mock_response(lines)

    _ttft, pt, ct, _total, _first, _last = bench_pptg.stream_once("test", 5, "http://localhost", "test", "key")
    assert pt == 1
    assert ct == 5
