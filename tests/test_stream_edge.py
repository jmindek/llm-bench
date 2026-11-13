"""Tests for streaming edge cases and error handling."""

import sys
from unittest import mock
from unittest.mock import patch

import pytest
import requests

sys.path.insert(0, "scripts")

import common


def mock_response_factory():
    """Factory for mock HTTP responses."""
    def make(lines):
        resp = mock.MagicMock()
        resp.__enter__ = mock.MagicMock(return_value=resp)
        resp.__exit__ = mock.MagicMock(return_value=None)
        resp.iter_lines = mock.MagicMock(return_value=iter([x.encode("utf-8") for x in lines]))
        return resp
    return make


@patch("common.requests.post")
def test_stream_retry_on_connection_error(mock_post):
    """Test that stream_once retries on connection errors."""
    mock_post.side_effect = [
        requests.exceptions.ConnectionError("Connection refused"),
        requests.exceptions.ConnectionError("Connection refused"),
        mock_response_factory()([
            'data: {"choices": [{"delta": {"content": "x"}}]}\n',
            'data: [DONE]\n',
        ]),
    ]
    # Should succeed on 3rd attempt
    _ttft, _pt, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ct == 1


@patch("common.requests.post")
def test_stream_fails_after_max_retries(mock_post):
    """Test that stream_once raises RuntimeError after max retries."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
    with pytest.raises(RuntimeError) as exc:
        common.stream_once(10, "http://localhost", "test", "key")
    assert "request failed after 3 attempts" in str(exc.value)


@patch("common.requests.post")
def test_stream_empty_response(mock_post):
    """Test stream_once with empty response."""
    mock_post.return_value = mock_response_factory()([])
    _ttft, _pt, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ct == 0  # No content means no tokens


@patch("common.requests.post")
def test_stream_no_content_field(mock_post):
    """Test stream_once with no content field in delta."""
    lines = [
        'data: {"choices": [{"delta": {}}]}\n',
        'data: [DONE]\n',
    ]
    mock_post.return_value = mock_response_factory()(lines)
    _ttft, _pt, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ct == 0  # No content means no tokens


@patch("common.requests.post")
def test_stream_with_reasoning_content(mock_post):
    """Test stream_once with reasoning_content field."""
    lines = [
        'data: {"choices": [{"delta": {"reasoning_content": "Thinking... "}}]}\n',
        'data: {"choices": [{"delta": {"content": "Answer"}}]}\n',
        'data: [DONE]\n',
    ]
    mock_post.return_value = mock_response_factory()(lines)
    _ttft, _pt, ct, _total, _first, _last = common.stream_once(10, "http://localhost", "test", "key")
    assert ct == 4  # Two tokens from reasoning, two from content


@patch("common.requests.post")
def test_pptg_stream_no_prompt_tokens(mock_post):
    """Test pptg stream without prompt tokens in usage."""
    lines = [
        'data: {"choices": [{"delta": {"content": "w"}}]}\n',
        'data: {"usage": {"completion_tokens": 1}}\n',
        'data: [DONE]\n',
    ]
    mock_post.return_value = mock_response_factory()(lines)
    _ttft, pt, ct, _total, _first, _last = common.stream_once(5, "http://localhost", "test", "key", "test prompt")
    assert ct == 1
    assert pt > 0  # Falls back to count_tokens


@patch("common.requests.post")
def test_pptg_stream_zero_completion_tokens(mock_post):
    """Test pptg stream with zero completion tokens."""
    lines = [
        'data: {"usage": {"prompt_tokens": 1000, "completion_tokens": 0}}\n',
        'data: [DONE]\n',
    ]
    mock_post.return_value = mock_response_factory()(lines)
    _ttft, pt, ct, _total, _first, _last = common.stream_once(5, "http://localhost", "test", "key", "test prompt")
    assert pt == 1000
    assert ct == 0


def test_build_request_body():
    """Test build_request_body produces valid JSON."""
    body = common.build_request_body("gpt-4", "test prompt", 100)
    assert body["model"] == "gpt-4"
    assert body["max_tokens"] == 100
    assert body["stream"] is True


def test_build_request_body_default_max_tokens():
    """Test build_request_body uses default max tokens."""
    body = common.build_request_body("gpt-4", "test prompt")
    assert body["max_tokens"] == common.MAX_TOKENS


def test_build_headers_no_key():
    """Test build_headers without API key."""
    headers = common.build_headers(None)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_build_headers_with_key():
    """Test build_headers with API key."""
    headers = common.build_headers("secret")
    assert headers["Authorization"] == "Bearer secret"
