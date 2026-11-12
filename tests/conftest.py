"""Shared test fixtures."""

import json
from unittest.mock import MagicMock


def mock_response_factory():
    """Factory for mock HTTP responses."""

    def make(lines):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=None)
        mock_resp.iter_lines = MagicMock(return_value=iter([x.encode("utf-8") for x in lines]))
        return mock_resp

    return make