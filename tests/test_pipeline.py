"""Tests for pipeline metrics extraction."""

import csv
import sys

import pytest

sys.path.insert(0, "scripts")

import extract_pipeline_metrics as epm


def test_expand_env_refs_no_refs():
    env = {"FOO": "bar", "BAZ": "qux"}
    result = epm.expand_env_refs(env)
    assert result == env


def test_expand_env_refs_simple():
    env = {"FOO": "bar", "MESSAGE": "Hello ${FOO}"}
    result = epm.expand_env_refs(env)
    assert result["MESSAGE"] == "Hello bar"


def test_expand_env_refs_nested():
    """Single-pass expansion doesn't resolve nested refs."""
    env = {"HOST": "localhost", "PORT": "5432", "URL": "postgresql://${HOST}:${PORT}/db"}
    result = epm.expand_env_refs(env)
    assert result["URL"] == "postgresql://localhost:${PORT}/db"


def test_expand_env_refs_missing_ref():
    env = {"MESSAGE": "Hello ${MISSING}"}
    result = epm.expand_env_refs(env)
    assert result["MESSAGE"] == "Hello ${MISSING}"


def test_load_env_missing_file():
    result = epm.load_env("/nonexistent/path/.env")
    assert result == {}


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("FOO=bar\nBAZ=qux\n# comment\n")
    return path


def test_load_env_simple(env_file):
    result = epm.load_env(str(env_file))
    assert result["FOO"] == "bar"
    assert result["BAZ"] == "qux"


def test_load_env_with_quotes(env_file):
    path = env_file
    path.write_text('FOO="bar"\nBAZ=\'qux\'\n')
    result = epm.load_env(str(path))
    assert result["FOO"] == "bar"
    assert result["BAZ"] == "qux"


def test_sum_usage_basic():
    usages = [
        {"usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0, "totalTokens": 15}},
        {"usage": {"input": 20, "output": 10, "cacheRead": 5, "cacheWrite": 0, "reasoning": 0, "totalTokens": 35}},
    ]
    result = epm.sum_usage(usages)
    assert result["input"] == 30
    assert result["output"] == 15
    assert result["cacheRead"] == 5
    assert result["totalTokens"] == 50


def test_sum_usage_empty():
    result = epm.sum_usage([])
    assert result["input"] == 0
    assert result["totalTokens"] == 0


def test_agent_models_of():
    records = [
        {"model": "gpt-4", "responseId": "1"},
        {"model": "gpt-4o", "responseId": "2"},
        {"model": "gpt-4", "responseId": "3"},
    ]
    result = epm.agent_models_of(records)
    assert result == ["gpt-4", "gpt-4o"]


def test_agent_models_of_empty():
    result = epm.agent_models_of([])
    assert result == []


def test_ttft_metrics_empty_rows():
    result = epm.ttft_metrics([], {"id1"})
    assert result["ttft_s"] == ""
    assert result["ptps"] == ""
    assert result["gtps"] == ""


def test_ttft_metrics_no_match():
    rows = [{"rid": "id2", "cstart": 10.0, "start": 5.0, "dur": 1000, "pt": 100, "ct": 10}]
    result = epm.ttft_metrics(rows, {"id1"})
    assert result["ttft_s"] == ""


def test_ttft_metrics_basic():
    rows = [
        {"rid": "id1", "cstart": 10.5, "start": 5.0, "dur": 20000, "pt": 100, "ct": 10},
    ]
    result = epm.ttft_metrics(rows, {"id1"})
    assert result["ttft_s"] == 5.5
    assert result["ptps"] == pytest.approx(18.2, rel=0.1)
    assert result["gtps"] == pytest.approx(0.69, rel=0.01)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "benchmarks.csv"
    return path


def test_next_run_id_no_csv(csv_file):
    result = epm.next_run_id(str(csv_file))
    assert result == "bench-001"


def test_next_run_id_existing(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id"])
        writer.writeheader()
        writer.writerow({"run_id": "bench-001"})
        writer.writerow({"run_id": "bench-003"})
    result = epm.next_run_id(str(csv_file))
    assert result == "bench-004"


def test_pg_connect_raises_on_missing_credentials():
    """pg_connect exits with error if credentials are missing."""
    with pytest.raises(SystemExit):
        epm.pg_connect({})
