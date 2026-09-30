#!/usr/bin/env python3
"""Extract per-agent benchmark metrics for a pipeline run; emit CSV row + archive raw evidence.

Sources:
  1. PI session JSONL files  -> per-agent token usage (input/output/cacheRead/reasoning/totalTokens)
  2. LiteLLM spend DB (postgres) -> TTFT / prompt tps / gen tps per request
  3. ~/llm_platform/.env     -> postgres credentials (never hardcoded)

Durability (schema_version=1):
  * Archive dir ~/Documents/coding/benchmarks/runs/<run_id>/ holds: raw spend rows
    (<run_id>.spend.jsonl), manifest.json (window + provenance), and copies of the
    orchestrator session + subagent transcripts used. The CSV row is reproducible
    from this dir even if the live DB or pi session is later cleaned.
  * CSV upsert: rows are replaced by run_id, never blind-appended.

Usage:
  extract_pipeline_metrics.py ORCH_SESSION.JSONL \
      --start "2026-09-19 15:57:20" --end "2026-09-19 16:45:10" \
      [--subagent-dir SESSIONS_DIR] [--out benchmarks.csv] [--run-id bench-NNN] \
      [--archive-dir ~/Documents/coding/benchmarks/runs] [--device "..."]
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys

DEFAULT_AGENTS = ["py-senior-dev", "py-lead", "py-staff-dev", "py-architect"]
SCHEMA_VERSION = "1"
DEFAULT_ARCHIVE = os.path.expanduser("~/Documents/coding/benchmarks/runs")
DEFAULT_ENV = os.path.expanduser("~/llm_platform/.env")


def expand_env_refs(env):
    """Resolve ${KEY} references inside .env values (nested, one pass, 10 deep)."""
    for _ in range(10):
        changed = False
        for k, v in list(env.items()):
            if "${" in v:
                nv = v
                for ref in set(v.split("${")[1:]):
                    key = ref.split("}", 1)[0]
                    if key in env:
                        nv = nv.replace("${" + key + "}", env[key])
                if nv != v:
                    env[k] = nv
                    changed = True
        if not changed:
            break
    return env


def load_env(env_path=None):
    """Parse KEY=VALUE lines from .env -> dict; expands ${KEY} refs. { } if missing."""
    env = {}
    try:
        for line in open(env_path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return expand_env_refs(env)


def pg_connect(env):
    """Build psql argv + env from .env values (fallback: defaults for litellm stack)."""
    psql = shutil.which("psql")
    if not psql:
        print("ERROR: psql not found in PATH. Install postgres client.", file=sys.stderr)
        sys.exit(1)
    env_vals = {
        "user": env.get("POSTGRES_USER", "litellm_user"),
        "password": env.get("POSTGRES_PASSWORD", "secure_password_change_me"),
        "db": env.get("POSTGRES_DB", "litellm_db"),
        "port": "5433",  # local docker-exposed port; .env DATABASE_URL points at the internal docker network
    }
    if "DATABASE_URL" in env and "@" in env["DATABASE_URL"]:
        # postgresql://user:pass@host:port/db
        auth, rest = env["DATABASE_URL"].split("://", 1)[1].split("@", 1)
        if ":" in auth:
            env_vals["user"], env_vals["password"] = auth.split(":", 1)
        host, _, db_port = rest.rpartition("/")
        env_vals["db"] = db_port
        if ":" in host:
            candidate = host.rsplit(":", 1)[-1]
            if candidate != "5432":  # 5432 = docker-internal, not locally reachable
                env_vals["port"] = candidate
    argv = [psql, "-h", "127.0.0.1", "-p", env_vals["port"], "-U", env_vals["user"], "-d", env_vals["db"]]
    pgenv = {"PGPASSWORD": env_vals["password"], "PATH": os.environ.get("PATH", "")}
    return argv, pgenv


def read_transcripts(sessions_dir, start_epoch, end_epoch, agents=None):
    """Return {agent: [ {usage..., model, responseId} ]} for subagent transcripts in window."""
    if agents is None:
        agents = DEFAULT_AGENTS
    per_agent = {a: [] for a in agents}
    for f in glob.glob(os.path.join(sessions_dir, "subagent-artifacts", "*_transcript.jsonl")):
        with open(f) as fh:
            first = fh.readline()
        try:
            first_ts = json.loads(first).get("timestamp", "")
        except Exception:
            continue
        name = os.path.basename(f).split("_", 1)[1].rsplit("_", 1)[0]
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            m = e.get("message", {})
            if not isinstance(m, dict) or not m.get("usage"):
                continue
            u = m["usage"]
            t = m.get("timestamp")
            if t is not None and not (start_epoch <= t / 1000 <= end_epoch):
                continue
            per_agent.setdefault(name, []).append({**u, "model": m.get("model", ""), "responseId": m.get("responseId", "")})
    return per_agent


def sum_usage(usages):
    out = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0, "totalTokens": 0}
    for rec in usages:
        u = rec if isinstance(rec, dict) and "usage" not in rec else rec["usage"]
        for k in out:
            out[k] += u.get(k) or 0
    return out


def agent_models_of(records):
    models = {r["model"] for r in records if isinstance(r, dict) and r.get("model")}
    return next(iter(models), "") if models else ""


def spend_rows(start_iso, end_iso, pg_argv, pg_env):
    sql = (
        'SELECT "request_id", "model", extract(epoch from "startTime")::float, '
        'extract(epoch from "completionStartTime")::float, "request_duration_ms", '
        '"prompt_tokens", "completion_tokens" FROM "LiteLLM_SpendLogs" '
        f"WHERE \"startTime\" >= '{start_iso}' AND \"startTime\" <= '{end_iso}' ORDER BY \"startTime\""
    )
    r = subprocess.run(pg_argv + ["-t", "-c", sql], capture_output=True, text=True, env=pg_env)
    rows = []
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) != 7:
            continue
        rows.append({"rid": p[0], "model": p[1], "start": float(p[2]),
                     "cstart": float(p[3]) if p[3] != "" else None,
                     "dur": int(p[4]), "pt": int(p[5]), "ct": int(p[6])})
    return rows


def ttft_metrics(rows, ids):
    """TTFT/tps for the rows whose request_id is in ids."""
    valid = [r for r in rows if r["rid"] in ids and r["cstart"] and r["cstart"] > r["start"]]
    if not valid:
        return {"ttft_s": "", "ptps": "", "gtps": ""}
    sum_ttft = sum(r["cstart"] - r["start"] for r in valid)
    prefill_s = sum(r["cstart"] - r["start"] for r in valid)
    gen_s = sum((r["dur"] / 1000 - (r["cstart"] - r["start"])) for r in valid
                if (r["dur"] / 1000 - (r["cstart"] - r["start"])) > 0)
    ptps = (sum(r["pt"] for r in valid) / prefill_s) if prefill_s else ""
    gtps = (sum(r["ct"] for r in valid) / gen_s) if gen_s else ""
    return {"ttft_s": round(sum_ttft, 3), "ptps": round(ptps, 1) if ptps != "" else "",
            "gtps": round(gtps, 2) if gtps != "" else ""}


def next_run_id(csv_path="~/Documents/coding/benchmarks.csv"):
    """bench-<max+1>; ignores filesystem archive dirs, only reads CSV rows."""
    path = os.path.expanduser(csv_path)
    if not os.path.exists(path):
        return "bench-001"
    try:
        with open(path) as fh:
            ids = [r.get("run_id") for r in csv.DictReader(fh) if r.get("run_id")]
        nums = [int(i.replace("bench-", "")) for i in ids if i.startswith("bench-")]
        return f"bench-{max(nums) + 1:03d}" if nums else "bench-001"
    except Exception:
        return "bench-001"


def git_sha(cwd):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
        return r.stdout.strip()[:12] if r.returncode == 0 else ""
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orch_session", help="path to orchestrator session.jsonl")
    ap.add_argument("--start", required=True, help="run start, ISO or 'YYYY-MM-DD HH:MM:SS' UTC")
    ap.add_argument("--end", required=True, help="run end")
    ap.add_argument("--subagent-dir", default=None,
                    help="sessions dir containing subagent-artifacts/ (default: dirname(orch_session))")
    ap.add_argument("--out", default=None, help="CSV path; row upserted by run_id (creates header if missing)")
    ap.add_argument("--run-id", default=None, help="run_id (default: auto from --out)")
    ap.add_argument("--device", default="MacBook Pro Apple M4 Max")
    ap.add_argument("--archive-dir", default=DEFAULT_ARCHIVE, help="parent dir for per-run archives")
    ap.add_argument("--no-archive", action="store_true", help="skip writing per-run archive dir")
    ap.add_argument("--agents", default=None,
                    help="comma-separated agent names (default: py-senior-dev,py-lead,py-staff-dev,py-architect)")
    ap.add_argument("--env", default=DEFAULT_ENV,
                    help="path to .env file with DB credentials")
    args = ap.parse_args()
    if not args.run_id:
        args.run_id = next_run_id(args.out) if args.out else "bench-NEXT"

    def parse(s):
        s = s.strip()
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)

    start_dt, end_dt = parse(args.start), parse(args.end)
    start_epoch, end_epoch = start_dt.timestamp(), end_dt.timestamp()
    sub_dir = args.subagent_dir or os.path.dirname(args.orch_session)

    env = load_env(args.env)
    pg_argv, pg_env = pg_connect(env)

    # --- orchestrator (main session): token usage in window ---
    orch_usage = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0, "totalTokens": 0}
    orch_ids = set()
    orch_model = ""
    for line in open(args.orch_session):
        e = json.loads(line)
        m = e.get("message", {})
        if not isinstance(m, dict) or not m.get("usage") or not m.get("timestamp"):
            continue
        t = m["timestamp"] / 1000
        if not (start_epoch <= t <= end_epoch):
            continue
        u = m["usage"]
        orch_model = m.get("model") or orch_model
        if m.get("responseId"):
            orch_ids.add(m["responseId"])
        for k in orch_usage:
            orch_usage[k] += u.get(k) or 0

    # --- subagents ---
    agents = args.agents.split(",") if args.agents else DEFAULT_AGENTS
    per_agent = read_transcripts(sub_dir, start_epoch, end_epoch, agents)
    agent_totals = {a: sum_usage(per_agent[a]) for a in agents}
    agent_models = {a: agent_models_of(per_agent[a]) for a in agents}

    # --- TTFT from spend DB (orchestrator; subagents only if calls routed via proxy) ---
    spend = spend_rows(args.start, args.end.replace(" ", "T"), pg_argv, pg_env)
    orch_ttft = ttft_metrics(spend, orch_ids)

    agent_ttft = {a: {"ttft_s": "", "ptps": "", "gtps": ""} for a in agents}
    for a in agents:
        ids = {r["responseId"] for r in per_agent[a] if r.get("responseId")}
        if ids:
            agent_ttft[a] = ttft_metrics(spend, ids)

    # --- Ordered row (schema_version=1) ---
    row = {
        "run_id": args.run_id,
        "device": args.device,
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration": round(end_epoch - start_epoch, 1),
        "schema_version": SCHEMA_VERSION,
    }
    agent_ttft_all = {**{"orchestrator": orch_ttft}, **agent_ttft}
    token_cols = ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")
    for a in ["orchestrator"] + agents:
        row[f"{a}_model"] = orch_model if a == "orchestrator" else agent_models[a]
        totals = orch_usage if a == "orchestrator" else agent_totals[a]
        for col in token_cols:
            row[f"{a}_{col}"] = totals[col]
        t = agent_ttft_all[a]
        row[f"{a}_ttft_s"] = t["ttft_s"]
        row[f"{a}_avg_prompt_tps"] = t["ptps"]
        row[f"{a}_avg_gen_tps"] = t["gtps"]
    row["cached_token_count"] = sum(v["cacheRead"] for v in [orch_usage, *agent_totals.values()])
    row["prefill_token_count"] = sum(v["input"] for v in [orch_usage, *agent_totals.values()])
    row["sum_total_ttft_s"] = orch_ttft["ttft_s"]
    row["avg_prompt_tps"] = orch_ttft["ptps"]
    row["avg_gen_tps"] = orch_ttft["gtps"]

    print(json.dumps(row, indent=1))

    # --- Per-run archive + manifest (durability) ---
    if not args.no_archive and args.run_id != "bench-NEXT":
        run_dir = os.path.join(os.path.expanduser(args.archive_dir), args.run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, f"{args.run_id}.spend.jsonl"), "w") as fh:
            for sd in spend:
                fh.write(json.dumps(sd) + "\n")
        manifest = {
            "run_id": args.run_id,
            "schema_version": SCHEMA_VERSION,
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "orch_session": os.path.basename(args.orch_session),
            "subagent_artifacts": sorted(os.path.basename(f) for f in glob.glob(os.path.join(sub_dir, "subagent-artifacts", "*_transcript.jsonl"))),
            "spend_rows": len(spend),
            "git_sha": git_sha(os.getcwd()),
            "script": os.path.basename(sys.argv[0]),
        }
        with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        # copy the evidence files used
        try:
            shutil.copy2(args.orch_session, os.path.join(run_dir, os.path.basename(args.orch_session)))
        except OSError:
            pass
        for t in glob.glob(os.path.join(sub_dir, "subagent-artifacts", "*_transcript.jsonl")):
            try:
                shutil.copy2(t, os.path.join(run_dir, os.path.basename(t)))
            except OSError:
                pass
        print(f"archived -> {run_dir}", file=sys.stderr)

    # --- CSV upsert by run_id (idempotent) ---
    out = getattr(args, "out", None)
    if out:
        header = list(row.keys())
        existed = os.path.exists(out)
        all_rows = []
        if existed:
            with open(out) as fh:
                all_rows = list(csv.DictReader(fh))
        # replace existing row with same run_id
        all_rows = [r for r in all_rows if r.get("run_id") != row["run_id"]] + [row]
        # normalize to union of old header + new keys
        if existed and all_rows:
            merged = list(dict.fromkeys(list(all_rows[0].keys()) + header))
        else:
            merged = header
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=merged)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in merged})


if __name__ == "__main__":
    main()