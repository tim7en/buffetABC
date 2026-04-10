#!/usr/bin/env python3
"""
End-to-end smoke-test for the spy_qqq_signal_api container.

What this tests:
  1. Docker image builds successfully
  2. Container starts and /health responds
  3. The pipeline runs a full data-download + signal computation
     (triggered automatically on startup, or manually via POST /refresh)
  4. /notice returns a single ENTER / HOLD / REDUCE action notice
  5. /signal returns a detailed signal with all model outputs
  6. /signal/history returns recent history

Usage (from repo root):
    python production/spy_qqq_signal_api/test_container.py

Options:
    --skip-build     Skip docker build (use existing image)
    --skip-teardown  Keep the container running after the test
    --base-url URL   Test against an already-running instance (skips docker management)
    --timeout N      Max seconds to wait for the pipeline to complete (default 1200)
    --token SECRET   Bearer token for POST /refresh (matches REFRESH_TOKEN env var)

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_NAME = "spy-qqq-signal-api:latest"
CONTAINER_NAME = "spy_qqq_signal_api_test"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT = 1200   # 20 minutes — first run trains walk-forward models

REPO_ROOT = "."          # relative to CWD; adjust if needed

# ANSI colours (fall back gracefully on Windows without ANSI support)
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{RESET}"


# --------------------------------------------------------------------------- #
# Docker helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def build_image() -> None:
    print(_c(BOLD, "\n[1/6] Building Docker image …"))
    _run([
        "docker", "build",
        "-f", "production/spy_qqq_signal_api/Dockerfile",
        "-t", IMAGE_NAME,
        ".",
    ])
    print(_c(GREEN, "      Image built OK."))


def start_container(port: int, refresh_token: str) -> None:
    print(_c(BOLD, f"\n[2/6] Starting container on port {port} …"))
    # Remove any leftover test container
    _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, capture=True)

    _run([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", f"{port}:{DEFAULT_PORT}",
        "-e", f"REFRESH_TOKEN={refresh_token}",
        "-e", "MAX_STATE_AGE_H=25",
        "-e", "DATA_DIR=/data",
        IMAGE_NAME,
    ])
    print(_c(GREEN, "      Container started."))


def stop_container() -> None:
    print(_c(BOLD, "\n[6/6] Tearing down container …"))
    _run(["docker", "rm", "-f", CONTAINER_NAME], check=False, capture=True)
    print(_c(GREEN, "      Container removed."))


def container_logs(tail: int = 40) -> str:
    result = _run(["docker", "logs", "--tail", str(tail), CONTAINER_NAME],
                  check=False, capture=True)
    return (result.stdout or "") + (result.stderr or "")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def _get(session: requests.Session, url: str) -> dict:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(session: requests.Session, url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = session.post(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Wait helpers
# --------------------------------------------------------------------------- #
def wait_for_health(base_url: str, timeout: int = 120) -> None:
    print(_c(BOLD, "\n[3/6] Waiting for container to become healthy …"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=10)
            if r.status_code == 200:
                print(_c(GREEN, f"      /health OK  ({r.elapsed.total_seconds():.2f}s)"))
                return
        except Exception:
            pass
        time.sleep(3)
    print(container_logs())
    raise RuntimeError(f"Container did not become healthy within {timeout}s")


def wait_for_fresh_signal(
    session: requests.Session,
    base_url: str,
    timeout: int = DEFAULT_TIMEOUT,
    refresh_token: str = "",
) -> None:
    """Trigger a refresh then poll until signal_ready=True and is_stale=False."""
    print(_c(BOLD, "\n[4/6] Triggering data download + signal computation …"))

    try:
        resp = _post(session, f"{base_url}/refresh", refresh_token)
        print(f"      POST /refresh → {resp}")
    except Exception as exc:
        print(f"      POST /refresh failed ({exc}) — relying on startup auto-refresh.")

    print(f"      Polling /health for up to {timeout // 60} minutes …")
    deadline = time.time() + timeout
    dot_count = 0
    while time.time() < deadline:
        try:
            h = _get(session, f"{base_url}/health")
            if h.get("signal_ready") and not h.get("is_stale") and not h.get("refresh_in_progress"):
                print()
                age_h = h.get("state_age_hours", "?")
                as_of = h.get("as_of_date", "?")
                print(_c(GREEN, f"      Signal ready — as_of={as_of}  age={age_h}h"))
                return
            # Still running — print a dot every 10 s
            if dot_count % 10 == 0:
                status_flag = (
                    "refreshing" if h.get("refresh_in_progress")
                    else "waiting-for-signal" if not h.get("signal_ready")
                    else "stale"
                )
                print(f"      … [{status_flag}] {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC", end="\r")
        except Exception as exc:
            print(f"      /health poll error: {exc}")

        time.sleep(10)
        dot_count += 1

    print()
    print(container_logs())
    raise RuntimeError(f"Pipeline did not complete within {timeout}s")


# --------------------------------------------------------------------------- #
# Test assertions
# --------------------------------------------------------------------------- #
def test_notice(session: requests.Session, base_url: str) -> None:
    print(_c(BOLD, "\n[5/6] Fetching /notice …"))
    data = _get(session, f"{base_url}/notice")

    notice   = data.get("notice", "???")
    detail   = data.get("detail", "")
    as_of    = data.get("as_of_date", "?")
    spy_sig  = data.get("spy_signal", "?")
    qqq_sig  = data.get("qqq_signal", "?")
    vix      = data.get("vix_level")
    computed = data.get("computed_at_utc", "?")
    stale    = data.get("is_stale", False)

    notice_colour = GREEN if notice.startswith("ENTER") else (RED if notice == "REDUCE" else YELLOW)

    print()
    print("  " + "─" * 60)
    print(f"  {_c(BOLD, 'TRADE NOTICE')}   [{as_of}]")
    print(f"  {_c(notice_colour + BOLD, notice)}")
    print(f"  {detail}")
    print()
    vix_txt = f"{vix:.1f}" if vix is not None else "n/a"
    print(f"  SPY gate : {spy_sig}")
    print(f"  QQQ sig  : {qqq_sig}")
    print(f"  VIX      : {vix_txt}")
    print(f"  Computed : {computed}")
    if stale:
        print(_c(YELLOW, f"  WARNING  : {data.get('stale_warning', 'stale signal')}"))
    print("  " + "─" * 60)
    print()

    # Assertions
    assert notice in ("ENTER", "ENTER — DOUBLE DCA", "HOLD", "REDUCE"), \
        f"Unexpected notice value: {notice!r}"
    assert as_of, "as_of_date is empty"
    assert computed, "computed_at_utc is empty"
    print(_c(GREEN, "  [PASS] /notice assertions OK"))


def test_signal_detail(session: requests.Session, base_url: str) -> None:
    data = _get(session, f"{base_url}/signal")
    ps = data.get("policy_signal", "")
    assert ps in ("risk_on", "neutral", "risk_off"), f"Invalid policy_signal: {ps!r}"
    probs = data.get("probabilities", {})
    assert probs, "/signal missing probabilities"
    print(_c(GREEN, f"  [PASS] /signal detail OK  (policy_signal={ps})"))


def test_signal_history(session: requests.Session, base_url: str) -> None:
    data = _get(session, f"{base_url}/signal/history?n=10")
    history = data.get("history", [])
    assert isinstance(history, list), "/signal/history is not a list"
    assert len(history) > 0, "/signal/history is empty"
    print(_c(GREEN, f"  [PASS] /signal/history OK  ({len(history)} records)"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-build",    action="store_true", help="Skip docker build")
    parser.add_argument("--skip-teardown", action="store_true", help="Leave container running")
    parser.add_argument("--base-url",      default="", help="Test an already-running instance")
    parser.add_argument("--port",          type=int, default=DEFAULT_PORT, help="Host port to bind")
    parser.add_argument("--timeout",       type=int, default=DEFAULT_TIMEOUT, help="Max wait seconds")
    parser.add_argument("--token",         default="", help="REFRESH_TOKEN value")
    args = parser.parse_args()

    manage_docker = not args.base_url
    base_url = args.base_url or f"http://localhost:{args.port}"

    print(_c(BOLD, "=" * 62))
    print(_c(BOLD, "  spy_qqq_signal_api — container end-to-end test"))
    print(_c(BOLD, "=" * 62))
    print(f"  base_url : {base_url}")
    print(f"  timeout  : {args.timeout}s")
    print()

    try:
        if manage_docker:
            if not args.skip_build:
                build_image()
            start_container(args.port, args.token)

        session = requests.Session()

        wait_for_health(base_url, timeout=120)
        wait_for_fresh_signal(session, base_url, timeout=args.timeout, refresh_token=args.token)

        # ---- run tests ------------------------------------------------ #
        print(_c(BOLD, "\n[5/6] Running API tests …"))
        failures: list[str] = []
        for fn in (test_notice, test_signal_detail, test_signal_history):
            try:
                fn(session, base_url)
            except AssertionError as exc:
                failures.append(f"{fn.__name__}: {exc}")
                print(_c(RED, f"  [FAIL] {fn.__name__}: {exc}"))
            except Exception as exc:
                failures.append(f"{fn.__name__}: {exc}")
                print(_c(RED, f"  [ERROR] {fn.__name__}: {exc}"))

        if failures:
            print(_c(RED, f"\n  {len(failures)} test(s) FAILED."))
            return 1

        print(_c(GREEN + BOLD, "\n  All tests PASSED."))
        return 0

    except Exception as exc:
        print(_c(RED, f"\n[FATAL] {exc}"))
        if manage_docker:
            print(container_logs(tail=60))
        return 1

    finally:
        if manage_docker and not args.skip_teardown:
            stop_container()
        elif manage_docker:
            print(f"\n  Container '{CONTAINER_NAME}' left running — stop with:")
            print(f"    docker rm -f {CONTAINER_NAME}")


if __name__ == "__main__":
    sys.exit(main())
