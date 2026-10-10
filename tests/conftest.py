"""Pytest hooks to capture direct test runs in vault-cacophony into VaultWares run telemetry."""

import os
import sys
import time
import pytest
from pathlib import Path

# Add vaultwares-adk or sibling checkout to path
_candidates = [
    str(Path(__file__).resolve().parents[2] / "vaultwares-adk"),
    r"C:\Users\Administrator\Desktop\Github Repos\vaultwares-adk",
]
for c in _candidates:
    if os.path.isdir(c) and c not in sys.path:
        sys.path.insert(0, c)

try:
    from vaultwares_adk.telemetry import record_run
except ImportError:
    record_run = None


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and record_run:
        duration_ms = round(report.duration * 1000, 2)
        status = "ok" if report.passed else ("error" if report.failed else "cancelled")
        try:
            record_run(
                provider="local",
                runtime="pytest",
                model=item.nodeid,
                task="test",
                project="vault-cacophony",
                service="test-runner",
                duration_ms=duration_ms,
                status=status,
                is_free=True,
                cost_usd=0.0,
                extra={
                    "file": str(item.fspath),
                    "test_name": item.name,
                    "outcome": report.outcome,
                },
            )
        except Exception:
            pass


def pytest_sessionfinish(session, exitstatus):
    """Ensure all recorded runs are flushed and spooled before exit."""
    try:
        from vaultwares_adk.telemetry.worker import get_worker
        get_worker().shutdown(timeout=3.0)
    except Exception:
        pass
