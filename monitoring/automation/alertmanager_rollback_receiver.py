from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("alertmanager_rollback_receiver")

PORT = int(os.getenv("AUTO_ROLLBACK_PORT", "8085"))
GITHUB_TOKEN = os.getenv("AUTO_ROLLBACK_GITHUB_TOKEN", "").strip()
GITHUB_REPOSITORY = os.getenv("AUTO_ROLLBACK_GITHUB_REPOSITORY", "").strip()
GITHUB_WORKFLOW = os.getenv(
    "AUTO_ROLLBACK_GITHUB_WORKFLOW",
    "rollback-canary.yml",
).strip()
GITHUB_REF = os.getenv("AUTO_ROLLBACK_GITHUB_REF", "main").strip()
REGISTRY_ACTION = os.getenv(
    "AUTO_ROLLBACK_REGISTRY_ACTION",
    "none",
).strip()
ALERT_NAMES = {
    item.strip()
    for item in os.getenv(
        "AUTO_ROLLBACK_ALERT_NAMES",
        "CandidateModelBehaviorRegression",
    ).split(",")
    if item.strip()
}
COOLDOWN_SECONDS = int(os.getenv("AUTO_ROLLBACK_COOLDOWN_SECONDS", "900"))
STATE_FILE = Path(
    os.getenv(
        "AUTO_ROLLBACK_STATE_FILE",
        "/state/auto_rollback_state.json",
    )
)


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Ignoring unreadable rollback state at %s", STATE_FILE)
        return {}


def save_state(payload: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def github_workflow_url() -> str:
    return (
        "https://api.github.com/repos/"
        f"{GITHUB_REPOSITORY}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"
    )


def dispatch_rollback(alerts: list[dict[str, Any]]) -> None:
    if not GITHUB_TOKEN:
        raise RuntimeError("AUTO_ROLLBACK_GITHUB_TOKEN is not configured.")
    if not GITHUB_REPOSITORY:
        raise RuntimeError("AUTO_ROLLBACK_GITHUB_REPOSITORY is not configured.")

    payload = {
        "ref": GITHUB_REF,
        "inputs": {
            "action": "rollback",
            "percent": "0",
            "registry_action": REGISTRY_ACTION,
        },
    }
    request = urllib.request.Request(
        github_workflow_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "tcb-auto-rollback-receiver",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            logger.warning(
                "Triggered rollback workflow | status=%s | alerts=%s",
                response.status,
                [alert["labels"].get("alertname") for alert in alerts],
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub workflow dispatch failed: {exc.code} {body}"
        ) from exc


def matched_firing_alerts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for alert in payload.get("alerts", []):
        if alert.get("status") != "firing":
            continue
        alert_name = alert.get("labels", {}).get("alertname", "")
        if alert_name in ALERT_NAMES:
            matched.append(alert)
    return matched


def should_dispatch(alerts: list[dict[str, Any]]) -> bool:
    now = int(time.time())
    state = load_state()
    last_dispatch_at = int(state.get("last_dispatch_at", 0) or 0)
    last_fingerprints = set(state.get("fingerprints", []))
    current_fingerprints = {
        alert.get("fingerprint", "")
        for alert in alerts
        if alert.get("fingerprint")
    }
    if (
        current_fingerprints
        and current_fingerprints == last_fingerprints
        and now - last_dispatch_at < COOLDOWN_SECONDS
    ):
        logger.info(
            "Skipping duplicate rollback trigger during cooldown | fingerprints=%s",
            sorted(current_fingerprints),
        )
        return False
    save_state(
        {
            "last_dispatch_at": now,
            "fingerprints": sorted(current_fingerprints),
            "last_alert_names": sorted(
                {
                    alert.get("labels", {}).get("alertname", "")
                    for alert in alerts
                }
            ),
        }
    )
    return True


class ReceiverHandler(BaseHTTPRequestHandler):
    def _json_response(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"status": "not_found"},
            )
            return
        self._json_response(
            HTTPStatus.OK,
            {
                "status": "ok",
                "alert_names": sorted(ALERT_NAMES),
                "cooldown_seconds": COOLDOWN_SECONDS,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/alertmanager":
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"status": "not_found"},
            )
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"status": "invalid_json"},
            )
            return

        alerts = matched_firing_alerts(payload)
        if not alerts:
            self._json_response(
                HTTPStatus.ACCEPTED,
                {"status": "ignored", "reason": "no_matching_firing_alerts"},
            )
            return

        if not should_dispatch(alerts):
            self._json_response(
                HTTPStatus.ACCEPTED,
                {"status": "cooldown", "alerts": len(alerts)},
            )
            return

        try:
            dispatch_rollback(alerts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Automatic rollback dispatch failed")
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "dispatch_failed", "error": str(exc)},
            )
            return

        self._json_response(
            HTTPStatus.OK,
            {"status": "rollback_triggered", "alerts": len(alerts)},
        )

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ReceiverHandler)
    logger.info(
        "Auto rollback receiver listening | port=%s | alerts=%s",
        PORT,
        sorted(ALERT_NAMES),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
