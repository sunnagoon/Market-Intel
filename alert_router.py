from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib import parse, request

STATE_PATH = Path(__file__).with_name("alert_router_state.json")

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_sent": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_sent": {}}
    if not isinstance(data, dict):
        return {"last_sent": {}}
    if "last_sent" not in data or not isinstance(data["last_sent"], dict):
        data["last_sent"] = {}
    return data


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _severity_ok(severity: str, minimum: str) -> bool:
    s = SEVERITY_RANK.get(str(severity).lower(), 0)
    m = SEVERITY_RANK.get(str(minimum).lower(), 2)
    return s >= m


def _fmt_alert_message(payload: dict[str, Any], alert: dict[str, Any]) -> str:
    updated = payload.get("updated_at")
    ts_text = "n/a"
    if isinstance(updated, datetime):
        ts_text = updated.strftime("%Y-%m-%d %I:%M:%S %p ET")

    regime = payload.get("regime_engine", {}).get("regime", "n/a")
    conf = payload.get("cross_asset_confirmation", {}).get("confirmation_ratio")
    conf_text = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "n/a"

    code = str(alert.get("code", "signal"))
    severity = str(alert.get("severity", "low")).upper()
    message = str(alert.get("message", ""))

    return (
        f"[{severity}] {code}\n"
        f"{message}\n"
        f"Regime: {regime} | Cross-asset confirmation: {conf_text}\n"
        f"Timestamp: {ts_text}"
    )


def _send_telegram(text: str, token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST")
    with request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Telegram HTTP {resp.status}")


def _send_discord(text: str, webhook: str) -> None:
    payload = json.dumps({"content": text}).encode("utf-8")
    req = request.Request(url=webhook, data=payload, method="POST", headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Discord HTTP {resp.status}")


def _send_email(text: str, host: str, port: int, username: str, password: str, mail_from: str, mail_to: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = "Market Pulse Alert"
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(text)

    with smtplib.SMTP(host, port, timeout=12) as smtp:
        smtp.starttls()
        if username:
            smtp.login(username, password)
        smtp.send_message(msg)


def route_alerts(payload: dict[str, Any]) -> dict[str, Any]:
    enabled = _parse_bool(os.getenv("ALERT_ROUTING_ENABLED"), default=False)
    minimum_severity = os.getenv("ALERT_ROUTING_MIN_SEVERITY", "medium").strip().lower()
    cooldown_min = max(1, int(os.getenv("ALERT_ROUTING_COOLDOWN_MIN", "30")))
    require_confirmation = _parse_bool(os.getenv("ALERT_ROUTING_REQUIRE_CONFIRMATION", "1"), default=True)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    smtp_host = os.getenv("ALERT_EMAIL_HOST", "").strip()
    smtp_port = int(os.getenv("ALERT_EMAIL_PORT", "587"))
    smtp_user = os.getenv("ALERT_EMAIL_USER", "").strip()
    smtp_pass = os.getenv("ALERT_EMAIL_PASS", "").strip()
    email_from = os.getenv("ALERT_EMAIL_FROM", "").strip()
    email_to = os.getenv("ALERT_EMAIL_TO", "").strip()

    destinations: list[str] = []
    if telegram_token and telegram_chat:
        destinations.append("telegram")
    if discord_webhook:
        destinations.append("discord")
    if smtp_host and email_from and email_to:
        destinations.append("email")

    alerts = payload.get("alerts", [])
    if not isinstance(alerts, list):
        alerts = []

    cross_ratio = payload.get("cross_asset_confirmation", {}).get("confirmation_ratio")
    cross_ok = isinstance(cross_ratio, (int, float)) and cross_ratio >= 0.5

    state = _load_state()
    last_sent = state.get("last_sent", {})

    now = _utc_now()
    cooldown = timedelta(minutes=cooldown_min)

    sent_count = 0
    skipped_count = 0
    error_count = 0
    errors: list[str] = []

    for alert in alerts:
        severity = str(alert.get("severity", "low")).lower()
        if not _severity_ok(severity, minimum_severity):
            skipped_count += 1
            continue

        if require_confirmation and severity != "high" and not cross_ok:
            skipped_count += 1
            continue

        code = str(alert.get("code", "signal"))
        msg = str(alert.get("message", ""))
        key = f"{code}|{msg}"

        last_text = last_sent.get(key)
        if isinstance(last_text, str):
            try:
                last_dt = datetime.fromisoformat(last_text)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                last_dt = None
            if last_dt is not None and now - last_dt < cooldown:
                skipped_count += 1
                continue

        outbound = _fmt_alert_message(payload, alert)

        if enabled and destinations:
            try:
                if "telegram" in destinations:
                    _send_telegram(outbound, telegram_token, telegram_chat)
                if "discord" in destinations:
                    _send_discord(outbound, discord_webhook)
                if "email" in destinations:
                    _send_email(outbound, smtp_host, smtp_port, smtp_user, smtp_pass, email_from, email_to)
                sent_count += 1
                last_sent[key] = now.isoformat()
            except Exception as exc:
                error_count += 1
                errors.append(f"{code}: {exc}")
        else:
            skipped_count += 1

    state["last_sent"] = last_sent
    try:
        _save_state(state)
    except Exception as exc:
        error_count += 1
        errors.append(f"state_save: {exc}")

    return {
        "enabled": enabled,
        "destinations": destinations,
        "min_severity": minimum_severity,
        "cooldown_min": cooldown_min,
        "require_confirmation": require_confirmation,
        "candidate_alerts": len(alerts),
        "sent_count": sent_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "errors": errors[:5],
        "cross_confirmation_ratio": cross_ratio,
    }
