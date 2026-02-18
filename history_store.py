from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).with_name("market_pulse_history.db")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts_utc TEXT PRIMARY KEY,
    spx_change REAL,
    ndx_change REAL,
    vix REAL,
    vix_change REAL,
    volume_ratio REAL,
    sentiment_score REAL,
    breadth_ratio REAL,
    cta_net REAL,
    rotation_daily REAL,
    rotation_weekly REAL,
    rotation_monthly REAL,
    best_sector TEXT,
    best_sector_daily REAL,
    worst_sector TEXT,
    worst_sector_daily REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    value REAL,
    UNIQUE(ts_utc, code, message)
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_utc DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _to_utc_iso(dt_value: Any) -> str:
    if isinstance(dt_value, datetime):
        dt = dt_value
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def persist_payload(payload: dict[str, Any]) -> None:
    init_db()

    ts_utc = _to_utc_iso(payload.get("updated_at"))
    snapshot = payload.get("snapshot", {})
    sentiment = payload.get("sentiment", {})
    volume = payload.get("volume_profile", {})
    cta_proxy = payload.get("cta_proxy", {})
    rotation = payload.get("rotation_signal", {})
    sectors = payload.get("sector_returns", pd.DataFrame())

    best_sector = None
    best_sector_daily = None
    worst_sector = None
    worst_sector_daily = None

    if isinstance(sectors, pd.DataFrame) and not sectors.empty and {"sector", "daily"}.issubset(set(sectors.columns)):
        ordered = sectors.sort_values("daily", ascending=False)
        best = ordered.iloc[0]
        worst = ordered.iloc[-1]
        best_sector = str(best["sector"])
        best_sector_daily = _safe_float(best["daily"])
        worst_sector = str(worst["sector"])
        worst_sector_daily = _safe_float(worst["daily"])

    row = (
        ts_utc,
        _safe_float(snapshot.get("^GSPC", {}).get("change_pct")),
        _safe_float(snapshot.get("^IXIC", {}).get("change_pct")),
        _safe_float(snapshot.get("^VIX", {}).get("last")),
        _safe_float(snapshot.get("^VIX", {}).get("change_pct")),
        _safe_float(volume.get("ratio")),
        _safe_float(sentiment.get("composite_score")),
        _safe_float(sentiment.get("breadth_ratio")),
        _safe_float(cta_proxy.get("net_score")),
        _safe_float(rotation.get("spread_daily")),
        _safe_float(rotation.get("spread_weekly")),
        _safe_float(rotation.get("spread_monthly")),
        best_sector,
        best_sector_daily,
        worst_sector,
        worst_sector_daily,
    )

    alerts = payload.get("alerts", [])

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO snapshots (
                ts_utc, spx_change, ndx_change, vix, vix_change, volume_ratio,
                sentiment_score, breadth_ratio, cta_net,
                rotation_daily, rotation_weekly, rotation_monthly,
                best_sector, best_sector_daily, worst_sector, worst_sector_daily
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )

        for alert in alerts:
            conn.execute(
                """
                INSERT OR IGNORE INTO alerts (ts_utc, severity, code, message, value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    str(alert.get("severity", "low")),
                    str(alert.get("code", "unknown")),
                    str(alert.get("message", "")),
                    _safe_float(alert.get("value")),
                ),
            )

        conn.commit()


def load_snapshot_history(hours: int = 72) -> pd.DataFrame:
    init_db()
    since = (datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours)))).replace(microsecond=0).isoformat()

    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM snapshots
            WHERE ts_utc >= ?
            ORDER BY ts_utc ASC
            """,
            conn,
            params=[since],
        )

    if df.empty:
        return df

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["timestamp_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    return df


def load_alert_history(limit: int = 50) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT ts_utc, severity, code, message, value
            FROM alerts
            ORDER BY ts_utc DESC, id DESC
            LIMIT ?
            """,
            conn,
            params=[max(1, int(limit))],
        )

    if df.empty:
        return df

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["timestamp_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    return df
