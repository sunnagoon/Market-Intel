from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import os
import numpy as np
import pandas as pd
import yfinance as yf

US_EASTERN = ZoneInfo("America/New_York")

SECTOR_ETFS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

MARKET_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y Treasury Yield",
    "UUP": "US Dollar",
    "CL=F": "WTI Crude",
}


YIELD_TICKERS = {
    "13W": "^IRX",
    "5Y": "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

YIELD_INDEX_TICKERS = {"^IRX", "^FVX", "^TNX", "^TYX"}

TREASURY_FUTURES = {
    "ZT=F": "ZT (2Y)",
    "ZN=F": "ZN (10Y)",
    "ZB=F": "ZB (30Y)",
    "UB=F": "UB (Ultra Bond)",
}


VOLUME_PROXY_ETFS = ["SPY", "QQQ", "DIA", "IWM"]

CTA_PROXY_TICKERS = {
    "SPY": "US Equities",
    "TLT": "Long Bonds",
    "UUP": "US Dollar",
    "GLD": "Gold",
    "DBC": "Broad Commodities",
    "USO": "Crude Oil",
    "HYG": "Credit Risk",
}

POSITIVE_WORDS = {
    "beat",
    "beats",
    "strong",
    "growth",
    "optimism",
    "surge",
    "rally",
    "upside",
    "bullish",
    "upgrade",
    "cooling",
    "decline",
    "easing",
}

NEGATIVE_WORDS = {
    "miss",
    "misses",
    "weak",
    "slowdown",
    "selloff",
    "drop",
    "downside",
    "bearish",
    "downgrade",
    "inflation",
    "hot",
    "war",
    "risk",
}

CYCLICAL_SECTOR_TICKERS = {"XLY", "XLF", "XLI", "XLB", "XLK", "XLE"}
DEFENSIVE_SECTOR_TICKERS = {"XLP", "XLU", "XLV", "XLRE"}

CROSS_ASSET_EXTRA_TICKERS = {"LQD"}
LIQUIDITY_MONITOR_TICKERS = {"TIP", "IEF", "HYG", "LQD", "UUP", "TLT", "SHY"}
PUT_SKEW_UNDERLYINGS = ["SPY", "QQQ", "IWM"]

DEFAULT_WATCHLIST = ["SPY", "QQQ", "IWM", "TLT", "UUP", "GLD", "HYG", "LQD", "XLF", "XLK"]

FOMC_CALENDAR = {
    2026: [
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ]
}


@dataclass
class SessionProgress:
    is_open: bool
    elapsed_minutes: int
    progress: float
    label: str


def _download_batch(
    tickers: list[str], period: str = "6mo", interval: str = "1d", prepost: bool = False
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            " ".join(tickers),
            period=period,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
            prepost=prepost,
        )
    except Exception:
        return pd.DataFrame()
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_convert(US_EASTERN).tz_localize(None)
    return data


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        out = data[ticker].dropna(how="all")
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)
    # Single ticker fallback.
    return data.dropna(how="all")


def _safe_return(close: pd.Series, periods_back: int) -> float:
    close = close.dropna()
    if len(close) <= periods_back:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-1 - periods_back]) - 1.0)


def get_session_progress(now: datetime | None = None) -> SessionProgress:
    now_et = now.astimezone(US_EASTERN) if now else datetime.now(US_EASTERN)
    market_open = time(9, 30)
    market_close = time(16, 0)
    is_weekday = now_et.weekday() < 5
    session_minutes = 390

    if not is_weekday:
        return SessionProgress(False, session_minutes, 1.0, "Weekend")
    if now_et.time() < market_open:
        return SessionProgress(False, 0, 0.0, "Pre-market")
    if now_et.time() >= market_close:
        return SessionProgress(False, session_minutes, 1.0, "Post-close")

    open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    elapsed = int((now_et - open_dt).total_seconds() // 60)
    elapsed = max(1, min(session_minutes, elapsed))
    return SessionProgress(True, elapsed, elapsed / session_minutes, "Regular session")


def _build_sector_returns(daily_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, sector_name in SECTOR_ETFS.items():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if len(close) < 30:
            continue
        rows.append(
            {
                "ticker": ticker,
                "sector": sector_name,
                "daily": _safe_return(close, 1),
                "weekly": _safe_return(close, 5),
                "monthly": _safe_return(close, 21),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("daily", ascending=False).reset_index(drop=True) if not out.empty else out


def _build_market_snapshot(daily_data: pd.DataFrame) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for ticker, label in MARKET_TICKERS.items():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if len(close) < 2:
            continue
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change_pct = (last / prev) - 1.0 if prev else np.nan
        snapshot[ticker] = {"label": label, "last": last, "change_pct": change_pct}
    return snapshot


def _compute_volume_profile(daily_data: pd.DataFrame) -> dict[str, Any]:
    intraday = _download_batch(VOLUME_PROXY_ETFS, period="1d", interval="1m", prepost=False)
    session = get_session_progress()
    component_rows: list[dict[str, Any]] = []
    total_current = 0.0
    total_expected = 0.0

    for ticker in VOLUME_PROXY_ETFS:
        daily_frame = _extract_ticker_frame(daily_data, ticker)
        intraday_frame = _extract_ticker_frame(intraday, ticker)
        if daily_frame.empty or "Volume" not in daily_frame:
            continue
        avg_20d = float(daily_frame["Volume"].dropna().tail(20).mean())
        today_volume = float(daily_frame["Volume"].dropna().iloc[-1]) if not daily_frame["Volume"].dropna().empty else 0.0
        intraday_volume = 0.0
        if not intraday_frame.empty and "Volume" in intraday_frame:
            intraday_volume = float(intraday_frame["Volume"].fillna(0).sum())
        observed = intraday_volume if session.is_open and intraday_volume > 0 else today_volume
        expected = avg_20d * (session.progress if session.is_open else 1.0)
        ratio = (observed / expected) if expected > 0 else np.nan
        total_current += observed
        total_expected += expected
        component_rows.append(
            {
                "ticker": ticker,
                "observed_volume": observed,
                "expected_volume": expected,
                "avg_20d_volume": avg_20d,
                "ratio": ratio,
            }
        )

    total_ratio = (total_current / total_expected) if total_expected > 0 else np.nan
    if np.isnan(total_ratio):
        regime = "Unavailable"
    elif total_ratio < 0.8:
        regime = "Light volume"
    elif total_ratio < 1.2:
        regime = "Normal volume"
    elif total_ratio < 1.5:
        regime = "Heavy volume"
    else:
        regime = "Very heavy volume"

    return {
        "session": session,
        "ratio": total_ratio,
        "regime": regime,
        "components": pd.DataFrame(component_rows),
    }


def _headline_sentiment_score(text: str) -> float:
    words = [w.strip(".,:;!?()[]{}\"'").lower() for w in text.split()]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (pos - neg) / max(4.0, float(len(words)))


def _parse_news_datetime(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=US_EASTERN)
    if isinstance(value, str) and value:
        try:
            iso = value.replace("Z", "+00:00")
            return datetime.fromisoformat(iso).astimezone(US_EASTERN)
        except ValueError:
            return None
    return None


def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content", {}) if isinstance(item, dict) else {}
    title = ((content.get("title") if isinstance(content, dict) else None) or item.get("title") or "").strip()
    if not title:
        return None

    canonical = content.get("canonicalUrl", {}) if isinstance(content, dict) else {}
    click = content.get("clickThroughUrl", {}) if isinstance(content, dict) else {}
    link = (
        (click.get("url") if isinstance(click, dict) else None)
        or (canonical.get("url") if isinstance(canonical, dict) else None)
        or item.get("link")
    )

    provider = content.get("provider", {}) if isinstance(content, dict) else {}
    publisher = (
        (provider.get("displayName") if isinstance(provider, dict) else None)
        or item.get("publisher")
        or "Unknown"
    )

    published = _parse_news_datetime(
        (content.get("pubDate") if isinstance(content, dict) else None)
        or item.get("providerPublishTime")
    )

    return {
        "title": title,
        "link": link,
        "publisher": publisher,
        "published": published,
        "sentiment_score": _headline_sentiment_score(title),
    }


def _fetch_headlines(limit: int = 8) -> list[dict[str, Any]]:
    symbols = ["SPY", "^GSPC", "QQQ", "DIA"]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for symbol in symbols:
        try:
            news = yf.Ticker(symbol).news or []
        except Exception:
            news = []
        for raw in news:
            if not isinstance(raw, dict):
                continue
            parsed = _normalize_news_item(raw)
            if not parsed:
                continue
            key = parsed["link"] or parsed["title"]
            if key in seen:
                continue
            seen.add(key)
            merged.append(parsed)
            if len(merged) >= limit * 2:
                break
        if len(merged) >= limit * 2:
            break

    merged.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=US_EASTERN), reverse=True)
    return merged[:limit]


def _build_sentiment_bundle(
    snapshot: dict[str, dict[str, float]],
    sector_returns: pd.DataFrame,
    headlines: list[dict[str, Any]],
    daily_data: pd.DataFrame,
) -> dict[str, Any]:
    vix_value = snapshot.get("^VIX", {}).get("last", np.nan)

    breadth_daily = np.nan
    breadth_weekly = np.nan
    breadth_monthly = np.nan
    if not sector_returns.empty:
        if "daily" in sector_returns:
            breadth_daily = float((sector_returns["daily"] > 0).mean())
        if "weekly" in sector_returns:
            breadth_weekly = float((sector_returns["weekly"] > 0).mean())
        if "monthly" in sector_returns:
            breadth_monthly = float((sector_returns["monthly"] > 0).mean())

    breadth = breadth_daily
    headline_score = float(np.mean([h["sentiment_score"] for h in headlines])) if headlines else 0.0

    spy_frame = _extract_ticker_frame(daily_data, "SPY")
    trend_component = 0.0
    if not spy_frame.empty and "Close" in spy_frame and len(spy_frame["Close"].dropna()) >= 60:
        close = spy_frame["Close"].dropna()
        c = float(close.iloc[-1])
        ma20 = float(close.tail(20).mean())
        ma50 = float(close.tail(50).mean())
        trend_component = 1.0 if c > ma20 and c > ma50 else (-1.0 if c < ma20 and c < ma50 else 0.0)

    if np.isnan(vix_value):
        vix_component = 0.0
    elif vix_value < 15:
        vix_component = 1.0
    elif vix_value < 22:
        vix_component = 0.2
    elif vix_value < 30:
        vix_component = -0.5
    else:
        vix_component = -1.0

    breadth_component = (breadth - 0.5) * 2 if not np.isnan(breadth) else 0.0
    news_component = float(np.clip(headline_score * 3, -1.0, 1.0))

    composite = np.mean([vix_component, breadth_component, trend_component, news_component]) * 100
    composite = float(np.clip(composite, -100, 100))

    if composite >= 35:
        regime = "Risk-on"
    elif composite <= -35:
        regime = "Risk-off"
    else:
        regime = "Neutral / mixed"

    return {
        "composite_score": composite,
        "regime": regime,
        "components": {
            "vix_component": vix_component * 100,
            "breadth_component": breadth_component * 100,
            "trend_component": trend_component * 100,
            "news_component": news_component * 100,
        },
        "breadth_ratio": breadth,
        "breadth_daily": breadth_daily,
        "breadth_weekly": breadth_weekly,
        "breadth_monthly": breadth_monthly,
        "breadth_metrics": {
            "daily": breadth_daily,
            "weekly": breadth_weekly,
            "monthly": breadth_monthly,
        },
        "headline_score": headline_score,
        "vix_value": vix_value,
    }


def _build_cta_proxy(daily_data: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    net = 0.0
    for ticker, asset_name in CTA_PROXY_TICKERS.items():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if len(close) < 120:
            continue
        last = float(close.iloc[-1])
        ma20 = float(close.tail(20).mean())
        ma100 = float(close.tail(100).mean())
        vol20 = float(close.pct_change().tail(20).std() * np.sqrt(252))
        vol20 = max(vol20, 0.01)
        trend_score = (1 if last > ma20 else -1) + (1 if last > ma100 else -1)
        strength = float(np.clip((last / ma100 - 1.0) / vol20, -3.0, 3.0))

        if trend_score == 2:
            stance = "Long"
        elif trend_score == 0 and last > ma100:
            stance = "Light Long"
        elif trend_score == 0 and last <= ma100:
            stance = "Light Short"
        else:
            stance = "Short"

        model_score = trend_score * 25 + strength * 8
        net += model_score
        rows.append(
            {
                "ticker": ticker,
                "asset": asset_name,
                "last": last,
                "ma20": ma20,
                "ma100": ma100,
                "trend_score": trend_score,
                "strength_z": strength,
                "stance": stance,
                "model_score": model_score,
            }
        )

    table = pd.DataFrame(rows).sort_values("model_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    net_score = float(np.clip(net, -100, 100))
    return {"net_score": net_score, "table": table}


def _basket_index(daily_data: pd.DataFrame, tickers: set[str]) -> pd.Series:
    series_list: list[pd.Series] = []
    for ticker in sorted(tickers):
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if close.empty:
            continue
        close.name = ticker
        series_list.append(close)

    if not series_list:
        return pd.Series(dtype=float)

    panel = pd.concat(series_list, axis=1).sort_index().ffill().dropna(how="all")
    if panel.empty:
        return pd.Series(dtype=float)

    panel = panel.tail(160)
    base = panel.iloc[0]
    norm = panel.divide(base)
    return norm.mean(axis=1)


def _build_sector_rotation_signal(sector_returns: pd.DataFrame, daily_data: pd.DataFrame) -> dict[str, Any]:
    cyc = sector_returns[sector_returns["ticker"].isin(CYCLICAL_SECTOR_TICKERS)] if not sector_returns.empty else pd.DataFrame()
    dfn = sector_returns[sector_returns["ticker"].isin(DEFENSIVE_SECTOR_TICKERS)] if not sector_returns.empty else pd.DataFrame()

    def avg_or_nan(frame: pd.DataFrame, col: str) -> float:
        if frame.empty or col not in frame:
            return np.nan
        return float(frame[col].mean())

    cyc_daily = avg_or_nan(cyc, "daily")
    cyc_weekly = avg_or_nan(cyc, "weekly")
    cyc_monthly = avg_or_nan(cyc, "monthly")
    def_daily = avg_or_nan(dfn, "daily")
    def_weekly = avg_or_nan(dfn, "weekly")
    def_monthly = avg_or_nan(dfn, "monthly")

    spread_daily = cyc_daily - def_daily if not np.isnan(cyc_daily) and not np.isnan(def_daily) else np.nan
    spread_weekly = cyc_weekly - def_weekly if not np.isnan(cyc_weekly) and not np.isnan(def_weekly) else np.nan
    spread_monthly = cyc_monthly - def_monthly if not np.isnan(cyc_monthly) and not np.isnan(def_monthly) else np.nan

    if not np.isnan(spread_daily) and not np.isnan(spread_weekly):
        if spread_daily > 0 and spread_weekly > 0:
            regime = "Cyclicals leading (risk-on rotation)"
        elif spread_daily < 0 and spread_weekly < 0:
            regime = "Defensives leading (risk-off rotation)"
        else:
            regime = "Mixed rotation"
    else:
        regime = "Rotation unavailable"

    cyc_index = _basket_index(daily_data, CYCLICAL_SECTOR_TICKERS)
    def_index = _basket_index(daily_data, DEFENSIVE_SECTOR_TICKERS)
    ratio_history = pd.DataFrame()
    if not cyc_index.empty and not def_index.empty:
        merged = pd.concat([cyc_index.rename("cyclical"), def_index.rename("defensive")], axis=1).dropna()
        if not merged.empty:
            merged = merged.tail(126)
            merged["ratio"] = merged["cyclical"] / merged["defensive"]
            ma63 = merged["ratio"].rolling(63).mean()
            sd63 = merged["ratio"].rolling(63).std()
            merged["ratio_z63"] = (merged["ratio"] - ma63) / sd63
            merged = merged.reset_index()
            date_col = merged.columns[0]
            ratio_history = merged.rename(columns={date_col: "date"})

    return {
        "regime": regime,
        "cyclical_daily": cyc_daily,
        "defensive_daily": def_daily,
        "spread_daily": spread_daily,
        "cyclical_weekly": cyc_weekly,
        "defensive_weekly": def_weekly,
        "spread_weekly": spread_weekly,
        "cyclical_monthly": cyc_monthly,
        "defensive_monthly": def_monthly,
        "spread_monthly": spread_monthly,
        "ratio_history": ratio_history,
    }



def _classify_curve_state(spread_bps: float) -> str:
    if np.isnan(spread_bps):
        return "Unavailable"
    if spread_bps < 0:
        return "Inverted"
    if spread_bps < 25:
        return "Flat"
    if spread_bps < 100:
        return "Normal"
    return "Steep"


def _classify_shift(delta_bps: float) -> str:
    if np.isnan(delta_bps):
        return "Insufficient data"
    if delta_bps > 3:
        return "Steepening"
    if delta_bps < -3:
        return "Flattening"
    return "Mostly unchanged"


def _build_yield_curve_signal() -> dict[str, Any]:
    yield_data = _download_batch(list(YIELD_TICKERS.values()), period="10mo", interval="1d")
    tenor_series: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []

    for tenor, ticker in YIELD_TICKERS.items():
        frame = _extract_ticker_frame(yield_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if close.empty:
            continue
        yield_pct = close / 10.0 if ticker in YIELD_INDEX_TICKERS else close
        yield_pct.name = tenor
        tenor_series[tenor] = yield_pct
        rows.append({"tenor": tenor, "yield_pct": float(yield_pct.iloc[-1]), "source": ticker})

    yields_table = pd.DataFrame(rows)
    if not yields_table.empty:
        order = {"13W": 1, "5Y": 2, "10Y": 3, "30Y": 4}
        yields_table["_order"] = yields_table["tenor"].map(lambda x: order.get(str(x), 99))
        yields_table = yields_table.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    spread_history = pd.DataFrame()
    current_spreads: dict[str, float] = {}

    if tenor_series:
        curve = pd.concat(tenor_series.values(), axis=1).dropna(how="all").sort_index()
        spread_history = pd.DataFrame(index=curve.index)

        if {"13W", "10Y"}.issubset(curve.columns):
            s = (curve["10Y"] - curve["13W"]) * 100
            spread_history["s13w10y_bps"] = s
            current_spreads["13W-10Y"] = float(s.dropna().iloc[-1]) if not s.dropna().empty else np.nan

        if {"5Y", "30Y"}.issubset(curve.columns):
            s = (curve["30Y"] - curve["5Y"]) * 100
            spread_history["s5s30_bps"] = s
            current_spreads["5Y-30Y"] = float(s.dropna().iloc[-1]) if not s.dropna().empty else np.nan

        if {"10Y", "30Y"}.issubset(curve.columns):
            s = (curve["30Y"] - curve["10Y"]) * 100
            spread_history["s10s30_bps"] = s
            current_spreads["10Y-30Y"] = float(s.dropna().iloc[-1]) if not s.dropna().empty else np.nan

    primary_name = None
    for name in ["13W-10Y", "5Y-30Y", "10Y-30Y"]:
        if name in current_spreads and not np.isnan(current_spreads[name]):
            primary_name = name
            break

    primary_spread = current_spreads.get(primary_name, np.nan) if primary_name else np.nan

    week_delta = np.nan
    month_delta = np.nan
    if primary_name and not spread_history.empty:
        col = {"13W-10Y": "s13w10y_bps", "5Y-30Y": "s5s30_bps", "10Y-30Y": "s10s30_bps"}.get(primary_name)
        if col and col in spread_history:
            s = spread_history[col].dropna()
            if len(s) > 5:
                week_delta = float(s.iloc[-1] - s.iloc[-6])
            if len(s) > 21:
                month_delta = float(s.iloc[-1] - s.iloc[-22])

    futures_data = _download_batch(list(TREASURY_FUTURES.keys()), period="8mo", interval="1d")
    fut_rows: list[dict[str, Any]] = []
    for ticker, label in TREASURY_FUTURES.items():
        frame = _extract_ticker_frame(futures_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if len(close) < 25:
            continue
        last = float(close.iloc[-1])
        w1 = _safe_return(close, 5)
        m1 = _safe_return(close, 21)
        ma10 = float(close.tail(10).mean())
        ma30 = float(close.tail(30).mean()) if len(close) >= 30 else np.nan
        if not np.isnan(ma30) and last > ma10 > ma30:
            trend = "Uptrend (bond prices up / yields down)"
        elif not np.isnan(ma30) and last < ma10 < ma30:
            trend = "Downtrend (bond prices down / yields up)"
        else:
            trend = "Range / mixed"
        fut_rows.append({
            "contract": label,
            "ticker": ticker,
            "last": last,
            "weekly_return": w1,
            "monthly_return": m1,
            "trend": trend,
        })

    futures_table = pd.DataFrame(fut_rows)

    if not spread_history.empty:
        spread_history = spread_history.reset_index()
        date_col = spread_history.columns[0]
        spread_history = spread_history.rename(columns={date_col: "date"})

    return {
        "curve_state": _classify_curve_state(primary_spread),
        "primary_spread_name": primary_name,
        "primary_spread_bps": primary_spread,
        "week_delta_bps": week_delta,
        "month_delta_bps": month_delta,
        "week_trend": _classify_shift(week_delta),
        "month_trend": _classify_shift(month_delta),
        "current_spreads": current_spreads,
        "yields_table": yields_table,
        "spread_history": spread_history,
        "futures_table": futures_table,
        "front_end_note": "Front-end uses 13W T-bill (^IRX) as proxy for ZT/2Y direction when direct 2Y index is unavailable.",
    }


def _clip_01(value: float) -> float:
    if value is None or np.isnan(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _last_rank(x: pd.Series) -> float:
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=max(20, window // 4)).apply(_last_rank, raw=False)


def _build_capitulation_signal(daily_data: pd.DataFrame) -> dict[str, Any]:
    empty_log = pd.DataFrame(
        columns=[
            "trigger_date",
            "direction",
            "trigger_score",
            "state",
            "confirmation_bars",
            "reversal_return",
        ]
    )
    empty_hist = pd.DataFrame(columns=["date", "downside_score", "upside_score", "close"])

    spy = _extract_ticker_frame(daily_data, "SPY")
    if spy.empty or any(col not in spy.columns for col in ["Close", "High", "Low", "Volume"]):
        return {
            "downside_score": np.nan,
            "upside_score": np.nan,
            "extreme_threshold": 75.0,
            "downside_trigger_today": False,
            "upside_trigger_today": False,
            "downside_status": "Unavailable",
            "upside_status": "Unavailable",
            "signal_log": empty_log,
            "score_history": empty_hist,
            "notes": "Insufficient SPY data for capitulation model.",
        }

    base = pd.concat(
        [
            spy["Close"].rename("close"),
            spy["High"].rename("high"),
            spy["Low"].rename("low"),
            spy["Volume"].replace(0, np.nan).rename("volume"),
        ],
        axis=1,
    ).dropna()

    if len(base) < 70:
        return {
            "downside_score": np.nan,
            "upside_score": np.nan,
            "extreme_threshold": 75.0,
            "downside_trigger_today": False,
            "upside_trigger_today": False,
            "downside_status": "Unavailable",
            "upside_status": "Unavailable",
            "signal_log": empty_log,
            "score_history": empty_hist,
            "notes": "Not enough history for capitulation model windows.",
        }

    base["ret"] = base["close"].pct_change()
    base["range_pct"] = (base["high"] - base["low"]) / base["close"].shift(1)
    base["atr14"] = (base["high"] - base["low"]).rolling(14).mean()
    base["ma20"] = base["close"].rolling(20).mean()
    base["stretch_atr"] = (base["close"] - base["ma20"]) / base["atr14"].replace(0, np.nan)

    base["ret_z"] = _rolling_z(base["ret"], 20)
    base["vol_z"] = _rolling_z(np.log(base["volume"]), 20)
    base["range_z"] = _rolling_z(base["range_pct"], 20)

    sector_rets: list[pd.Series] = []
    for ticker in sorted(SECTOR_ETFS.keys()):
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        s = frame["Close"].pct_change().rename(ticker)
        sector_rets.append(s)

    breadth = pd.Series(np.nan, index=base.index)
    if sector_rets:
        breadth_panel = pd.concat(sector_rets, axis=1)
        breadth_raw = (breadth_panel > 0).sum(axis=1) / breadth_panel.notna().sum(axis=1)
        breadth = breadth_raw.reindex(base.index)

    vix_rank = pd.Series(np.nan, index=base.index)
    vix_frame = _extract_ticker_frame(daily_data, "^VIX")
    if not vix_frame.empty and "Close" in vix_frame:
        vix = vix_frame["Close"].dropna()
        vix_rank = _rolling_percentile(vix, 252).reindex(base.index)

    down_ret = ((-base["ret_z"] - 0.8) / 2.2).clip(0, 1)
    up_ret = ((base["ret_z"] - 0.8) / 2.2).clip(0, 1)
    vol_ext = ((base["vol_z"] - 0.8) / 2.2).clip(0, 1)
    range_ext = ((base["range_z"] - 0.8) / 2.2).clip(0, 1)

    down_breadth = ((0.35 - breadth) / 0.35).clip(0, 1)
    up_breadth = ((breadth - 0.65) / 0.35).clip(0, 1)

    down_stretch = ((-base["stretch_atr"] - 1.2) / 2.5).clip(0, 1)
    up_stretch = ((base["stretch_atr"] - 1.2) / 2.5).clip(0, 1)

    vix_high = ((vix_rank - 0.65) / 0.35).clip(0, 1)
    vix_low = ((0.35 - vix_rank) / 0.35).clip(0, 1)

    downside_score = (
        0.24 * down_ret
        + 0.18 * vol_ext
        + 0.16 * range_ext
        + 0.16 * down_breadth
        + 0.16 * down_stretch
        + 0.10 * vix_high
    ) * 100.0

    upside_score = (
        0.24 * up_ret
        + 0.18 * vol_ext
        + 0.16 * range_ext
        + 0.16 * up_breadth
        + 0.16 * up_stretch
        + 0.10 * vix_low
    ) * 100.0

    extreme_threshold = 60.0
    down_trigger = (downside_score >= extreme_threshold) & ((downside_score.shift(1) < extreme_threshold) | downside_score.shift(1).isna())
    up_trigger = (upside_score >= extreme_threshold) & ((upside_score.shift(1) < extreme_threshold) | upside_score.shift(1).isna())

    close = base["close"]
    trigger_rows: list[dict[str, Any]] = []

    def _evaluate_trigger(idx: int, side: str, score_series: pd.Series) -> dict[str, Any]:
        n = len(close)
        trigger_date = close.index[idx]
        trigger_close = float(close.iloc[idx])

        if idx >= n - 1:
            return {
                "trigger_date": trigger_date,
                "direction": side,
                "trigger_score": float(score_series.iloc[idx]),
                "state": "pending",
                "confirmation_bars": np.nan,
                "reversal_return": np.nan,
            }

        lookahead = close.iloc[idx + 1 : min(n, idx + 4)]
        rel = (lookahead / trigger_close) - 1.0

        if side == "Downside Capitulation":
            confirmed = rel[rel >= 0.006]
        else:
            confirmed = rel[rel <= -0.006]

        if not confirmed.empty:
            first_bar = confirmed.index[0]
            bars = int(lookahead.index.get_loc(first_bar) + 1)
            return {
                "trigger_date": trigger_date,
                "direction": side,
                "trigger_score": float(score_series.iloc[idx]),
                "state": "confirmed",
                "confirmation_bars": bars,
                "reversal_return": float(confirmed.iloc[0]),
            }

        if idx + 3 < n:
            final_rel = float(rel.iloc[-1]) if not rel.empty else np.nan
            return {
                "trigger_date": trigger_date,
                "direction": side,
                "trigger_score": float(score_series.iloc[idx]),
                "state": "failed",
                "confirmation_bars": 3,
                "reversal_return": final_rel,
            }

        return {
            "trigger_date": trigger_date,
            "direction": side,
            "trigger_score": float(score_series.iloc[idx]),
            "state": "pending",
            "confirmation_bars": np.nan,
            "reversal_return": float(rel.iloc[-1]) if not rel.empty else np.nan,
        }

    down_positions = np.where(down_trigger.fillna(False).values)[0]
    for pos in down_positions:
        trigger_rows.append(_evaluate_trigger(int(pos), "Downside Capitulation", downside_score))

    up_positions = np.where(up_trigger.fillna(False).values)[0]
    for pos in up_positions:
        trigger_rows.append(_evaluate_trigger(int(pos), "Upside Exhaustion", upside_score))

    log_df = pd.DataFrame(trigger_rows)
    if not log_df.empty:
        log_df = log_df.sort_values("trigger_date", ascending=False).reset_index(drop=True)

    def _latest_status(direction: str, triggered_today: bool) -> str:
        if triggered_today:
            return "Triggered today - awaiting 1-3 bar reversal confirmation"
        if log_df.empty:
            return "No recent trigger"
        subset = log_df[log_df["direction"] == direction]
        if subset.empty:
            return "No recent trigger"
        row = subset.iloc[0]
        state = str(row["state"])
        if state == "confirmed":
            return f"Last trigger confirmed in {int(row['confirmation_bars'])} bars"
        if state == "failed":
            return "Last trigger failed confirmation"
        return "Prior trigger pending confirmation"

    hist = pd.DataFrame(
        {
            "date": base.index,
            "downside_score": downside_score.values,
            "upside_score": upside_score.values,
            "close": base["close"].values,
        }
    ).dropna(subset=["downside_score", "upside_score"])

    downside_trigger_today = bool(down_trigger.iloc[-1]) if len(down_trigger) else False
    upside_trigger_today = bool(up_trigger.iloc[-1]) if len(up_trigger) else False

    down_val = float(downside_score.iloc[-1]) if len(downside_score.dropna()) else np.nan
    up_val = float(upside_score.iloc[-1]) if len(upside_score.dropna()) else np.nan

    return {
        "downside_score": down_val,
        "upside_score": up_val,
        "extreme_threshold": extreme_threshold,
        "downside_trigger_today": downside_trigger_today,
        "upside_trigger_today": upside_trigger_today,
        "downside_status": _latest_status("Downside Capitulation", downside_trigger_today),
        "upside_status": _latest_status("Upside Exhaustion", upside_trigger_today),
        "signal_log": log_df,
        "score_history": hist.tail(180).reset_index(drop=True),
        "notes": "Trigger requires extreme score today; confirmation requires reversal move within next 1-3 bars.",
    }


def _build_sector_capitulation_signals(daily_data: pd.DataFrame) -> dict[str, Any]:
    threshold = 60.0
    summary_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for ticker, sector_name in SECTOR_ETFS.items():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or any(col not in frame.columns for col in ["Close", "High", "Low", "Volume"]):
            continue

        base = pd.concat(
            [
                frame["Close"].rename("close"),
                frame["High"].rename("high"),
                frame["Low"].rename("low"),
                frame["Volume"].replace(0, np.nan).rename("volume"),
            ],
            axis=1,
        ).dropna()

        if len(base) < 70:
            continue

        base["ret"] = base["close"].pct_change()
        base["range_pct"] = (base["high"] - base["low"]) / base["close"].shift(1)
        base["atr14"] = (base["high"] - base["low"]).rolling(14).mean()
        base["ma20"] = base["close"].rolling(20).mean()
        base["stretch_atr"] = (base["close"] - base["ma20"]) / base["atr14"].replace(0, np.nan)
        base["ret_z"] = _rolling_z(base["ret"], 20)
        base["vol_z"] = _rolling_z(np.log(base["volume"]), 20)
        base["range_z"] = _rolling_z(base["range_pct"], 20)

        down_ret = ((-base["ret_z"] - 0.8) / 2.2).clip(0, 1)
        up_ret = ((base["ret_z"] - 0.8) / 2.2).clip(0, 1)
        vol_ext = ((base["vol_z"] - 0.8) / 2.2).clip(0, 1)
        range_ext = ((base["range_z"] - 0.8) / 2.2).clip(0, 1)
        down_stretch = ((-base["stretch_atr"] - 1.2) / 2.5).clip(0, 1)
        up_stretch = ((base["stretch_atr"] - 1.2) / 2.5).clip(0, 1)

        downside_score = (
            0.36 * down_ret
            + 0.24 * vol_ext
            + 0.20 * range_ext
            + 0.20 * down_stretch
        ) * 100.0

        upside_score = (
            0.36 * up_ret
            + 0.24 * vol_ext
            + 0.20 * range_ext
            + 0.20 * up_stretch
        ) * 100.0

        down_trigger = (downside_score >= threshold) & ((downside_score.shift(1) < threshold) | downside_score.shift(1).isna())
        up_trigger = (upside_score >= threshold) & ((upside_score.shift(1) < threshold) | upside_score.shift(1).isna())

        close = base["close"]

        def _evaluate(idx: int, direction: str, score_series: pd.Series) -> dict[str, Any]:
            n = len(close)
            trigger_date = close.index[idx]
            trigger_close = float(close.iloc[idx])

            if idx >= n - 1:
                return {
                    "trigger_date": trigger_date,
                    "sector": sector_name,
                    "ticker": ticker,
                    "direction": direction,
                    "trigger_score": float(score_series.iloc[idx]),
                    "state": "pending",
                    "confirmation_bars": np.nan,
                    "reversal_return": np.nan,
                }

            lookahead = close.iloc[idx + 1 : min(n, idx + 4)]
            rel = (lookahead / trigger_close) - 1.0
            if direction == "Downside Capitulation":
                confirmed = rel[rel >= 0.006]
            else:
                confirmed = rel[rel <= -0.006]

            if not confirmed.empty:
                first_bar = confirmed.index[0]
                bars = int(lookahead.index.get_loc(first_bar) + 1)
                return {
                    "trigger_date": trigger_date,
                    "sector": sector_name,
                    "ticker": ticker,
                    "direction": direction,
                    "trigger_score": float(score_series.iloc[idx]),
                    "state": "confirmed",
                    "confirmation_bars": bars,
                    "reversal_return": float(confirmed.iloc[0]),
                }

            if idx + 3 < n:
                return {
                    "trigger_date": trigger_date,
                    "sector": sector_name,
                    "ticker": ticker,
                    "direction": direction,
                    "trigger_score": float(score_series.iloc[idx]),
                    "state": "failed",
                    "confirmation_bars": 3,
                    "reversal_return": float(rel.iloc[-1]) if not rel.empty else np.nan,
                }

            return {
                "trigger_date": trigger_date,
                "sector": sector_name,
                "ticker": ticker,
                "direction": direction,
                "trigger_score": float(score_series.iloc[idx]),
                "state": "pending",
                "confirmation_bars": np.nan,
                "reversal_return": float(rel.iloc[-1]) if not rel.empty else np.nan,
            }

        down_positions = np.where(down_trigger.fillna(False).values)[0]
        for pos in down_positions:
            log_rows.append(_evaluate(int(pos), "Downside Capitulation", downside_score))

        up_positions = np.where(up_trigger.fillna(False).values)[0]
        for pos in up_positions:
            log_rows.append(_evaluate(int(pos), "Upside Exhaustion", upside_score))

        down_today = bool(down_trigger.iloc[-1]) if len(down_trigger) else False
        up_today = bool(up_trigger.iloc[-1]) if len(up_trigger) else False

        def _status(direction: str, triggered_today: bool) -> str:
            if triggered_today:
                return "Triggered today"
            local = [r for r in log_rows if r["sector"] == sector_name and r["direction"] == direction]
            if not local:
                return "No recent trigger"
            latest = sorted(local, key=lambda x: x["trigger_date"], reverse=True)[0]
            state = str(latest["state"])
            if state == "confirmed":
                return f"Confirmed in {int(latest['confirmation_bars'])} bars"
            if state == "failed":
                return "Failed confirmation"
            return "Pending confirmation"

        summary_rows.append(
            {
                "sector": sector_name,
                "ticker": ticker,
                "downside_score": float(downside_score.iloc[-1]) if len(downside_score.dropna()) else np.nan,
                "upside_score": float(upside_score.iloc[-1]) if len(upside_score.dropna()) else np.nan,
                "downside_trigger_today": down_today,
                "upside_trigger_today": up_today,
                "downside_status": _status("Downside Capitulation", down_today),
                "upside_status": _status("Upside Exhaustion", up_today),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["downside_score", "upside_score"], ascending=False).reset_index(drop=True)

    log_df = pd.DataFrame(log_rows)
    if not log_df.empty:
        log_df = log_df.sort_values("trigger_date", ascending=False).reset_index(drop=True)

    return {
        "threshold": threshold,
        "table": summary_df,
        "signal_log": log_df,
        "notes": "Sector triggers require score >= threshold today and reversal confirmation within 1-3 bars.",
    }


def _load_watchlist_tickers() -> list[str]:
    raw = os.getenv("WATCHLIST_TICKERS", ",".join(DEFAULT_WATCHLIST))
    items: list[str] = []
    for token in raw.split(","):
        ticker = token.strip().upper()
        if ticker and ticker not in items:
            items.append(ticker)
    return items[:30]


def _daily_change_for_ticker(snapshot: dict[str, dict[str, float]], daily_data: pd.DataFrame, ticker: str) -> float:
    snap = snapshot.get(ticker, {}).get("change_pct", np.nan)
    if snap is not None and not np.isnan(snap):
        return float(snap)
    frame = _extract_ticker_frame(daily_data, ticker)
    if frame.empty or "Close" not in frame:
        return np.nan
    close = frame["Close"].dropna()
    if len(close) < 2:
        return np.nan
    prev = float(close.iloc[-2])
    if prev == 0:
        return np.nan
    return float(close.iloc[-1] / prev - 1.0)


def _ticker_label(ticker: str) -> str:
    return (
        MARKET_TICKERS.get(ticker)
        or CTA_PROXY_TICKERS.get(ticker)
        or SECTOR_ETFS.get(ticker)
        or TREASURY_FUTURES.get(ticker)
        or ticker
    )


def _build_watchlist_monitor(daily_data: pd.DataFrame) -> dict[str, Any]:
    threshold_pct = abs(float(os.getenv("WATCHLIST_MOVE_PCT", "1.5")))
    threshold = threshold_pct / 100.0
    rows: list[dict[str, Any]] = []

    for ticker in _load_watchlist_tickers():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if len(close) < 2:
            continue
        daily = _safe_return(close, 1)
        weekly = _safe_return(close, 5)
        monthly = _safe_return(close, 21)
        last = float(close.iloc[-1])
        rows.append(
            {
                "ticker": ticker,
                "label": _ticker_label(ticker),
                "last": last,
                "daily": daily,
                "weekly": weekly,
                "monthly": monthly,
                "is_mover": bool(pd.notna(daily) and abs(daily) >= threshold),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table["abs_daily"] = table["daily"].abs()
        table = table.sort_values("abs_daily", ascending=False).drop(columns=["abs_daily"]).reset_index(drop=True)

    movers = table[table["is_mover"]].copy().reset_index(drop=True) if not table.empty else pd.DataFrame()
    return {
        "threshold_pct": threshold_pct,
        "table": table,
        "movers": movers,
        "notes": "Watchlist movers trigger when absolute daily move exceeds threshold.",
    }


def _build_sector_heatmap_flow(daily_data: pd.DataFrame, sector_returns: pd.DataFrame) -> dict[str, Any]:
    if sector_returns.empty:
        return {
            "heatmap_table": pd.DataFrame(),
            "flow_shift_score": np.nan,
            "flow_regime": "Unavailable",
            "turnover_ratio": np.nan,
            "dispersion": np.nan,
            "leaders_today": pd.DataFrame(),
            "leaders_prev": pd.DataFrame(),
            "notes": "Sector flow model unavailable.",
        }

    heatmap = sector_returns[["sector", "ticker", "daily", "weekly", "monthly"]].copy()

    returns_panel: list[pd.Series] = []
    for ticker, sector_name in SECTOR_ETFS.items():
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        ret = frame["Close"].pct_change().rename(sector_name)
        returns_panel.append(ret)

    if not returns_panel:
        return {
            "heatmap_table": heatmap,
            "flow_shift_score": np.nan,
            "flow_regime": "Unavailable",
            "turnover_ratio": np.nan,
            "dispersion": np.nan,
            "leaders_today": pd.DataFrame(),
            "leaders_prev": pd.DataFrame(),
            "notes": "Not enough sector return history for flow shift model.",
        }

    panel = pd.concat(returns_panel, axis=1).dropna(how="all")
    if len(panel) < 3:
        return {
            "heatmap_table": heatmap,
            "flow_shift_score": np.nan,
            "flow_regime": "Unavailable",
            "turnover_ratio": np.nan,
            "dispersion": np.nan,
            "leaders_today": pd.DataFrame(),
            "leaders_prev": pd.DataFrame(),
            "notes": "Not enough sector return history for flow shift model.",
        }

    today = panel.iloc[-1].dropna().sort_values(ascending=False)
    prev = panel.iloc[-2].dropna().sort_values(ascending=False)
    top_n = min(3, len(today), len(prev))

    top_today = list(today.head(top_n).index)
    top_prev = list(prev.head(top_n).index)

    overlap = len(set(top_today).intersection(set(top_prev)))
    turnover_ratio = 1.0 - (overlap / max(1, top_n))
    dispersion = float(today.max() - today.min()) if not today.empty else np.nan

    flow_shift_score = float(np.clip(turnover_ratio * 70 + min(30, max(0.0, dispersion) * 900), 0, 100))
    if flow_shift_score >= 70:
        flow_regime = "Rapid leadership rotation"
    elif flow_shift_score >= 45:
        flow_regime = "Moderate rotation"
    else:
        flow_regime = "Stable leadership"

    leaders_today = pd.DataFrame({"sector": today.head(3).index, "daily": today.head(3).values})
    leaders_prev = pd.DataFrame({"sector": prev.head(3).index, "daily": prev.head(3).values})

    return {
        "heatmap_table": heatmap.sort_values("daily", ascending=False).reset_index(drop=True),
        "flow_shift_score": flow_shift_score,
        "flow_regime": flow_regime,
        "turnover_ratio": turnover_ratio,
        "dispersion": dispersion,
        "leaders_today": leaders_today,
        "leaders_prev": leaders_prev,
        "notes": "Flow shift combines top-sector turnover and cross-sector return dispersion.",
    }


def _forward_return_profile(close: pd.Series, trigger_date: Any, direction: str) -> dict[str, float]:
    result = {
        "ret_1d": np.nan,
        "ret_3d": np.nan,
        "ret_5d": np.nan,
        "aligned_1d": np.nan,
        "aligned_3d": np.nan,
        "aligned_5d": np.nan,
        "max_adverse_5d": np.nan,
    }

    close = close.dropna()
    if close.empty:
        return result

    ts = pd.to_datetime(trigger_date, errors="coerce")
    if pd.isna(ts):
        return result

    try:
        loc = close.index.get_loc(ts)
        if isinstance(loc, slice):
            idx = int(loc.start)
        elif isinstance(loc, (np.ndarray, list)):
            idx = int(loc[0])
        else:
            idx = int(loc)
    except KeyError:
        idx = int(close.index.searchsorted(ts))
        if idx >= len(close):
            return result

    trigger_close = float(close.iloc[idx])
    if trigger_close == 0:
        return result

    is_downside = str(direction).lower().startswith("downside")

    for h in (1, 3, 5):
        if idx + h < len(close):
            ret = float(close.iloc[idx + h] / trigger_close - 1.0)
            result[f"ret_{h}d"] = ret
            result[f"aligned_{h}d"] = ret if is_downside else -ret

    lookahead = close.iloc[idx + 1 : min(len(close), idx + 6)]
    if not lookahead.empty:
        path_ret = lookahead / trigger_close - 1.0
        aligned_path = path_ret if is_downside else -path_ret
        result["max_adverse_5d"] = float(aligned_path.min())

    return result


def _build_signal_quality_tracker(
    daily_data: pd.DataFrame,
    capitulation: dict[str, Any],
    sector_capitulation: dict[str, Any],
) -> dict[str, Any]:
    trade_rows: list[dict[str, Any]] = []

    def _append_from_log(log_df: pd.DataFrame, source: str) -> None:
        if not isinstance(log_df, pd.DataFrame) or log_df.empty:
            return
        for _, row in log_df.iterrows():
            ticker = str(row.get("ticker") or "SPY")
            frame = _extract_ticker_frame(daily_data, ticker)
            if frame.empty or "Close" not in frame:
                continue
            close = frame["Close"].dropna()
            profile = _forward_return_profile(close, row.get("trigger_date"), str(row.get("direction", "")))
            trade_rows.append(
                {
                    "source": source,
                    "sector": row.get("sector", "Index"),
                    "ticker": ticker,
                    "direction": row.get("direction"),
                    "trigger_date": row.get("trigger_date"),
                    "trigger_score": row.get("trigger_score", np.nan),
                    "state": row.get("state", "unknown"),
                    "ret_1d": profile["ret_1d"],
                    "ret_3d": profile["ret_3d"],
                    "ret_5d": profile["ret_5d"],
                    "aligned_1d": profile["aligned_1d"],
                    "aligned_3d": profile["aligned_3d"],
                    "aligned_5d": profile["aligned_5d"],
                    "max_adverse_5d": profile["max_adverse_5d"],
                }
            )

    _append_from_log(capitulation.get("signal_log", pd.DataFrame()), "Index")
    _append_from_log(sector_capitulation.get("signal_log", pd.DataFrame()), "Sector")

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        return {
            "summary": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "overall_hit_rate_5d": np.nan,
            "edge_state": "Unavailable",
            "notes": "Signal quality tracker needs at least one historical trigger.",
        }

    eval_5d = trades.dropna(subset=["aligned_5d"]).copy()
    overall_hit = float((eval_5d["aligned_5d"] > 0).mean() * 100) if not eval_5d.empty else np.nan

    grouped = []
    for (source, direction), grp in trades.groupby(["source", "direction"], dropna=False):
        row = {
            "source": source,
            "direction": direction,
            "signals": int(len(grp)),
            "evaluated_5d": int(grp["aligned_5d"].notna().sum()),
            "hit_rate_1d": float((grp["aligned_1d"] > 0).mean() * 100) if grp["aligned_1d"].notna().any() else np.nan,
            "hit_rate_3d": float((grp["aligned_3d"] > 0).mean() * 100) if grp["aligned_3d"].notna().any() else np.nan,
            "hit_rate_5d": float((grp["aligned_5d"] > 0).mean() * 100) if grp["aligned_5d"].notna().any() else np.nan,
            "avg_aligned_5d": float(grp["aligned_5d"].mean() * 100) if grp["aligned_5d"].notna().any() else np.nan,
            "avg_max_adverse_5d": float(grp["max_adverse_5d"].mean() * 100) if grp["max_adverse_5d"].notna().any() else np.nan,
            "avg_trigger_score": float(grp["trigger_score"].mean()) if grp["trigger_score"].notna().any() else np.nan,
        }
        grouped.append(row)

    summary = pd.DataFrame(grouped).sort_values(["hit_rate_5d", "signals"], ascending=[False, False]).reset_index(drop=True)

    if pd.notna(overall_hit):
        if overall_hit >= 58:
            edge_state = "Positive edge"
        elif overall_hit <= 45:
            edge_state = "Negative edge"
        else:
            edge_state = "Mixed edge"
    else:
        edge_state = "Insufficient 5D outcomes"

    trades = trades.sort_values("trigger_date", ascending=False).reset_index(drop=True)

    return {
        "summary": summary,
        "trades": trades,
        "overall_hit_rate_5d": overall_hit,
        "edge_state": edge_state,
        "notes": "Forward outcome model scores trigger quality over 1D/3D/5D horizons using aligned returns.",
    }


def _build_liquidity_real_rate_monitor(
    daily_data: pd.DataFrame,
    snapshot: dict[str, dict[str, float]],
    volume_profile: dict[str, Any],
) -> dict[str, Any]:
    panel_series: list[pd.Series] = []

    for ticker in sorted(LIQUIDITY_MONITOR_TICKERS):
        frame = _extract_ticker_frame(daily_data, ticker)
        if frame.empty or "Close" not in frame:
            continue
        close = frame["Close"].dropna()
        if close.empty:
            continue
        panel_series.append(close.rename(ticker))

    frame_tnx = _extract_ticker_frame(daily_data, "^TNX")
    nominal_yield = pd.Series(dtype=float)
    if not frame_tnx.empty and "Close" in frame_tnx:
        nominal_yield = (frame_tnx["Close"].dropna() / 10.0).rename("nominal_10y")

    if panel_series:
        base = pd.concat(panel_series, axis=1).sort_index()
    else:
        base = pd.DataFrame()

    if not nominal_yield.empty:
        base = pd.concat([base, nominal_yield], axis=1).sort_index()

    if base.empty:
        return {
            "liquidity_stress_score": np.nan,
            "liquidity_regime": "Unavailable",
            "real_rate_proxy_z": np.nan,
            "real_rate_regime": "Unavailable",
            "nominal_10y_yield": snapshot.get("^TNX", {}).get("last", np.nan) / 10.0 if pd.notna(snapshot.get("^TNX", {}).get("last", np.nan)) else np.nan,
            "daily_delta_score": np.nan,
            "week_delta_score": np.nan,
            "month_delta_score": np.nan,
            "daily_trend": "Unavailable",
            "week_trend": "Unavailable",
            "month_trend": "Unavailable",
            "details_table": pd.DataFrame(),
            "history": pd.DataFrame(),
            "notes": "Insufficient data for liquidity/real-rate monitor.",
        }

    idx = base.index

    tip_ief_ratio = pd.Series(np.nan, index=idx)
    if {"TIP", "IEF"}.issubset(set(base.columns)):
        tip_ief_ratio = (base["TIP"] / base["IEF"]).replace([np.inf, -np.inf], np.nan)

    nominal_series = base["nominal_10y"] if "nominal_10y" in base else pd.Series(np.nan, index=idx)
    nominal_z = _rolling_z(nominal_series, 126)
    breakeven_proxy_z = _rolling_z(tip_ief_ratio, 126)
    real_rate_proxy_z = nominal_z - breakeven_proxy_z

    credit_ratio = pd.Series(np.nan, index=idx)
    if {"LQD", "HYG"}.issubset(set(base.columns)):
        credit_ratio = (base["LQD"] / base["HYG"]).replace([np.inf, -np.inf], np.nan)
    credit_stress_z = _rolling_z(credit_ratio, 126)

    usd_z = _rolling_z(base["UUP"], 126) if "UUP" in base else pd.Series(np.nan, index=idx)

    tlt_vol_z = pd.Series(np.nan, index=idx)
    if "TLT" in base:
        tlt_vol = base["TLT"].pct_change().rolling(20).std() * np.sqrt(252)
        tlt_vol_z = _rolling_z(tlt_vol, 126)

    funding_pressure = pd.Series(np.nan, index=idx)
    if "SHY" in base:
        shy_1m = base["SHY"].pct_change(21)
        funding_pressure = -shy_1m

    spy_volume_ratio = pd.Series(np.nan, index=idx)
    spy = _extract_ticker_frame(daily_data, "SPY")
    if not spy.empty and "Volume" in spy:
        vv = spy["Volume"].replace(0, np.nan)
        spy_volume_ratio = (vv / vv.rolling(20).mean()).reindex(idx)

    credit_comp = ((credit_stress_z + 0.5) / 2.5).clip(0, 1)
    usd_comp = ((usd_z + 0.3) / 2.0).clip(0, 1)
    vol_comp = ((tlt_vol_z + 0.2) / 2.2).clip(0, 1)
    funding_comp = ((funding_pressure - 0.002) / 0.015).clip(0, 1)
    real_rate_comp = ((real_rate_proxy_z + 0.2) / 2.2).clip(0, 1)
    volume_comp = ((spy_volume_ratio - 1.0) / 0.8).clip(0, 1)

    liquidity_score_series = (
        0.27 * credit_comp
        + 0.22 * usd_comp
        + 0.18 * vol_comp
        + 0.13 * funding_comp
        + 0.12 * real_rate_comp
        + 0.08 * volume_comp
    ) * 100.0

    history = pd.DataFrame(
        {
            "date": idx,
            "liquidity_stress_score": liquidity_score_series,
            "real_rate_proxy_z": real_rate_proxy_z,
            "nominal_10y_yield": nominal_series,
            "tip_ief_ratio": tip_ief_ratio,
            "credit_stress_z": credit_stress_z,
            "usd_z": usd_z,
        }
    ).dropna(how="all", subset=["liquidity_stress_score", "real_rate_proxy_z", "nominal_10y_yield"])

    def _last(series: pd.Series) -> float:
        s = series.dropna()
        return float(s.iloc[-1]) if not s.empty else np.nan

    liq_score = _last(liquidity_score_series)
    real_z = _last(real_rate_proxy_z)
    nominal_now = _last(nominal_series)
    tip_ief_now = _last(tip_ief_ratio)
    credit_now = _last(credit_stress_z)
    usd_now = _last(usd_z)
    tlt_vol_now = _last(tlt_vol_z)
    funding_now = _last(funding_pressure)
    vol_ratio_now = _last(spy_volume_ratio)

    day_delta = np.nan
    week_delta = np.nan
    month_delta = np.nan
    s_liq = liquidity_score_series.dropna()
    if len(s_liq) > 1:
        day_delta = float(s_liq.iloc[-1] - s_liq.iloc[-2])
    if len(s_liq) > 5:
        week_delta = float(s_liq.iloc[-1] - s_liq.iloc[-6])
    if len(s_liq) > 21:
        month_delta = float(s_liq.iloc[-1] - s_liq.iloc[-22])

    def _trend(delta: float) -> str:
        if pd.isna(delta):
            return "Unavailable"
        if delta >= 8:
            return "Tightening fast"
        if delta >= 3:
            return "Tightening"
        if delta <= -8:
            return "Easing fast"
        if delta <= -3:
            return "Easing"
        return "Mostly stable"

    if pd.isna(liq_score):
        liquidity_regime = "Unavailable"
    elif liq_score >= 70:
        liquidity_regime = "Severe tightness"
    elif liq_score >= 50:
        liquidity_regime = "Tight"
    elif liq_score >= 32:
        liquidity_regime = "Neutral"
    else:
        liquidity_regime = "Easy"

    if pd.isna(real_z):
        real_regime = "Unavailable"
    elif real_z >= 1.0:
        real_regime = "Restrictive real-rate impulse"
    elif real_z >= 0.35:
        real_regime = "Mildly restrictive"
    elif real_z <= -1.0:
        real_regime = "Accommodative real-rate impulse"
    elif real_z <= -0.35:
        real_regime = "Mildly accommodative"
    else:
        real_regime = "Neutral real-rate impulse"

    details = pd.DataFrame(
        [
            {
                "metric": "10Y nominal yield",
                "value": nominal_now,
                "unit": "%",
                "signal": "Higher can tighten financial conditions" if pd.notna(nominal_now) else "Unavailable",
            },
            {
                "metric": "TIP/IEF ratio",
                "value": tip_ief_now,
                "unit": "ratio",
                "signal": "Breakeven inflation proxy (higher = inflation expectations firmer)" if pd.notna(tip_ief_now) else "Unavailable",
            },
            {
                "metric": "Real-rate proxy (z)",
                "value": real_z,
                "unit": "z",
                "signal": real_regime,
            },
            {
                "metric": "Credit stress (LQD/HYG z)",
                "value": credit_now,
                "unit": "z",
                "signal": "Higher = tighter credit risk appetite" if pd.notna(credit_now) else "Unavailable",
            },
            {
                "metric": "USD liquidity (UUP z)",
                "value": usd_now,
                "unit": "z",
                "signal": "Higher = global dollar tightness" if pd.notna(usd_now) else "Unavailable",
            },
            {
                "metric": "Duration vol (TLT vol z)",
                "value": tlt_vol_now,
                "unit": "z",
                "signal": "Higher = rates volatility stress" if pd.notna(tlt_vol_now) else "Unavailable",
            },
            {
                "metric": "Funding pressure (-SHY 1M)",
                "value": funding_now,
                "unit": "%",
                "signal": "Higher = front-end pressure rising" if pd.notna(funding_now) else "Unavailable",
            },
            {
                "metric": "SPY volume / 20D",
                "value": vol_ratio_now,
                "unit": "x",
                "signal": "Higher = stress-driven volume more likely" if pd.notna(vol_ratio_now) else "Unavailable",
            },
        ]
    )

    return {
        "liquidity_stress_score": liq_score,
        "liquidity_regime": liquidity_regime,
        "real_rate_proxy_z": real_z,
        "real_rate_regime": real_regime,
        "nominal_10y_yield": nominal_now,
        "daily_delta_score": day_delta,
        "week_delta_score": week_delta,
        "month_delta_score": month_delta,
        "daily_trend": _trend(day_delta),
        "week_trend": _trend(week_delta),
        "month_trend": _trend(month_delta),
        "details_table": details,
        "history": history.tail(220).reset_index(drop=True),
        "notes": "Real-rate signal is a proxy: 10Y nominal yield z-score minus TIP/IEF breakeven proxy z-score.",
    }


def _pick_option_iv_near_strike(df: pd.DataFrame, spot: float, target_ratio: float) -> tuple[float, float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or pd.isna(spot) or spot <= 0:
        return np.nan, np.nan
    if "strike" not in df.columns or "impliedVolatility" not in df.columns:
        return np.nan, np.nan

    work = df.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["impliedVolatility"] = pd.to_numeric(work["impliedVolatility"], errors="coerce")
    if "openInterest" in work.columns:
        work["openInterest"] = pd.to_numeric(work["openInterest"], errors="coerce").fillna(0)
    else:
        work["openInterest"] = 0.0
    if "volume" in work.columns:
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
    else:
        work["volume"] = 0.0

    work = work.dropna(subset=["strike", "impliedVolatility"])
    work = work[work["impliedVolatility"] > 0]
    if work.empty:
        return np.nan, np.nan

    work["distance"] = (work["strike"] / float(spot) - target_ratio).abs()
    row = work.sort_values(["distance", "openInterest", "volume"], ascending=[True, False, False]).iloc[0]
    return float(row["impliedVolatility"]), float(row["strike"])


def _build_put_skew_monitor(daily_data: pd.DataFrame, snapshot: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    today = datetime.now(US_EASTERN).date()

    for symbol in PUT_SKEW_UNDERLYINGS:
        frame = _extract_ticker_frame(daily_data, symbol)
        spot = np.nan
        if not frame.empty and "Close" in frame.columns:
            close = frame["Close"].dropna()
            if not close.empty:
                spot = float(close.iloc[-1])
        if pd.isna(spot):
            spot = snapshot.get(symbol, {}).get("last", np.nan)

        status = "Unavailable"
        chosen_exp = None
        dte = np.nan
        put_iv = np.nan
        call_iv = np.nan
        put_strike = np.nan
        call_strike = np.nan
        oi_ratio = np.nan
        vol_ratio = np.nan
        skew = np.nan

        try:
            ticker = yf.Ticker(symbol)
            expiries = ticker.options or []
        except Exception:
            expiries = []

        exp_pairs: list[tuple[str, int]] = []
        for exp in expiries:
            try:
                exp_date = datetime.strptime(str(exp), "%Y-%m-%d").date()
                days = (exp_date - today).days
                if days >= 0:
                    exp_pairs.append((str(exp), int(days)))
            except Exception:
                continue

        if exp_pairs:
            near = [x for x in exp_pairs if 7 <= x[1] <= 45]
            if near:
                chosen_exp, dte = sorted(near, key=lambda x: x[1])[0]
            else:
                chosen_exp, dte = sorted(exp_pairs, key=lambda x: x[1])[0]

        if chosen_exp:
            try:
                chain = yf.Ticker(symbol).option_chain(chosen_exp)
                puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()
                calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()

                put_iv, put_strike = _pick_option_iv_near_strike(puts, float(spot), 0.95)
                call_iv, call_strike = _pick_option_iv_near_strike(calls, float(spot), 1.05)

                if isinstance(puts, pd.DataFrame) and not puts.empty and "openInterest" in puts.columns and isinstance(calls, pd.DataFrame) and not calls.empty and "openInterest" in calls.columns:
                    p_oi = pd.to_numeric(puts["openInterest"], errors="coerce").fillna(0).sum()
                    c_oi = pd.to_numeric(calls["openInterest"], errors="coerce").fillna(0).sum()
                    oi_ratio = float(p_oi / c_oi) if c_oi > 0 else np.nan

                if isinstance(puts, pd.DataFrame) and not puts.empty and "volume" in puts.columns and isinstance(calls, pd.DataFrame) and not calls.empty and "volume" in calls.columns:
                    p_vol = pd.to_numeric(puts["volume"], errors="coerce").fillna(0).sum()
                    c_vol = pd.to_numeric(calls["volume"], errors="coerce").fillna(0).sum()
                    vol_ratio = float(p_vol / c_vol) if c_vol > 0 else np.nan

                if pd.notna(put_iv) and pd.notna(call_iv):
                    skew = float((put_iv - call_iv) * 100.0)
                    if skew >= 6:
                        status = "Severe downside skew"
                    elif skew >= 3:
                        status = "Elevated downside skew"
                    elif skew <= -1:
                        status = "Call skew / upside chase"
                    else:
                        status = "Neutral skew"
                else:
                    status = "Partial chain data"
            except Exception:
                status = "Chain fetch failed"

        rows.append(
            {
                "symbol": symbol,
                "spot": float(spot) if pd.notna(spot) else np.nan,
                "expiry": chosen_exp,
                "dte": dte,
                "put_iv_95": put_iv,
                "call_iv_105": call_iv,
                "put_strike": put_strike,
                "call_strike": call_strike,
                "put_call_iv_spread": skew,
                "put_call_oi_ratio": oi_ratio,
                "put_call_vol_ratio": vol_ratio,
                "status": status,
            }
        )

    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["put_call_iv_spread"]) if not table.empty else pd.DataFrame()

    avg_skew = float(valid["put_call_iv_spread"].mean()) if not valid.empty else np.nan
    avg_oi = float(table["put_call_oi_ratio"].dropna().mean()) if not table.empty and table["put_call_oi_ratio"].notna().any() else np.nan
    avg_vol = float(table["put_call_vol_ratio"].dropna().mean()) if not table.empty and table["put_call_vol_ratio"].notna().any() else np.nan
    stressed = int((valid["put_call_iv_spread"] >= 3.0).sum()) if not valid.empty else 0

    if pd.isna(avg_skew):
        regime = "Unavailable"
    elif avg_skew >= 6:
        regime = "High crash-hedge demand"
    elif avg_skew >= 3:
        regime = "Moderate downside hedging"
    elif avg_skew <= -1:
        regime = "Upside call demand"
    else:
        regime = "Balanced skew"

    return {
        "table": table,
        "avg_put_call_iv_spread": avg_skew,
        "avg_put_call_oi_ratio": avg_oi,
        "avg_put_call_vol_ratio": avg_vol,
        "stressed_count": stressed,
        "total_symbols": int(len(table)),
        "regime": regime,
        "notes": "Skew proxy uses near-term 95% put IV minus 105% call IV (nearest available strikes).",
    }


def _build_cross_asset_confirmation(
    snapshot: dict[str, dict[str, float]],
    daily_data: pd.DataFrame,
    sentiment: dict[str, Any],
    rotation_signal: dict[str, Any],
    cta_proxy: dict[str, Any],
) -> dict[str, Any]:
    spx_change = snapshot.get("^GSPC", {}).get("change_pct", np.nan)
    sentiment_score = float(sentiment.get("composite_score", 0.0))
    rot_spread = rotation_signal.get("spread_daily", np.nan)
    cta_net = float(cta_proxy.get("net_score", 0.0))

    bias_score = 0.0
    if pd.notna(spx_change):
        if spx_change >= 0.002:
            bias_score += 1.0
        elif spx_change <= -0.002:
            bias_score -= 1.0
    if sentiment_score >= 15:
        bias_score += 1.0
    elif sentiment_score <= -15:
        bias_score -= 1.0
    if pd.notna(rot_spread):
        if rot_spread >= 0.005:
            bias_score += 0.8
        elif rot_spread <= -0.005:
            bias_score -= 0.8
    if cta_net >= 25:
        bias_score += 0.4
    elif cta_net <= -25:
        bias_score -= 0.4

    if bias_score >= 1.0:
        bias = "Bullish"
    elif bias_score <= -1.0:
        bias = "Bearish"
    else:
        bias = "Neutral"

    vix_chg = _daily_change_for_ticker(snapshot, daily_data, "^VIX")
    usd_chg = _daily_change_for_ticker(snapshot, daily_data, "UUP")
    tnx_chg = _daily_change_for_ticker(snapshot, daily_data, "^TNX")
    tlt_chg = _daily_change_for_ticker(snapshot, daily_data, "TLT")
    hyg_chg = _daily_change_for_ticker(snapshot, daily_data, "HYG")
    lqd_chg = _daily_change_for_ticker(snapshot, daily_data, "LQD")

    credit_spread = hyg_chg - lqd_chg if pd.notna(hyg_chg) and pd.notna(lqd_chg) else np.nan

    rows = [
        {
            "asset": "Volatility",
            "signal": "VIX day change",
            "value": vix_chg,
            "supports_bullish": bool(pd.notna(vix_chg) and vix_chg < 0),
            "supports_bearish": bool(pd.notna(vix_chg) and vix_chg > 0),
        },
        {
            "asset": "US Dollar",
            "signal": "UUP day change",
            "value": usd_chg,
            "supports_bullish": bool(pd.notna(usd_chg) and usd_chg < 0),
            "supports_bearish": bool(pd.notna(usd_chg) and usd_chg > 0),
        },
        {
            "asset": "Credit",
            "signal": "HYG-LQD spread",
            "value": credit_spread,
            "supports_bullish": bool(pd.notna(credit_spread) and credit_spread > 0),
            "supports_bearish": bool(pd.notna(credit_spread) and credit_spread < 0),
        },
        {
            "asset": "Rates",
            "signal": "10Y yield proxy (^TNX)",
            "value": tnx_chg,
            "supports_bullish": bool(pd.notna(tnx_chg) and tnx_chg > 0),
            "supports_bearish": bool(pd.notna(tnx_chg) and tnx_chg < 0),
        },
        {
            "asset": "Duration",
            "signal": "TLT day change",
            "value": tlt_chg,
            "supports_bullish": bool(pd.notna(tlt_chg) and tlt_chg < 0),
            "supports_bearish": bool(pd.notna(tlt_chg) and tlt_chg > 0),
        },
    ]

    available = 0
    supportive = 0
    contradictions: list[str] = []

    for row in rows:
        if bias == "Bullish":
            supports_bias = row["supports_bullish"] if pd.notna(row["value"]) else None
        elif bias == "Bearish":
            supports_bias = row["supports_bearish"] if pd.notna(row["value"]) else None
        else:
            supports_bias = None
        row["supports_bias"] = supports_bias

        if supports_bias is not None:
            available += 1
            if supports_bias:
                supportive += 1
            else:
                contradictions.append(str(row["asset"]))

    ratio = (supportive / available) if available > 0 else np.nan
    if bias == "Neutral":
        status = "No directional bias"
    elif available == 0:
        status = "Insufficient confirmation data"
    elif ratio >= 0.75:
        status = "Strong confirmation"
    elif ratio >= 0.50:
        status = "Moderate confirmation"
    else:
        status = "Weak confirmation"

    table = pd.DataFrame(rows)
    return {
        "bias": bias,
        "bias_score": bias_score,
        "confirmation_ratio": ratio,
        "status": status,
        "supportive_count": supportive,
        "available_count": available,
        "contradictions": contradictions,
        "credit_spread": credit_spread,
        "table": table,
    }


def _build_regime_engine(
    snapshot: dict[str, dict[str, float]],
    sentiment: dict[str, Any],
    volume_profile: dict[str, Any],
    rotation_signal: dict[str, Any],
    yield_curve: dict[str, Any],
    cross_asset: dict[str, Any],
    cta_proxy: dict[str, Any],
) -> dict[str, Any]:
    sent = float(sentiment.get("composite_score", 0.0))
    sent_n = float(np.clip(sent / 100.0, -1.0, 1.0))
    vix = snapshot.get("^VIX", {}).get("last", np.nan)
    vix_chg = snapshot.get("^VIX", {}).get("change_pct", np.nan)
    usd_chg = snapshot.get("UUP", {}).get("change_pct", np.nan)
    tnx_chg = snapshot.get("^TNX", {}).get("change_pct", np.nan)
    rot = rotation_signal.get("spread_daily", np.nan)
    vol_ratio = volume_profile.get("ratio", np.nan)
    curve_state = str(yield_curve.get("curve_state", "Unavailable"))
    curve_week = yield_curve.get("week_delta_bps", np.nan)
    cross_ratio = cross_asset.get("confirmation_ratio", np.nan)
    cross_bias = str(cross_asset.get("bias", "Neutral"))
    cta_n = float(cta_proxy.get("net_score", 0.0)) / 100.0
    credit_spread = cross_asset.get("credit_spread", np.nan)

    rot_pos = float(np.clip((rot if pd.notna(rot) else 0.0) / 0.015, -1.0, 1.0))

    scores = {
        "Risk-on": 0.0,
        "Risk-off": 0.0,
        "Growth scare": 0.0,
        "Inflation scare": 0.0,
        "Liquidity squeeze": 0.0,
    }

    scores["Risk-on"] += max(0.0, sent_n) * 35
    scores["Risk-on"] += max(0.0, rot_pos) * 20
    scores["Risk-on"] += max(0.0, cta_n) * 15
    scores["Risk-on"] += (15 if pd.notna(vix) and vix < 18 else 0)
    scores["Risk-on"] += (8 if pd.notna(usd_chg) and usd_chg < 0 else 0)
    scores["Risk-on"] += (7 if cross_bias == "Bullish" and pd.notna(cross_ratio) else 0) * (cross_ratio if pd.notna(cross_ratio) else 0)

    scores["Risk-off"] += max(0.0, -sent_n) * 35
    scores["Risk-off"] += max(0.0, -rot_pos) * 20
    scores["Risk-off"] += max(0.0, -cta_n) * 15
    scores["Risk-off"] += (15 if pd.notna(vix) and vix >= 22 else 0)
    scores["Risk-off"] += (8 if pd.notna(usd_chg) and usd_chg > 0 else 0)
    scores["Risk-off"] += (7 if cross_bias == "Bearish" and pd.notna(cross_ratio) else 0) * (cross_ratio if pd.notna(cross_ratio) else 0)

    scores["Growth scare"] += max(0.0, -sent_n) * 25
    scores["Growth scare"] += (18 if pd.notna(tnx_chg) and tnx_chg < 0 else 0)
    scores["Growth scare"] += (15 if curve_state in {"Inverted", "Flat"} else 0)
    scores["Growth scare"] += (10 if pd.notna(curve_week) and curve_week < 0 else 0)
    scores["Growth scare"] += (12 if pd.notna(vix_chg) and vix_chg > 0.05 else 0)

    scores["Inflation scare"] += (18 if pd.notna(tnx_chg) and tnx_chg > 0 else 0)
    scores["Inflation scare"] += (15 if pd.notna(usd_chg) and usd_chg > 0 else 0)
    scores["Inflation scare"] += (12 if pd.notna(curve_week) and curve_week > 3 else 0)
    scores["Inflation scare"] += max(0.0, rot_pos) * 10
    scores["Inflation scare"] += max(0.0, sent_n) * 8

    scores["Liquidity squeeze"] += (20 if pd.notna(vix_chg) and vix_chg > 0.08 else 0)
    scores["Liquidity squeeze"] += (20 if pd.notna(usd_chg) and usd_chg > 0.004 else 0)
    scores["Liquidity squeeze"] += (20 if pd.notna(credit_spread) and credit_spread < -0.002 else 0)
    scores["Liquidity squeeze"] += (20 if pd.notna(vol_ratio) and vol_ratio > 1.35 else 0)
    scores["Liquidity squeeze"] += (10 if sent <= -35 else 0)

    regime = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    top = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    confidence = float(np.clip(45 + (top - second), 25, 95))

    if confidence >= 75:
        conviction = "High"
    elif confidence >= 58:
        conviction = "Medium"
    else:
        conviction = "Low"

    components = pd.DataFrame(
        [
            {"factor": "Sentiment score", "value": sent, "signal": "Risk-on" if sent > 20 else ("Risk-off" if sent < -20 else "Mixed")},
            {"factor": "VIX level", "value": vix, "signal": "Calm" if pd.notna(vix) and vix < 18 else ("Stress" if pd.notna(vix) and vix > 22 else "Neutral")},
            {"factor": "USD daily change", "value": usd_chg, "signal": "Dollar down" if pd.notna(usd_chg) and usd_chg < 0 else ("Dollar up" if pd.notna(usd_chg) and usd_chg > 0 else "n/a")},
            {"factor": "Rates daily change (^TNX)", "value": tnx_chg, "signal": "Yields up" if pd.notna(tnx_chg) and tnx_chg > 0 else ("Yields down" if pd.notna(tnx_chg) and tnx_chg < 0 else "n/a")},
            {"factor": "Cross-asset confirmation", "value": cross_ratio, "signal": cross_asset.get("status", "n/a")},
            {"factor": "Curve state", "value": yield_curve.get("primary_spread_bps", np.nan), "signal": curve_state},
        ]
    )

    return {
        "regime": regime,
        "confidence": confidence,
        "conviction": conviction,
        "scores": scores,
        "components": components,
        "notes": "Regime engine blends sentiment, volatility, rates, dollar, credit, rotation, and trend-following proxies.",
    }


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    m = month + delta
    y = year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return y, m


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _business_day_near(year: int, month: int, day_hint: int) -> date:
    max_day = monthrange(year, month)[1]
    d = date(year, month, max(1, min(day_hint, max_day)))
    while d.weekday() >= 5:
        d += timedelta(days=1)
        if d.month != month:
            d -= timedelta(days=3)
            break
    return d


def _last_business_day(year: int, month: int) -> date:
    d = date(year, month, monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _build_event_risk_overlay(now: datetime | None = None) -> dict[str, Any]:
    now_et = now.astimezone(US_EASTERN) if now else datetime.now(US_EASTERN)
    today = now_et.date()

    rows: list[dict[str, Any]] = []
    for offset in range(-1, 4):
        y, m = _shift_month(today.year, today.month, offset)
        rows.append({"event": "NFP", "category": "Labor", "date": _first_friday(y, m), "weight": 26})
        rows.append({"event": "CPI", "category": "Inflation", "date": _business_day_near(y, m, 12), "weight": 30})
        rows.append({"event": "PCE", "category": "Inflation", "date": _last_business_day(y, m), "weight": 18})

    for _, fomc_dates in FOMC_CALENDAR.items():
        for d in fomc_dates:
            rows.append({"event": "FOMC Rate Decision", "category": "Central Bank", "date": d, "weight": 40})

    events = pd.DataFrame(rows).drop_duplicates(subset=["event", "date"])
    if events.empty:
        return {
            "risk_score": 0.0,
            "risk_level": "Low",
            "next_event": None,
            "events": pd.DataFrame(),
            "notes": "Macro event schedule unavailable.",
        }

    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.date
    events = events.dropna(subset=["date"]).copy()
    events["days_to_event"] = events["date"].map(lambda d: (d - today).days if isinstance(d, date) else np.nan)
    events = events[(events["days_to_event"] >= -1) & (events["days_to_event"] <= 30)].copy()
    if events.empty:
        return {
            "risk_score": 0.0,
            "risk_level": "Low",
            "next_event": None,
            "events": pd.DataFrame(),
            "notes": "No major macro events in the next 30 days.",
        }

    def _prox(days: int) -> float:
        if days < -1:
            return 0.0
        if days < 0:
            return 0.4
        return 1.0 / (days + 1)

    events["proximity"] = events["days_to_event"].map(_prox)
    events["risk_contrib"] = events["weight"] * events["proximity"]

    risk_score = float(np.clip(events["risk_contrib"].sum(), 0, 100))
    if risk_score >= 60:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    upcoming = events[events["days_to_event"] >= 0].sort_values(["days_to_event", "weight"], ascending=[True, False]).reset_index(drop=True)
    next_event = None
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        next_event = {
            "event": str(nxt["event"]),
            "date": nxt["date"],
            "days_to_event": int(nxt["days_to_event"]),
            "category": str(nxt["category"]),
        }

    events = events.sort_values(["days_to_event", "weight"], ascending=[True, False]).reset_index(drop=True)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "next_event": next_event,
        "events": events,
        "notes": "Overlay uses major US macro events (FOMC, CPI, NFP, PCE) and proximity-weighted risk.",
    }


def _alert(code: str, severity: str, message: str, value: float | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "value": value,
    }


def _generate_alerts(
    snapshot: dict[str, dict[str, float]],
    volume_profile: dict[str, Any],
    sentiment: dict[str, Any],
    sector_returns: pd.DataFrame,
    rotation_signal: dict[str, Any],
    yield_curve: dict[str, Any],
    capitulation: dict[str, Any],
    cta_proxy: dict[str, Any],
    sector_capitulation: dict[str, Any],
    signal_quality: dict[str, Any],
    cross_asset: dict[str, Any],
    regime_engine: dict[str, Any],
    event_risk: dict[str, Any],
    watchlist: dict[str, Any],
    liquidity_monitor: dict[str, Any],
    put_skew: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    put_skew = put_skew or {}

    vol_ratio = volume_profile.get("ratio", np.nan)
    if not np.isnan(vol_ratio):
        if vol_ratio >= 1.45:
            alerts.append(_alert("volume_heavy", "medium", f"Tape volume is very heavy at {vol_ratio:.2f}x expected.", vol_ratio))
        elif vol_ratio <= 0.75:
            alerts.append(_alert("volume_light", "low", f"Tape volume is light at {vol_ratio:.2f}x expected.", vol_ratio))

    sentiment_score = float(sentiment.get("composite_score", 0.0))
    if sentiment_score >= 55:
        alerts.append(_alert("sentiment_risk_on", "medium", "Composite sentiment is strongly risk-on.", sentiment_score))
    elif sentiment_score <= -55:
        alerts.append(_alert("sentiment_risk_off", "high", "Composite sentiment is strongly risk-off.", sentiment_score))

    vix_change = snapshot.get("^VIX", {}).get("change_pct", np.nan)
    if not np.isnan(vix_change) and vix_change >= 0.10:
        alerts.append(_alert("vix_spike", "high", f"VIX is spiking ({vix_change * 100:+.1f}% day/day).", vix_change))

    spread_daily = rotation_signal.get("spread_daily", np.nan)
    if not np.isnan(spread_daily):
        if spread_daily >= 0.012:
            alerts.append(_alert("rotation_cyclical", "low", "Cyclicals are materially outperforming defensives today.", spread_daily))
        elif spread_daily <= -0.012:
            alerts.append(_alert("rotation_defensive", "medium", "Defensives are materially outperforming cyclicals today.", spread_daily))

    if not sector_returns.empty and "daily" in sector_returns:
        dispersion = float(sector_returns["daily"].max() - sector_returns["daily"].min())
        if dispersion >= 0.035:
            alerts.append(_alert("sector_dispersion", "medium", "Sector dispersion is elevated, indicating concentrated flows.", dispersion))

    yc_spread = yield_curve.get("primary_spread_bps", np.nan)
    yc_week = yield_curve.get("week_delta_bps", np.nan)
    if not np.isnan(yc_spread) and yc_spread < 0:
        alerts.append(_alert("curve_inversion", "high", f"Yield curve inversion: {yield_curve.get('primary_spread_name', 'spread')} at {yc_spread:+.1f} bps.", yc_spread))
    if not np.isnan(yc_week) and yc_week >= 8:
        alerts.append(_alert("curve_fast_steepen", "medium", f"Curve is steepening quickly over 1W ({yc_week:+.1f} bps).", yc_week))
    elif not np.isnan(yc_week) and yc_week <= -8:
        alerts.append(_alert("curve_fast_flatten", "medium", f"Curve is flattening quickly over 1W ({yc_week:+.1f} bps).", yc_week))

    down_score = capitulation.get("downside_score", np.nan)
    up_score = capitulation.get("upside_score", np.nan)
    cap_threshold = float(capitulation.get("extreme_threshold", 75.0))
    if not np.isnan(down_score) and down_score >= cap_threshold and bool(capitulation.get("downside_trigger_today", False)):
        alerts.append(_alert("capitulation_down_trigger", "high", f"Downside capitulation trigger fired ({down_score:.0f}/100); awaiting 1-3 bar reversal confirmation.", down_score))
    if not np.isnan(up_score) and up_score >= cap_threshold and bool(capitulation.get("upside_trigger_today", False)):
        alerts.append(_alert("capitulation_up_trigger", "medium", f"Upside exhaustion trigger fired ({up_score:.0f}/100); awaiting 1-3 bar reversal confirmation.", up_score))

    cap_log = capitulation.get("signal_log", pd.DataFrame())
    if isinstance(cap_log, pd.DataFrame) and not cap_log.empty:
        conf = cap_log[cap_log["state"] == "confirmed"]
        if not conf.empty:
            latest_conf = conf.iloc[0]
            direction = str(latest_conf.get("direction", "Signal"))
            bars = latest_conf.get("confirmation_bars")
            score = latest_conf.get("trigger_score")
            if direction == "Downside Capitulation":
                alerts.append(_alert("capitulation_down_confirmed", "medium", f"Downside capitulation confirmed in {int(bars)} bars (trigger {float(score):.0f}/100).", float(score)))
            elif direction == "Upside Exhaustion":
                alerts.append(_alert("capitulation_up_confirmed", "low", f"Upside exhaustion confirmed in {int(bars)} bars (trigger {float(score):.0f}/100).", float(score)))

    sec_cap_log = sector_capitulation.get("signal_log", pd.DataFrame())
    if isinstance(sec_cap_log, pd.DataFrame) and not sec_cap_log.empty:
        latest = sec_cap_log.iloc[0]
        if str(latest.get("state", "")) == "confirmed":
            alerts.append(
                _alert(
                    "sector_capitulation_confirmed",
                    "medium",
                    f"Sector signal confirmed: {latest.get('sector', 'n/a')} {latest.get('direction', '')} in {int(latest.get('confirmation_bars', 0))} bars.",
                    float(latest.get("trigger_score", np.nan)) if pd.notna(latest.get("trigger_score", np.nan)) else None,
                )
            )

    cta_net = float(cta_proxy.get("net_score", 0.0))
    if cta_net >= 65:
        alerts.append(_alert("cta_crowded_long", "low", "CTA proxy is crowded long across the basket.", cta_net))
    elif cta_net <= -65:
        alerts.append(_alert("cta_crowded_short", "medium", "CTA proxy is crowded short across the basket.", cta_net))

    cross_ratio = cross_asset.get("confirmation_ratio", np.nan)
    cross_bias = cross_asset.get("bias", "Neutral")
    if cross_bias != "Neutral" and pd.notna(cross_ratio) and cross_ratio < 0.4:
        alerts.append(_alert("cross_asset_divergence", "medium", f"Cross-asset confirmation is weak ({cross_ratio * 100:.0f}%) for {cross_bias.lower()} bias.", cross_ratio))

    regime = regime_engine.get("regime", "")
    regime_conf = regime_engine.get("confidence", np.nan)
    if regime and pd.notna(regime_conf) and regime_conf >= 78:
        alerts.append(_alert("regime_high_conviction", "low", f"Regime engine: {regime} with high conviction ({regime_conf:.0f}/100).", regime_conf))

    event_level = str(event_risk.get("risk_level", "Low"))
    event_score = event_risk.get("risk_score", np.nan)
    next_event = event_risk.get("next_event") or {}
    if event_level == "High":
        msg = f"Macro event risk is HIGH ({event_score:.0f}/100)."
        if isinstance(next_event, dict) and next_event:
            msg += f" Next: {next_event.get('event', 'event')} in {next_event.get('days_to_event', 'n/a')} day(s)."
        alerts.append(_alert("event_risk_high", "medium", msg, event_score))

    movers = watchlist.get("movers", pd.DataFrame())
    if isinstance(movers, pd.DataFrame) and not movers.empty:
        top = movers.iloc[0]
        alerts.append(
            _alert(
                "watchlist_mover",
                "low",
                f"Watchlist mover: {top.get('ticker', '')} {_fmt_pct(top.get('daily', np.nan))} (threshold {watchlist.get('threshold_pct', 0):.1f}%).",
                float(top.get("daily", np.nan)) if pd.notna(top.get("daily", np.nan)) else None,
            )
        )

    quality_hit = signal_quality.get("overall_hit_rate_5d", np.nan)
    if pd.notna(quality_hit) and quality_hit < 45:
        alerts.append(_alert("signal_quality_soft", "medium", f"Capitulation signal quality has softened (5D hit rate {quality_hit:.0f}%).", quality_hit))

    liq_score = liquidity_monitor.get("liquidity_stress_score", np.nan)
    real_z = liquidity_monitor.get("real_rate_proxy_z", np.nan)
    if pd.notna(liq_score) and liq_score >= 70:
        alerts.append(_alert("liquidity_tight", "high", f"Liquidity monitor flags severe tightness ({liq_score:.0f}/100).", liq_score))
    elif pd.notna(liq_score) and liq_score <= 25:
        alerts.append(_alert("liquidity_easy", "low", f"Liquidity conditions look easy ({liq_score:.0f}/100).", liq_score))

    if pd.notna(real_z) and real_z >= 1.2:
        alerts.append(_alert("real_rate_restrictive", "medium", f"Real-rate proxy is restrictive at {real_z:+.2f} z.", real_z))
    elif pd.notna(real_z) and real_z <= -1.0:
        alerts.append(_alert("real_rate_accommodative", "low", f"Real-rate proxy is accommodative at {real_z:+.2f} z.", real_z))

    skew_avg = put_skew.get("avg_put_call_iv_spread", np.nan)
    skew_oi = put_skew.get("avg_put_call_oi_ratio", np.nan)
    stressed = int(put_skew.get("stressed_count", 0) or 0)
    total = int(put_skew.get("total_symbols", 0) or 0)

    if pd.notna(skew_avg) and skew_avg >= 6:
        alerts.append(_alert("put_skew_extreme", "high", f"Put skew is extreme at {skew_avg:.1f} vol pts (95p-105c).", skew_avg))
    elif pd.notna(skew_avg) and skew_avg >= 3.5:
        alerts.append(_alert("put_skew_elevated", "medium", f"Put skew is elevated at {skew_avg:.1f} vol pts.", skew_avg))

    if pd.notna(skew_oi) and skew_oi >= 1.6:
        alerts.append(_alert("put_call_oi_heavy", "medium", f"Put/Call open interest ratio is heavy at {skew_oi:.2f}.", skew_oi))

    if total > 0 and stressed >= max(2, total):
        alerts.append(_alert("put_skew_broad", "medium", f"Downside skew is broad across {stressed}/{total} monitored underlyings.", float(stressed / total)))

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    alerts.sort(key=lambda x: severity_rank.get(str(x.get("severity", "")).lower(), 0), reverse=True)
    return alerts


def _fmt_pct(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:+.2f}%"


def _build_market_narrative(
    snapshot: dict[str, dict[str, float]],
    sector_returns: pd.DataFrame,
    volume_profile: dict[str, Any],
    sentiment: dict[str, Any],
    rotation_signal: dict[str, Any],
    yield_curve: dict[str, Any],
    headlines: list[dict[str, Any]],
) -> list[str]:
    spx = snapshot.get("^GSPC", {})
    ndx = snapshot.get("^IXIC", {})
    vix = snapshot.get("^VIX", {})
    tnx = snapshot.get("^TNX", {})
    usd = snapshot.get("UUP", {})

    breadth_ratio = sentiment.get("breadth_ratio", np.nan)
    breadth_text = (
        f"{int(round(breadth_ratio * 100))}% of sectors are green."
        if not np.isnan(breadth_ratio)
        else "Sector breadth is unavailable."
    )
    line1 = (
        f"{spx.get('label', 'S&P 500')} {_fmt_pct(spx.get('change_pct', np.nan))}, "
        f"{ndx.get('label', 'Nasdaq')} {_fmt_pct(ndx.get('change_pct', np.nan))}. {breadth_text}"
    )

    if not sector_returns.empty:
        leaders = sector_returns.nlargest(2, "daily")[["sector", "daily"]].values.tolist()
        laggards = sector_returns.nsmallest(2, "daily")[["sector", "daily"]].values.tolist()
        line2 = (
            f"Leadership: {leaders[0][0]} {_fmt_pct(leaders[0][1])}, {leaders[1][0]} {_fmt_pct(leaders[1][1])}. "
            f"Laggards: {laggards[0][0]} {_fmt_pct(laggards[0][1])}, {laggards[1][0]} {_fmt_pct(laggards[1][1])}."
        )
    else:
        line2 = "Sector leadership could not be computed from current data."

    ratio = volume_profile.get("ratio", np.nan)
    regime = volume_profile.get("regime", "Unavailable")
    session = volume_profile.get("session")
    session_label = session.label if session else "Session"
    line3 = (
        f"Volume regime: {regime}. Composite tape is {ratio:.2f}x expected for {session_label}."
        if not np.isnan(ratio)
        else f"Volume regime: {regime}."
    )

    tnx_move_bp = np.nan
    if not np.isnan(tnx.get("change_pct", np.nan)) and not np.isnan(tnx.get("last", np.nan)):
        prev_tnx = tnx["last"] / (1 + tnx["change_pct"]) if (1 + tnx["change_pct"]) else np.nan
        if prev_tnx and not np.isnan(prev_tnx):
            tnx_move_bp = (tnx["last"] - prev_tnx) * 10

    macro_bits = [
        f"VIX {vix.get('last', np.nan):.2f} ({_fmt_pct(vix.get('change_pct', np.nan))})"
        if not np.isnan(vix.get("last", np.nan))
        else "VIX n/a",
        f"10Y move {tnx_move_bp:+.1f} bps" if not np.isnan(tnx_move_bp) else "10Y move n/a",
        f"USD {_fmt_pct(usd.get('change_pct', np.nan))}",
    ]
    line4 = "Macro tape: " + ", ".join(macro_bits) + "."

    rot_spread = rotation_signal.get("spread_daily", np.nan)
    if not np.isnan(rot_spread):
        line5 = (
            f"Rotation: {rotation_signal.get('regime', 'n/a')} "
            f"(cyclical-defensive spread {_fmt_pct(rot_spread)} today)."
        )
    else:
        line5 = "Rotation: unavailable."

    yc_name = yield_curve.get("primary_spread_name")
    yc_spread = yield_curve.get("primary_spread_bps", np.nan)
    yc_week = yield_curve.get("week_delta_bps", np.nan)
    if yc_name and not np.isnan(yc_spread):
        if not np.isnan(yc_week):
            line6 = (
                f"Yield curve: {yield_curve.get('curve_state', 'n/a')} on {yc_name} at {yc_spread:+.1f} bps; "
                f"1W change {yc_week:+.1f} bps ({yield_curve.get('week_trend', 'n/a').lower()})."
            )
        else:
            line6 = f"Yield curve: {yield_curve.get('curve_state', 'n/a')} on {yc_name} at {yc_spread:+.1f} bps."
    else:
        line6 = "Yield curve: unavailable."

    top_headline = headlines[0]["title"] if headlines else ""
    if top_headline:
        line7 = f"Headline driver watch: {top_headline}"
        return [line1, line2, line3, line4, line5, line6, line7]
    return [line1, line2, line3, line4, line5, line6]


def build_dashboard_payload() -> dict[str, Any]:
    watchlist_tickers = _load_watchlist_tickers()
    all_tickers = sorted(
        set(
            list(SECTOR_ETFS.keys())
            + list(MARKET_TICKERS.keys())
            + VOLUME_PROXY_ETFS
            + list(CTA_PROXY_TICKERS.keys())
            + list(CROSS_ASSET_EXTRA_TICKERS)
            + list(LIQUIDITY_MONITOR_TICKERS)
            + watchlist_tickers
        )
    )
    daily_data = _download_batch(all_tickers, period="8mo", interval="1d")

    snapshot = _build_market_snapshot(daily_data)
    sector_returns = _build_sector_returns(daily_data)
    volume_profile = _compute_volume_profile(daily_data)
    headlines = _fetch_headlines(limit=8)
    sentiment = _build_sentiment_bundle(snapshot, sector_returns, headlines, daily_data)
    cta_proxy = _build_cta_proxy(daily_data)
    rotation_signal = _build_sector_rotation_signal(sector_returns, daily_data)
    yield_curve = _build_yield_curve_signal()
    capitulation = _build_capitulation_signal(daily_data)
    sector_capitulation = _build_sector_capitulation_signals(daily_data)
    signal_quality = _build_signal_quality_tracker(daily_data, capitulation, sector_capitulation)
    sector_flow = _build_sector_heatmap_flow(daily_data, sector_returns)
    watchlist = _build_watchlist_monitor(daily_data)
    cross_asset_confirmation = _build_cross_asset_confirmation(snapshot, daily_data, sentiment, rotation_signal, cta_proxy)
    regime_engine = _build_regime_engine(snapshot, sentiment, volume_profile, rotation_signal, yield_curve, cross_asset_confirmation, cta_proxy)
    event_risk = _build_event_risk_overlay()
    liquidity_monitor = _build_liquidity_real_rate_monitor(daily_data, snapshot, volume_profile)
    put_skew = _build_put_skew_monitor(daily_data, snapshot)

    alerts = _generate_alerts(
        snapshot,
        volume_profile,
        sentiment,
        sector_returns,
        rotation_signal,
        yield_curve,
        capitulation,
        cta_proxy,
        sector_capitulation,
        signal_quality,
        cross_asset_confirmation,
        regime_engine,
        event_risk,
        watchlist,
        liquidity_monitor,
        put_skew,
    )
    narrative = _build_market_narrative(snapshot, sector_returns, volume_profile, sentiment, rotation_signal, yield_curve, headlines)

    updated_at = datetime.now(US_EASTERN)
    return {
        "updated_at": updated_at,
        "snapshot": snapshot,
        "sector_returns": sector_returns,
        "volume_profile": volume_profile,
        "sentiment": sentiment,
        "cta_proxy": cta_proxy,
        "rotation_signal": rotation_signal,
        "yield_curve": yield_curve,
        "capitulation": capitulation,
        "sector_capitulation": sector_capitulation,
        "signal_quality": signal_quality,
        "sector_flow": sector_flow,
        "watchlist": watchlist,
        "cross_asset_confirmation": cross_asset_confirmation,
        "regime_engine": regime_engine,
        "event_risk": event_risk,
        "liquidity_monitor": liquidity_monitor,
        "put_skew": put_skew,
        "alerts": alerts,
        "headlines": headlines,
        "narrative": narrative,
    }

