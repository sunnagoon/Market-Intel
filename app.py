from __future__ import annotations

import html
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from alert_router import route_alerts
from history_store import load_alert_history, load_snapshot_history, persist_payload
from market_intel import build_dashboard_payload, build_near_52w_low_screener_payload


st.set_page_config(
    page_title="Market Pulse Intelligence",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(37, 123, 192, 0.45), transparent 60%),
                radial-gradient(1000px 700px at 95% 5%, rgba(39, 174, 96, 0.25), transparent 45%),
                linear-gradient(170deg, #06111f 0%, #081a2e 45%, #0a2036 100%);
            color: #e8f1ff;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071222 0%, #0a1b30 100%);
            border-right: 1px solid rgba(126, 163, 204, 0.25);
        }
        h1, h2, h3, h4, .stMarkdown, .stText {
            font-family: "Space Grotesk", sans-serif !important;
            color: #f8fbff !important;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.22);
        }
        .mono {
            font-family: "IBM Plex Mono", monospace;
            color: #f8fbff !important;
            letter-spacing: 0.02em;
            font-size: 0.86rem;
            text-shadow: 0 0 8px rgba(255, 255, 255, 0.20);
        }
        [data-testid="stAppViewContainer"] *,
        [data-testid="stSidebar"] *,
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stCaptionContainer"] *,
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] *,
        [data-testid="stTabs"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-baseweb="select"] *,
        label,
        p,
        span,
        li,
        a,
        a:visited {
            color: #f8fbff !important;
            text-shadow: 0 0 8px rgba(255, 255, 255, 0.16);
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] *,
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="menu"] * {
            color: #111111 !important;
            text-shadow: none !important;
        }
        .panel {
            background: linear-gradient(180deg, rgba(10, 24, 43, 0.88) 0%, rgba(8, 22, 40, 0.94) 100%);
            border: 1px solid rgba(125, 165, 208, 0.28);
            border-radius: 14px;
            padding: 18px 20px;
            margin: 0 0 18px 0;
            box-shadow: 0 10px 30px rgba(2, 9, 18, 0.35);
        }
        .panel h1, .panel h2, .panel h3, .panel p, .panel li {
            line-height: 1.5;
        }
        .headline-item {
            border: 1px solid rgba(126, 163, 204, 0.26);
            border-radius: 12px;
            padding: 10px 12px;
            margin: 0 0 10px 0;
            background: linear-gradient(170deg, rgba(16, 37, 62, 0.60) 0%, rgba(10, 27, 46, 0.60) 100%);
        }
        .headline-link {
            color: #e8f1ff;
            text-decoration: none;
            font-weight: 600;
        }
        .headline-link:hover {
            text-decoration: underline;
        }
        .alert-item {
            border: 1px solid rgba(126, 163, 204, 0.24);
            border-left: 4px solid rgba(126, 163, 204, 0.40);
            border-radius: 10px;
            padding: 9px 12px;
            margin: 0 0 8px 0;
            background: rgba(8, 22, 40, 0.55);
        }
        .alert-high { border-left-color: rgba(255, 96, 96, 0.95); }
        .alert-medium { border-left-color: rgba(246, 205, 97, 0.95); }
        .alert-low { border-left-color: rgba(88, 211, 144, 0.95); }
        div[data-testid="metric-container"] {
            background: linear-gradient(170deg, rgba(16, 37, 62, 0.92) 0%, rgba(10, 27, 46, 0.95) 100%);
            border: 1px solid rgba(126, 163, 204, 0.24);
            border-radius: 14px;
            padding: 10px 12px;
            margin: 4px 0 10px 0;
        }
        [data-testid="stSidebar"] .stButton > button {
            border: 1px solid rgba(126, 163, 204, 0.35);
            border-radius: 10px;
            background: linear-gradient(170deg, rgba(14, 34, 57, 0.94) 0%, rgba(9, 24, 41, 0.96) 100%);
            color: #f8fbff !important;
            margin-bottom: 0.2rem;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(157, 196, 236, 0.70);
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
            color: #f8fbff !important;
            text-shadow: none !important;
        }
        .focus-banner {
            border: 1px solid rgba(125, 165, 208, 0.36);
            background: rgba(10, 27, 46, 0.72);
            border-radius: 12px;
            padding: 10px 12px;
            margin: 0 0 14px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30, show_spinner=False)
def _get_payload() -> dict:
    return build_dashboard_payload()


@st.cache_data(ttl=1800, show_spinner=False)
def _get_near_52w_low_screener_payload() -> dict:
    return build_near_52w_low_screener_payload()

SECTION_NAV_ITEMS = [
    ("overview", "Overview"),
    ("actionability", "Actionability Lab"),
    ("alerts", "Signal Alerts"),
    ("regime_cross", "Regime + Cross-Asset"),
    ("event_watch", "Event Risk + Watchlist"),
    ("near_52w_low", "Near 52W Low Screener"),
    ("liquidity", "Liquidity / Real Rates"),
    ("vol_structure", "Volatility Structure"),
    ("put_skew", "Put Skew"),
    ("capitulation", "Capitulation"),
    ("sector_capitulation", "Sector Capitulation"),
    ("signal_quality", "Signal Quality"),
    ("volume", "Volume"),
    ("rotation", "Sector Rotation"),
    ("sector_flow", "Sector Flow"),
    ("rotation_flow_map", "Rotation Flow Map"),
    ("factor_baskets", "Factor Baskets"),
    ("yield_curve", "Yield Curve"),
    ("sectors", "Sector Performance"),
    ("cta", "CTA Proxy"),
    ("alert_routing", "Alert Routing"),
    ("history", "Historical Trace"),
    ("headlines", "Headlines"),
]
SECTION_LABEL_BY_KEY = {key: label for key, label in SECTION_NAV_ITEMS}
SECTION_KEY_BY_LABEL = {label: key for key, label in SECTION_NAV_ITEMS}


def _pct_color(value: float) -> str:
    if np.isnan(value):
        return "#9eb5d2"
    return "#44d47e" if value >= 0 else "#ff6b6b"


def _format_pct(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:+.2f}%"


def _apply_chart_theme(fig: go.Figure) -> go.Figure:
    title_text = fig.layout.title.text if fig.layout.title is not None else ""
    if title_text is None or str(title_text).strip().lower() == "undefined":
        title_text = ""

    legend_title = ""
    if fig.layout.legend is not None and fig.layout.legend.title is not None:
        raw_legend_title = fig.layout.legend.title.text
        if raw_legend_title is not None and str(raw_legend_title).strip().lower() != "undefined":
            legend_title = str(raw_legend_title)

    right_margin = 20
    if fig.layout.margin is not None and fig.layout.margin.r is not None:
        try:
            right_margin = int(fig.layout.margin.r)
        except (TypeError, ValueError):
            right_margin = 20

    has_legend_content = any(
        getattr(trace, "type", "") != "indicator" and str(getattr(trace, "name", "")).strip() not in ("", "None", "undefined")
        for trace in fig.data
    )

    legend_config = dict(
        title_text=legend_title,
        orientation="v",
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=1.02,
        font=dict(color="#ffffff", size=12),
        title_font=dict(color="#ffffff", size=12),
        bgcolor="rgba(7, 22, 39, 0.55)",
        bordercolor="rgba(140,170,205,0.35)",
        borderwidth=1,
        tracegroupgap=8,
    )

    fig.update_layout(
        title_text=title_text,
        font=dict(color="#ffffff", family="Space Grotesk", size=14),
        title_font=dict(color="#ffffff", size=20),
        showlegend=has_legend_content,
        legend=legend_config,
        margin=dict(r=max(right_margin, 175) if has_legend_content else max(right_margin, 30)),
    )
    fig.update_xaxes(
        color="#ffffff",
        title_font=dict(color="#ffffff", size=14),
        tickfont=dict(color="#ffffff", size=12),
    )
    fig.update_yaxes(
        color="#ffffff",
        title_font=dict(color="#ffffff", size=14),
        tickfont=dict(color="#ffffff", size=12),
    )
    return fig


def _sector_bar_chart(data: pd.DataFrame, col: str, title: str) -> go.Figure:
    if data.empty:
        return go.Figure()
    df = data.sort_values(col, ascending=True)
    colors = ["#42d680" if x >= 0 else "#ff6f6f" for x in df[col]]
    fig = go.Figure(
        go.Bar(
            x=df[col] * 100,
            y=df["sector"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:+.2f}%" for x in df[col] * 100],
            textposition="outside",
            textfont=dict(color="#ffffff", size=13),
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Return %", gridcolor="rgba(140,170,205,0.18)", zerolinecolor="rgba(140,170,205,0.25)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=420,
    )
    return _apply_chart_theme(fig)


def _gauge(value: float, title: str, low: float, high: float, suffix: str = "") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": "#ffffff", "size": 34}},
            title={"text": title, "font": {"color": "#ffffff", "size": 18}},
            gauge={
                "axis": {"range": [low, high], "tickcolor": "#ffffff", "tickfont": {"color": "#ffffff", "size": 12}},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "rgba(126,163,204,0.35)",
                "bar": {"color": "#34c18f"},
                "steps": [
                    {"range": [low, low + (high - low) * 0.35], "color": "rgba(255,107,107,0.32)"},
                    {"range": [low + (high - low) * 0.35, low + (high - low) * 0.7], "color": "rgba(246,205,97,0.27)"},
                    {"range": [low + (high - low) * 0.7, high], "color": "rgba(68,212,126,0.30)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(family="Space Grotesk"),
        height=260,
    )
    return _apply_chart_theme(fig)


def _rotation_ratio_chart(ratio_history: pd.DataFrame) -> go.Figure:
    if ratio_history.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratio_history["date"],
            y=ratio_history["ratio"],
            mode="lines",
            name="Cyclical / Defensive",
            line=dict(color="#53d390", width=2),
        )
    )
    if "ratio_z63" in ratio_history:
        fig.add_trace(
            go.Scatter(
                x=ratio_history["date"],
                y=ratio_history["ratio_z63"],
                mode="lines",
                name="Ratio Z-Score (63d)",
                line=dict(color="#7db3ff", width=1.7, dash="dot"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title="Cyclical vs Defensive Ratio",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title="Ratio", gridcolor="rgba(140,170,205,0.18)"),
        yaxis2=dict(title="Z-Score", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.14, x=0),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=300,
    )
    return _apply_chart_theme(fig)


def _history_chart(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp_et"],
            y=history_df["sentiment_score"],
            mode="lines",
            name="Sentiment",
            line=dict(color="#59d39a", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp_et"],
            y=history_df["volume_ratio"],
            mode="lines",
            name="Volume Ratio",
            line=dict(color="#7db3ff", width=2),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=history_df["timestamp_et"],
            y=history_df["spx_change"] * 100,
            name="S&P %",
            marker=dict(color="rgba(255, 203, 109, 0.45)"),
            opacity=0.45,
        ),
        secondary_y=False,
    )

    fig.update_xaxes(gridcolor="rgba(140,170,205,0.18)")
    fig.update_yaxes(title_text="Sentiment / S&P %", gridcolor="rgba(140,170,205,0.18)", secondary_y=False)
    fig.update_yaxes(title_text="Volume Ratio", secondary_y=True)
    fig.update_layout(
        title="Sentiment / Volume / S&P Trace",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=360,
    )
    return _apply_chart_theme(fig)


def _render_header(updated_at: datetime) -> None:
    st.title("Market Pulse Intelligence")
    st.markdown(
        "<div class='mono'>Built with Yahoo Finance proxies. Headlines and intraday fields can be delayed by the feed provider.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='mono'>Last refresh: {updated_at.strftime('%Y-%m-%d %I:%M:%S %p ET')}</div>",
        unsafe_allow_html=True,
    )


def _render_market_metrics(snapshot: dict) -> None:
    wanted = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^TNX"]
    cols = st.columns(6)
    for idx, ticker in enumerate(wanted):
        with cols[idx]:
            item = snapshot.get(ticker, {})
            label = item.get("label", ticker)
            last = item.get("last", np.nan)
            chg = item.get("change_pct", np.nan)
            st.metric(label=label, value=f"{last:,.2f}" if not np.isnan(last) else "n/a", delta=_format_pct(chg))


def _render_narrative(narrative: list[str]) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Why The Market Is Moving")
    for line in narrative:
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_alerts(alerts: list[dict], alert_history: pd.DataFrame) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Signal Alerts")

    if not alerts:
        st.info("No active threshold alerts right now.")
    else:
        for alert in alerts:
            severity = str(alert.get("severity", "low")).lower()
            message = html.escape(str(alert.get("message", "")))
            label = html.escape(str(alert.get("code", "signal")).replace("_", " ").title())
            value = alert.get("value")
            value_text = f" | {value:+.2f}" if isinstance(value, (int, float)) and not np.isnan(value) else ""
            st.markdown(
                f"<div class='alert-item alert-{severity}'><strong>{label}</strong><br>{message}{value_text}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='mono' style='margin-top:8px;'>Recent alert log</div>", unsafe_allow_html=True)
    if alert_history.empty:
        st.caption("No historical alerts recorded yet.")
    else:
        view = alert_history.copy()
        view["time"] = view["timestamp_et"].dt.strftime("%m-%d %I:%M %p")
        st.dataframe(view[["time", "severity", "message"]].head(12), width='stretch', hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _capitulation_history_chart(score_history: pd.DataFrame, threshold: float) -> go.Figure:
    if score_history.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score_history["date"],
            y=score_history["downside_score"],
            mode="lines",
            name="Downside Capitulation",
            line=dict(color="#ff7d7d", width=2.1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=score_history["date"],
            y=score_history["upside_score"],
            mode="lines",
            name="Upside Exhaustion",
            line=dict(color="#66dca3", width=2.1),
        )
    )
    fig.add_hline(y=threshold, line_color="rgba(246,205,97,0.9)", line_dash="dot")
    fig.update_layout(
        title="Capitulation Scores (0-100)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=30, b=10),
        yaxis=dict(title="Score (0-100)", range=[0, 100], gridcolor="rgba(140,170,205,0.18)"),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        legend=dict(orientation="h", y=1.12, x=0),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=320,
    )
    return _apply_chart_theme(fig)


def _sector_capitulation_scores_chart(table: pd.DataFrame, threshold: float) -> go.Figure:
    if table.empty:
        return go.Figure()

    df = table.copy()
    for col in ["downside_score", "upside_score"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["max_score"] = df[["downside_score", "upside_score"]].max(axis=1)
    df = df.sort_values("max_score", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["downside_score"],
            y=df["sector"],
            orientation="h",
            name="Downside Capitulation",
            marker=dict(color="#ff7d7d", opacity=0.9),
            text=[f"{x:.0f}" if pd.notna(x) else "n/a" for x in df["downside_score"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["upside_score"],
            y=df["sector"],
            orientation="h",
            name="Upside Exhaustion",
            marker=dict(color="#66dca3", opacity=0.9),
            text=[f"{x:.0f}" if pd.notna(x) else "n/a" for x in df["upside_score"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=threshold, line_color="rgba(246,205,97,0.9)", line_dash="dot")
    fig.update_layout(
        title="Sector Capitulation Scores (0-100)",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=28, t=35, b=10),
        xaxis=dict(title="Score", range=[0, 100], gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(gridcolor="rgba(140,170,205,0.12)"),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=420,
    )
    return _apply_chart_theme(fig)


def _render_capitulation_panel(capitulation: dict) -> None:
    st.subheader("Capitulation / Exhaustion Model")
    threshold = float(capitulation.get("extreme_threshold", 75.0))
    down = capitulation.get("downside_score", np.nan)
    up = capitulation.get("upside_score", np.nan)

    c1, c2 = st.columns([1, 1])
    with c1:
        if pd.notna(down):
            st.plotly_chart(_gauge(float(down), "Downside Capitulation", 0, 100), width='stretch')
            st.markdown(f"**Status:** {capitulation.get('downside_status', 'n/a')}")
        else:
            st.info("Downside score unavailable.")
    with c2:
        if pd.notna(up):
            st.plotly_chart(_gauge(float(up), "Upside Exhaustion", 0, 100), width='stretch')
            st.markdown(f"**Status:** {capitulation.get('upside_status', 'n/a')}")
        else:
            st.info("Upside score unavailable.")

    st.caption(f"Trigger rule: score >= {threshold:.0f} today, then reversal confirmation in next 1-3 bars.")
    st.caption(str(capitulation.get("notes", "")))

    hist = capitulation.get("score_history", pd.DataFrame())
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        st.plotly_chart(_capitulation_history_chart(hist, threshold), width='stretch')
    else:
        st.info("Score history unavailable.")

    log = capitulation.get("signal_log", pd.DataFrame())
    st.markdown("**Signal Log**")
    if isinstance(log, pd.DataFrame) and not log.empty:
        show = log.copy()
        if "trigger_date" in show.columns:
            show["trigger_date"] = pd.to_datetime(show["trigger_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "trigger_score" in show.columns:
            show["trigger_score"] = show["trigger_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
        if "reversal_return" in show.columns:
            show["reversal_return"] = show["reversal_return"].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        if "confirmation_bars" in show.columns:
            show["confirmation_bars"] = show["confirmation_bars"].map(lambda x: f"{int(x)}" if pd.notna(x) else "-")

        st.dataframe(
            show[["trigger_date", "direction", "trigger_score", "state", "confirmation_bars", "reversal_return"]].head(20),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No triggers logged in the current lookback window.")


def _render_sector_capitulation_panel(sector_cap: dict) -> None:
    st.subheader("Sector Capitulation / Exhaustion")

    threshold = float(sector_cap.get("threshold", 60.0))
    st.caption(f"Trigger rule: score >= {threshold:.0f} today, then reversal confirmation in next 1-3 bars.")
    st.caption(str(sector_cap.get("notes", "")))

    table = sector_cap.get("table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.plotly_chart(_sector_capitulation_scores_chart(table, threshold), width='stretch')

        show = table.copy()
        if "downside_score" in show.columns:
            show["downside_score"] = show["downside_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
        if "upside_score" in show.columns:
            show["upside_score"] = show["upside_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
        for flag_col in ["downside_trigger_today", "upside_trigger_today"]:
            if flag_col in show.columns:
                show[flag_col] = show[flag_col].map(lambda x: "Yes" if bool(x) else "No")

        st.markdown("**Sector Scoreboard**")
        st.dataframe(
            show[
                [
                    "sector",
                    "ticker",
                    "downside_score",
                    "upside_score",
                    "downside_trigger_today",
                    "upside_trigger_today",
                    "downside_status",
                    "upside_status",
                ]
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("Sector capitulation table unavailable.")

    st.markdown("**Sector Signal Log**")
    log = sector_cap.get("signal_log", pd.DataFrame())
    if isinstance(log, pd.DataFrame) and not log.empty:
        show = log.copy()
        if "trigger_date" in show.columns:
            show["trigger_date"] = pd.to_datetime(show["trigger_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "trigger_score" in show.columns:
            show["trigger_score"] = show["trigger_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
        if "reversal_return" in show.columns:
            show["reversal_return"] = show["reversal_return"].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        if "confirmation_bars" in show.columns:
            show["confirmation_bars"] = show["confirmation_bars"].map(lambda x: f"{int(x)}" if pd.notna(x) else "-")

        st.dataframe(
            show[
                [
                    "trigger_date",
                    "sector",
                    "ticker",
                    "direction",
                    "trigger_score",
                    "state",
                    "confirmation_bars",
                    "reversal_return",
                ]
            ].head(40),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No sector triggers logged in the current lookback window.")


def _render_volume_panel(volume_profile: dict) -> None:
    ratio = volume_profile.get("ratio", np.nan)
    regime = volume_profile.get("regime", "Unavailable")
    session = volume_profile.get("session")
    if isinstance(session, dict):
        session_label = str(session.get("label", "Session"))
        progress = float(session.get("progress", 0.0)) * 100
    else:
        session_label = getattr(session, "label", "Session")
        progress = float(getattr(session, "progress", 0.0)) * 100
    col1, col2 = st.columns([1, 1])
    with col1:
        if np.isnan(ratio):
            st.info("Volume ratio unavailable from current feed.")
        else:
            st.plotly_chart(_gauge(ratio, "Volume vs Expected", 0, 2.2, "x"), width='stretch')
        st.markdown(
            f"<div class='mono'>Regime: {regime} | Session: {session_label} | Progress: {progress:.0f}%</div>",
            unsafe_allow_html=True,
        )
    with col2:
        comp = volume_profile.get("components", pd.DataFrame()).copy()
        if not comp.empty:
            comp["ratio"] = comp["ratio"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "n/a")
            comp["observed_volume"] = comp["observed_volume"].map(lambda x: f"{x:,.0f}")
            comp["expected_volume"] = comp["expected_volume"].map(lambda x: f"{x:,.0f}")
            if "method" in comp.columns:
                method_map = {
                    "intraday_sum": "Intraday sum",
                    "intraday_filtered": "Intraday filtered",
                    "daily_live_fallback": "Daily live fallback",
                    "daily_close": "Daily close",
                    "daily_fallback": "Daily fallback",
                }
                comp["method"] = comp["method"].map(lambda x: method_map.get(str(x), str(x)))
            display_cols = ["ticker", "ratio", "observed_volume", "expected_volume"]
            if "method" in comp.columns:
                display_cols.append("method")
            st.dataframe(
                comp[display_cols],
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("No volume component data.")


def _render_rotation_panel(rotation_signal: dict) -> None:
    st.subheader("Sector Rotation")
    spread_d = rotation_signal.get("spread_daily", np.nan)
    spread_w = rotation_signal.get("spread_weekly", np.nan)
    spread_m = rotation_signal.get("spread_monthly", np.nan)

    c1, c2 = st.columns([1, 2])
    with c1:
        if np.isnan(spread_d):
            st.info("Rotation signal unavailable.")
        else:
            st.plotly_chart(_gauge(spread_d * 100, "Daily Cyc-Def Spread", -3, 3, "%"), width='stretch')
        st.markdown(f"**Regime:** {rotation_signal.get('regime', 'n/a')}")
        st.markdown(f"**Daily Spread:** {_format_pct(spread_d)}")
        st.markdown(f"**Weekly Spread:** {_format_pct(spread_w)}")
        st.markdown(f"**Monthly Spread:** {_format_pct(spread_m)}")
    with c2:
        history = rotation_signal.get("ratio_history", pd.DataFrame())
        if isinstance(history, pd.DataFrame) and not history.empty:
            st.plotly_chart(_rotation_ratio_chart(history), width='stretch')
        else:
            st.info("Rotation ratio history unavailable.")


def _yield_curve_shape_chart(yields_table: pd.DataFrame) -> go.Figure:
    if yields_table.empty:
        return go.Figure()

    tenor_order = {"13W": 1, "5Y": 2, "10Y": 3, "30Y": 4}
    df = yields_table.copy()
    df["order"] = df["tenor"].map(lambda x: tenor_order.get(str(x), 99))
    df = df.sort_values("order")

    fig = go.Figure(
        go.Scatter(
            x=df["tenor"],
            y=df["yield_pct"],
            mode="lines+markers+text",
            text=[f"{x:.2f}%" for x in df["yield_pct"]],
            textposition="top center",
            line=dict(color="#7dc1ff", width=2.5),
            marker=dict(color="#59d39a", size=8),
            name="Yield Curve",
        )
    )
    fig.update_layout(
        title="Current Yield Curve",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        yaxis=dict(title="Yield %", gridcolor="rgba(140,170,205,0.18)"),
        xaxis=dict(title="Tenor"),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=300,
    )
    return _apply_chart_theme(fig)


def _yield_spread_history_chart(spread_history: pd.DataFrame, primary_col: str | None) -> go.Figure:
    if spread_history.empty or not primary_col or primary_col not in spread_history:
        return go.Figure()

    name_map = {
        "s13w10y_bps": "13W-10Y Spread",
        "s5s30_bps": "5Y-30Y Spread",
        "s10s30_bps": "10Y-30Y Spread",
    }
    display_name = name_map.get(primary_col, str(primary_col))

    fig = go.Figure(
        go.Scatter(
            x=spread_history["date"],
            y=spread_history[primary_col],
            mode="lines",
            name=display_name,
            line=dict(color="#53d390", width=2.2),
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,110,110,0.8)", line_dash="dash")
    fig.update_layout(
        title=f"{display_name} History",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        yaxis=dict(title="Spread (bps)", gridcolor="rgba(140,170,205,0.18)"),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=300,
    )
    return _apply_chart_theme(fig)


def _render_yield_curve_panel(yield_curve: dict) -> None:
    st.subheader("Yield Curve + Treasury Futures")

    state = yield_curve.get("curve_state", "Unavailable")
    primary_name = yield_curve.get("primary_spread_name")
    primary_spread = yield_curve.get("primary_spread_bps", np.nan)
    week_delta = yield_curve.get("week_delta_bps", np.nan)
    month_delta = yield_curve.get("month_delta_bps", np.nan)
    week_trend = yield_curve.get("week_trend", "n/a")
    month_trend = yield_curve.get("month_trend", "n/a")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Curve State", state)
    with cols[1]:
        label = primary_name if primary_name else "Primary Spread"
        val = f"{primary_spread:+.1f} bps" if pd.notna(primary_spread) else "n/a"
        st.metric(label, val)
    with cols[2]:
        val = f"{week_delta:+.1f} bps" if pd.notna(week_delta) else "n/a"
        st.metric("1W Slope Change", val, delta=week_trend)
    with cols[3]:
        val = f"{month_delta:+.1f} bps" if pd.notna(month_delta) else "n/a"
        st.metric("1M Slope Change", val, delta=month_trend)

    c1, c2 = st.columns([1, 1])
    with c1:
        ytbl = yield_curve.get("yields_table", pd.DataFrame())
        if isinstance(ytbl, pd.DataFrame) and not ytbl.empty:
            st.plotly_chart(_yield_curve_shape_chart(ytbl), width='stretch')
            table = ytbl.copy()
            table["yield_pct"] = table["yield_pct"].map(lambda x: f"{x:.2f}%")
            st.dataframe(table[["tenor", "yield_pct", "source"]], width='stretch', hide_index=True)
        else:
            st.info("Yield tenor data unavailable.")

    with c2:
        spread_history = yield_curve.get("spread_history", pd.DataFrame())
        spread_col_map = {
            "13W-10Y": "s13w10y_bps",
            "5Y-30Y": "s5s30_bps",
            "10Y-30Y": "s10s30_bps",
        }
        primary_col = spread_col_map.get(primary_name)
        if isinstance(spread_history, pd.DataFrame) and not spread_history.empty and primary_col:
            st.plotly_chart(_yield_spread_history_chart(spread_history, primary_col), width='stretch')
        else:
            st.info("Spread history unavailable.")

        current_spreads = yield_curve.get("current_spreads", {})
        if current_spreads:
            spread_rows = pd.DataFrame(
                [{"spread": k, "bps": v} for k, v in current_spreads.items()]
            )
            spread_rows["bps"] = spread_rows["bps"].map(lambda x: f"{x:+.1f}")
            st.dataframe(spread_rows, width='stretch', hide_index=True)

    st.caption(yield_curve.get("front_end_note", ""))

    fut = yield_curve.get("futures_table", pd.DataFrame())
    if isinstance(fut, pd.DataFrame) and not fut.empty:
        show = fut.copy()
        show["last"] = show["last"].map(lambda x: f"{x:,.3f}")
        show["weekly_return"] = show["weekly_return"].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        show["monthly_return"] = show["monthly_return"].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        st.markdown("**Treasury Futures Focus (ZT/ZN/ZB/UB)**")
        st.dataframe(
            show[["contract", "ticker", "last", "weekly_return", "monthly_return", "trend"]],
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("Treasury futures table unavailable.")


def _factor_basket_bar_chart(summary: pd.DataFrame, col: str, title: str) -> go.Figure:
    if not isinstance(summary, pd.DataFrame) or summary.empty or col not in summary:
        return go.Figure()

    df = summary.dropna(subset=[col]).copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values(col, ascending=True)

    colors = ["#42d680" if x >= 0 else "#ff6f6f" for x in df[col]]
    fig = go.Figure(
        go.Bar(
            x=df[col] * 100,
            y=df["basket"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:+.2f}%" for x in df[col] * 100],
            textposition="outside",
            textfont=dict(color="#ffffff", size=12),
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Return %", gridcolor="rgba(140,170,205,0.18)", zerolinecolor="rgba(140,170,205,0.25)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=420,
    )
    return _apply_chart_theme(fig)


def _factor_basket_history_chart(history: pd.DataFrame, basket_name: str) -> go.Figure:
    if not isinstance(history, pd.DataFrame) or history.empty or "date" not in history or "basket_index" not in history:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["basket_index"],
            mode="lines",
            name=f"{basket_name} Index",
            line=dict(color="#66dca3", width=2.1),
        )
    )
    if "rel_vs_spy" in history:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["rel_vs_spy"],
                mode="lines",
                name="Relative vs SPY",
                line=dict(color="#7db3ff", width=1.9, dash="dot"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"{basket_name} Trend and Relative Performance",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title="Basket Index", gridcolor="rgba(140,170,205,0.18)"),
        yaxis2=dict(title="Relative vs SPY", overlaying="y", side="right", showgrid=False),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=320,
    )
    return _apply_chart_theme(fig)


def _render_factor_basket_panel(factor_baskets: dict) -> None:
    st.subheader("Sub-Factor Basket Rotation")

    summary = factor_baskets.get("summary", pd.DataFrame())
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        st.info("Sub-factor basket data unavailable.")
        return

    tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
    with tab1:
        st.plotly_chart(_factor_basket_bar_chart(summary, "daily", "Sub-Factor Baskets (1D)"), width='stretch')
    with tab2:
        st.plotly_chart(_factor_basket_bar_chart(summary, "weekly", "Sub-Factor Baskets (1W)"), width='stretch')
    with tab3:
        st.plotly_chart(_factor_basket_bar_chart(summary, "monthly", "Sub-Factor Baskets (1M)"), width='stretch')

    options = summary["basket"].astype(str).tolist()
    selected = st.selectbox("Inspect basket holdings", options=options, key="factor_basket_select")

    selected_row = summary[summary["basket"] == selected]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = selected_row["daily"].iloc[0] if not selected_row.empty else np.nan
        st.metric("Basket 1D", _format_pct(val) if pd.notna(val) else "n/a")
    with c2:
        val = selected_row["weekly"].iloc[0] if not selected_row.empty else np.nan
        st.metric("Basket 1W", _format_pct(val) if pd.notna(val) else "n/a")
    with c3:
        val = selected_row["monthly"].iloc[0] if not selected_row.empty else np.nan
        st.metric("Basket 1M", _format_pct(val) if pd.notna(val) else "n/a")
    with c4:
        avail = int(selected_row["holdings_available"].iloc[0]) if not selected_row.empty else 0
        total = int(selected_row["holdings_total"].iloc[0]) if not selected_row.empty else 0
        st.metric("Holdings Available", f"{avail}/{total}")

    history_map = factor_baskets.get("history", {})
    history = history_map.get(selected, pd.DataFrame()) if isinstance(history_map, dict) else pd.DataFrame()
    if isinstance(history, pd.DataFrame) and not history.empty:
        st.plotly_chart(_factor_basket_history_chart(history, selected), width='stretch')

    holdings_map = factor_baskets.get("holdings", {})
    holdings = holdings_map.get(selected, pd.DataFrame()) if isinstance(holdings_map, dict) else pd.DataFrame()

    st.markdown(f"**{selected} Holdings**")
    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        show = holdings.copy()
        for col in ["daily", "weekly", "monthly"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        if "available" in show.columns:
            show["available"] = show["available"].map(lambda x: "Yes" if bool(x) else "No")
        st.dataframe(show[["ticker", "name", "daily", "weekly", "monthly", "available"]], width='stretch', hide_index=True)
    else:
        st.info("No holdings data for this basket.")

    st.caption(str(factor_baskets.get("notes", "")))


def _render_sector_panels(sectors: pd.DataFrame) -> None:
    st.subheader("Sector Performance")
    if sectors.empty:
        st.warning("Sector data unavailable.")
        return
    tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
    with tab1:
        st.plotly_chart(_sector_bar_chart(sectors, "daily", "Best / Worst Sectors (1D)"), width='stretch')
    with tab2:
        st.plotly_chart(_sector_bar_chart(sectors, "weekly", "Best / Worst Sectors (1W)"), width='stretch')
    with tab3:
        st.plotly_chart(_sector_bar_chart(sectors, "monthly", "Best / Worst Sectors (1M)"), width='stretch')


def _render_sentiment_panel(sentiment: dict) -> None:
    st.subheader("Sentiment Monitor")
    score = float(sentiment.get("composite_score", 0))
    regime = sentiment.get("regime", "n/a")

    st.plotly_chart(_gauge(score, f"Composite Sentiment ({regime})", -100, 100), width='stretch')
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    comps = sentiment.get("components", {})
    rows = [
        ("VIX Regime", comps.get("vix_component", np.nan)),
        ("Sector Breadth", comps.get("breadth_component", np.nan)),
        ("Trend", comps.get("trend_component", np.nan)),
        ("Headline Tone", comps.get("news_component", np.nan)),
    ]
    for label, value in rows:
        color = _pct_color(value / 100 if pd.notna(value) else np.nan)
        value_text = "n/a" if pd.isna(value) else f"{value:+.0f}"
        st.markdown(f"<span style='color:{color}; font-weight:600'>{label}: {value_text}</span>", unsafe_allow_html=True)

    breadth = sentiment.get("breadth_ratio", np.nan)
    breadth_d = sentiment.get("breadth_daily", breadth)
    breadth_w = sentiment.get("breadth_weekly", np.nan)
    breadth_m = sentiment.get("breadth_monthly", np.nan)

    st.markdown(
        f"<div class='mono'>Breadth (D): {breadth * 100:.0f}% positive sectors</div>"
        if not np.isnan(breadth)
        else "<div class='mono'>Breadth unavailable</div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Breadth 1D", f"{breadth_d * 100:.0f}%" if pd.notna(breadth_d) else "n/a")
    with b2:
        st.metric("Breadth 1W", f"{breadth_w * 100:.0f}%" if pd.notna(breadth_w) else "n/a")
    with b3:
        st.metric("Breadth 1M", f"{breadth_m * 100:.0f}%" if pd.notna(breadth_m) else "n/a")


def _render_cta_panel(cta_proxy: dict) -> None:
    st.subheader("CTA Positioning Proxy")
    st.caption(
        "This is a trend-following proxy model built from liquid ETFs, not official CFTC positioning."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(_gauge(float(cta_proxy.get("net_score", 0.0)), "Net CTA Proxy", -100, 100), width='stretch')
    with c2:
        table = cta_proxy.get("table", pd.DataFrame()).copy()
        if table.empty:
            st.info("CTA proxy data unavailable.")
        else:
            table["last"] = table["last"].map(lambda x: f"{x:,.2f}")
            table["strength_z"] = table["strength_z"].map(lambda x: f"{x:+.2f}")
            table["model_score"] = table["model_score"].map(lambda x: f"{x:+.1f}")
            st.dataframe(
                table[["asset", "ticker", "stance", "strength_z", "model_score", "last"]],
                width='stretch',
                hide_index=True,
            )


def _regime_scores_chart(scores: dict[str, float]) -> go.Figure:
    if not scores:
        return go.Figure()
    df = pd.DataFrame([{"regime": k, "score": float(v)} for k, v in scores.items()]).sort_values("score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["score"],
            y=df["regime"],
            orientation="h",
            marker=dict(color="#7db3ff"),
            text=[f"{x:.0f}" for x in df["score"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Regime Probability Stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Score", gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=300,
    )
    return _apply_chart_theme(fig)


def _liquidity_real_rate_history_chart(history: pd.DataFrame) -> go.Figure:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["liquidity_stress_score"],
            mode="lines",
            name="Liquidity Stress Score",
            line=dict(color="#ff8f70", width=2.2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["real_rate_proxy_z"],
            mode="lines",
            name="Real-Rate Proxy (z)",
            line=dict(color="#64d9a2", width=2.0),
        ),
        secondary_y=True,
    )

    if "nominal_10y_yield" in history:
        fig.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["nominal_10y_yield"],
                mode="lines",
                name="10Y Nominal Yield %",
                line=dict(color="#7db3ff", width=1.6, dash="dot"),
            ),
            secondary_y=True,
        )

    fig.add_hline(y=50, line_dash="dot", line_color="rgba(246,205,97,0.8)")
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,120,120,0.8)")
    fig.update_layout(
        title="Liquidity Stress + Real-Rate History",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=340,
    )
    fig.update_yaxes(title_text="Liquidity Score", range=[0, 100], gridcolor="rgba(140,170,205,0.18)", secondary_y=False)
    fig.update_yaxes(title_text="Real-Rate z / 10Y %", secondary_y=True)
    return _apply_chart_theme(fig)


def _render_liquidity_real_rate_panel(liquidity_monitor: dict) -> None:
    st.subheader("Advanced Liquidity + Real-Rate Monitor")

    liq_score = liquidity_monitor.get("liquidity_stress_score", np.nan)
    liq_regime = liquidity_monitor.get("liquidity_regime", "Unavailable")
    real_z = liquidity_monitor.get("real_rate_proxy_z", np.nan)
    real_regime = liquidity_monitor.get("real_rate_regime", "Unavailable")
    nominal_10y = liquidity_monitor.get("nominal_10y_yield", np.nan)

    c1, c2 = st.columns([1, 1])
    with c1:
        if pd.notna(liq_score):
            st.plotly_chart(_gauge(float(liq_score), "Liquidity Stress", 0, 100), width='stretch')
        else:
            st.info("Liquidity score unavailable.")
        st.markdown(f"**Liquidity Regime:** {liq_regime}")
    with c2:
        if pd.notna(real_z):
            st.plotly_chart(_gauge(float(real_z), "Real-Rate Proxy (z)", -3, 3), width='stretch')
        else:
            st.info("Real-rate proxy unavailable.")
        st.markdown(f"**Real-Rate Regime:** {real_regime}")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("10Y Nominal", f"{nominal_10y:.2f}%" if pd.notna(nominal_10y) else "n/a")
    with m2:
        d1 = liquidity_monitor.get("daily_delta_score", np.nan)
        st.metric("1D Liquidity Delta", f"{d1:+.1f}" if pd.notna(d1) else "n/a", delta=str(liquidity_monitor.get("daily_trend", "n/a")))
    with m3:
        wk = liquidity_monitor.get("week_delta_score", np.nan)
        st.metric("1W Liquidity Delta", f"{wk:+.1f}" if pd.notna(wk) else "n/a", delta=str(liquidity_monitor.get("week_trend", "n/a")))
    with m4:
        mo = liquidity_monitor.get("month_delta_score", np.nan)
        st.metric("1M Liquidity Delta", f"{mo:+.1f}" if pd.notna(mo) else "n/a", delta=str(liquidity_monitor.get("month_trend", "n/a")))
    with m5:
        st.metric("Real-Rate Proxy", f"{real_z:+.2f} z" if pd.notna(real_z) else "n/a")

    history = liquidity_monitor.get("history", pd.DataFrame())
    if isinstance(history, pd.DataFrame) and not history.empty:
        st.plotly_chart(_liquidity_real_rate_history_chart(history), width='stretch')
    else:
        st.info("Liquidity/real-rate history unavailable.")

    details = liquidity_monitor.get("details_table", pd.DataFrame())
    if isinstance(details, pd.DataFrame) and not details.empty:
        show = details.copy()

        def _fmt_metric(row: pd.Series) -> str:
            val = row.get("value", np.nan)
            unit = str(row.get("unit", ""))
            if pd.isna(val):
                return "n/a"
            if unit == "%":
                return f"{float(val):+.2f}%"
            if unit == "x":
                return f"{float(val):.2f}x"
            if unit == "z":
                return f"{float(val):+.2f}"
            if unit == "ratio":
                return f"{float(val):.4f}"
            return f"{float(val):+.3f}"

        show["value"] = show.apply(_fmt_metric, axis=1)
        st.dataframe(show[["metric", "value", "signal"]], width='stretch', hide_index=True)

    st.caption(str(liquidity_monitor.get("notes", "")))


def _volatility_structure_history_chart(history: pd.DataFrame) -> go.Figure:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return go.Figure()

    df = history.copy()
    if "date" not in df.columns:
        date_candidates = [c for c in df.columns if str(c).strip().lower() in {"index", "datetime", "timestamp", "time"}]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: "date"})
        else:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "vix" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["vix"],
                mode="lines",
                name="VIX",
                line=dict(color="#7db3ff", width=2.0),
            ),
            secondary_y=False,
        )
    if "vix9d" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["vix9d"],
                mode="lines",
                name="VIX9D",
                line=dict(color="#66dca3", width=1.8),
            ),
            secondary_y=False,
        )
    if "vix3m" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["vix3m"],
                mode="lines",
                name="VIX3M",
                line=dict(color="#f6cd61", width=1.8),
            ),
            secondary_y=False,
        )
    if "slope_spread" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["slope_spread"],
                mode="lines",
                name="VIX - VIX3M",
                line=dict(color="#ff8f70", width=2.1, dash="dot"),
            ),
            secondary_y=True,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,125,125,0.8)", secondary_y=True)
    fig.update_layout(
        title="Volatility Term Structure History",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(gridcolor="rgba(140,170,205,0.18)"),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=340,
    )
    fig.update_yaxes(title_text="VIX Levels", gridcolor="rgba(140,170,205,0.18)", secondary_y=False)
    fig.update_yaxes(title_text="Spread (vol pts)", secondary_y=True)
    return _apply_chart_theme(fig)


def _render_volatility_structure_panel(vol_structure: dict) -> None:
    st.subheader("Volatility Structure Module")

    structure_regime = vol_structure.get("structure_regime", "Unavailable")
    stress_regime = vol_structure.get("stress_regime", "Unavailable")
    slope = vol_structure.get("slope_spread", np.nan)
    front = vol_structure.get("front_spread", np.nan)
    score = vol_structure.get("stress_score", np.nan)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Term Regime", str(structure_regime))
    with c2:
        st.metric("Stress Regime", str(stress_regime))
    with c3:
        st.metric("VIX - VIX3M", f"{slope:+.2f}" if pd.notna(slope) else "n/a")
    with c4:
        st.metric("VIX9D - VIX", f"{front:+.2f}" if pd.notna(front) else "n/a")
    with c5:
        st.metric("Vol Stress Score", f"{score:.0f}/100" if pd.notna(score) else "n/a")

    d1, d2, d3 = st.columns(3)
    with d1:
        delta = vol_structure.get("daily_slope_delta", np.nan)
        st.metric("1D Slope Delta", f"{delta:+.2f}" if pd.notna(delta) else "n/a")
    with d2:
        delta = vol_structure.get("week_slope_delta", np.nan)
        st.metric("1W Slope Delta", f"{delta:+.2f}" if pd.notna(delta) else "n/a", delta=str(vol_structure.get("week_trend", "n/a")))
    with d3:
        delta = vol_structure.get("month_slope_delta", np.nan)
        st.metric("1M Slope Delta", f"{delta:+.2f}" if pd.notna(delta) else "n/a", delta=str(vol_structure.get("month_trend", "n/a")))

    history = vol_structure.get("history", pd.DataFrame())
    if isinstance(history, pd.DataFrame) and not history.empty:
        st.plotly_chart(_volatility_structure_history_chart(history), width="stretch")
    else:
        st.info("Volatility term-structure history unavailable.")

    levels = pd.DataFrame(
        [
            {"metric": "VIX9D", "value": vol_structure.get("vix9d_level", np.nan)},
            {"metric": "VIX", "value": vol_structure.get("vix_level", np.nan)},
            {"metric": "VIX3M", "value": vol_structure.get("vix3m_level", np.nan)},
            {"metric": "VVIX", "value": vol_structure.get("vvix_level", np.nan)},
            {"metric": "MOVE", "value": vol_structure.get("move_level", np.nan)},
            {"metric": "Curve Ratio (VIX/VIX3M)", "value": vol_structure.get("curve_ratio", np.nan)},
        ]
    )
    levels["value"] = levels["value"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    st.dataframe(levels, width="stretch", hide_index=True)
    st.caption(str(vol_structure.get("notes", "")))


def _render_regime_panel(regime_engine: dict) -> None:
    st.subheader("Regime Engine")
    regime = regime_engine.get("regime", "Unavailable")
    confidence = regime_engine.get("confidence", np.nan)
    conviction = regime_engine.get("conviction", "n/a")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current Regime", str(regime))
    with c2:
        st.metric("Confidence", f"{confidence:.0f}/100" if pd.notna(confidence) else "n/a")
    with c3:
        st.metric("Conviction", str(conviction))

    scores = regime_engine.get("scores", {})
    if isinstance(scores, dict) and scores:
        st.plotly_chart(_regime_scores_chart(scores), width='stretch')

    components = regime_engine.get("components", pd.DataFrame())
    if isinstance(components, pd.DataFrame) and not components.empty:
        show = components.copy()
        show["value"] = show["value"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "n/a")
        st.dataframe(show[["factor", "value", "signal"]], width='stretch', hide_index=True)

    st.caption(str(regime_engine.get("notes", "")))


def _render_cross_asset_panel(cross_asset: dict) -> None:
    st.subheader("Cross-Asset Confirmation")
    bias = cross_asset.get("bias", "Neutral")
    ratio = cross_asset.get("confirmation_ratio", np.nan)
    status = cross_asset.get("status", "n/a")

    c1, c2 = st.columns([1, 2])
    with c1:
        if pd.notna(ratio):
            st.plotly_chart(_gauge(float(ratio * 100), "Confirmation", 0, 100, "%"), width='stretch')
        else:
            st.info("Confirmation ratio unavailable.")
    with c2:
        st.markdown(f"**Bias:** {bias}")
        st.markdown(f"**Status:** {status}")
        st.markdown(
            f"**Supportive Legs:** {int(cross_asset.get('supportive_count', 0))}/{int(cross_asset.get('available_count', 0))}"
        )
        contradictions = cross_asset.get("contradictions", [])
        if contradictions:
            st.markdown(f"**Contradictions:** {', '.join(contradictions)}")

    table = cross_asset.get("table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        show = table.copy()
        if "value" in show:
            show["value"] = show["value"].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        for flag in ["supports_bullish", "supports_bearish", "supports_bias"]:
            if flag in show:
                show[flag] = show[flag].map(
                    lambda x: "Yes" if (pd.notna(x) and bool(x)) else ("No" if pd.notna(x) else "n/a")
                )

        cols = ["asset", "signal", "value"]
        if "supports_bullish" in show:
            cols.append("supports_bullish")
        if "supports_bearish" in show:
            cols.append("supports_bearish")
        if "supports_bias" in show:
            cols.append("supports_bias")

        st.dataframe(show[cols], width='stretch', hide_index=True)


def _put_skew_chart(table: pd.DataFrame) -> go.Figure:
    if not isinstance(table, pd.DataFrame) or table.empty or "put_call_iv_spread" not in table.columns:
        return go.Figure()

    df = table.copy().dropna(subset=["put_call_iv_spread"])
    if df.empty:
        return go.Figure()
    df = df.sort_values("put_call_iv_spread", ascending=True)
    colors = ["#ff7d7d" if x >= 3 else ("#f6cd61" if x >= 1 else "#66dca3") for x in df["put_call_iv_spread"]]

    fig = go.Figure(
        go.Bar(
            x=df["put_call_iv_spread"],
            y=df["symbol"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:+.1f}" for x in df["put_call_iv_spread"]],
            textposition="outside",
            name="95p - 105c IV",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(140,170,205,0.6)")
    fig.add_vline(x=3, line_dash="dash", line_color="rgba(255,120,120,0.8)")
    fig.update_layout(
        title="Put Skew by Underlying (vol pts)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="IV Spread (95% Put - 105% Call)", gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=280,
    )
    return _apply_chart_theme(fig)


def _render_put_skew_panel(put_skew: dict) -> None:
    st.subheader("Put Skew Monitor")

    avg_skew = put_skew.get("avg_put_call_iv_spread", np.nan)
    avg_oi = put_skew.get("avg_put_call_oi_ratio", np.nan)
    avg_vol = put_skew.get("avg_put_call_vol_ratio", np.nan)
    regime = put_skew.get("regime", "Unavailable")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Skew Regime", str(regime))
    with c2:
        st.metric("Avg IV Skew", f"{avg_skew:+.2f}" if pd.notna(avg_skew) else "n/a")
    with c3:
        st.metric("Avg Put/Call OI", f"{avg_oi:.2f}" if pd.notna(avg_oi) else "n/a")
    with c4:
        st.metric("Avg Put/Call Vol", f"{avg_vol:.2f}" if pd.notna(avg_vol) else "n/a")

    table = put_skew.get("table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.plotly_chart(_put_skew_chart(table), width='stretch')

        show = table.copy()
        if "spot" in show.columns:
            show["spot"] = show["spot"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
        for col in ["put_iv_95", "call_iv_105"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")
        for col in ["put_strike", "call_strike"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
        if "put_call_iv_spread" in show.columns:
            show["put_call_iv_spread"] = show["put_call_iv_spread"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a")
        for col in ["put_call_oi_ratio", "put_call_vol_ratio"]:
            if col in show.columns:
                show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
        if "dte" in show.columns:
            show["dte"] = show["dte"].map(lambda x: int(x) if pd.notna(x) else "n/a")

        st.dataframe(
            show[
                [
                    "symbol",
                    "spot",
                    "expiry",
                    "dte",
                    "put_iv_95",
                    "call_iv_105",
                    "put_call_iv_spread",
                    "put_call_oi_ratio",
                    "put_call_vol_ratio",
                    "status",
                ]
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("Put skew data unavailable from option chain feed.")

    st.caption(str(put_skew.get("notes", "")))


def _render_event_risk_panel(event_risk: dict) -> None:
    st.subheader("Event Risk Overlay")
    score = event_risk.get("risk_score", np.nan)
    level = event_risk.get("risk_level", "Low")
    next_event = event_risk.get("next_event")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Risk Level", str(level))
    with c2:
        st.metric("Risk Score", f"{score:.0f}/100" if pd.notna(score) else "n/a")
    with c3:
        if isinstance(next_event, dict) and next_event:
            st.metric("Next Event", f"{next_event.get('event', 'n/a')} ({next_event.get('days_to_event', 'n/a')}d)")
        else:
            st.metric("Next Event", "n/a")

    events = event_risk.get("events", pd.DataFrame())
    if isinstance(events, pd.DataFrame) and not events.empty:
        show = events.copy()
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        show["risk_contrib"] = show["risk_contrib"].map(lambda x: f"{x:.1f}")
        st.dataframe(show[["event", "category", "date", "days_to_event", "risk_contrib"]].head(14), width='stretch', hide_index=True)

    st.caption(str(event_risk.get("notes", "")))


def _render_watchlist_panel(watchlist: dict) -> None:
    st.subheader("Watchlist")
    threshold = watchlist.get("threshold_pct", 0.0)
    st.caption(f"Mover threshold: {threshold:.1f}% daily move")

    movers = watchlist.get("movers", pd.DataFrame())
    if isinstance(movers, pd.DataFrame) and not movers.empty:
        tags = []
        for _, row in movers.head(4).iterrows():
            tags.append(f"{row.get('ticker', '')} {row.get('daily', 0.0) * 100:+.2f}%")
        if tags:
            st.markdown(f"**Active movers:** {' | '.join(tags)}")

    table = watchlist.get("table", pd.DataFrame())
    if isinstance(table, pd.DataFrame) and not table.empty:
        show = table.copy()
        show["last"] = show["last"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
        for col in ["daily", "weekly", "monthly"]:
            if col in show:
                show[col] = show[col].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        if "is_mover" in show:
            show["is_mover"] = show["is_mover"].map(lambda x: "Yes" if bool(x) else "No")
        st.dataframe(show[["ticker", "label", "last", "daily", "weekly", "monthly", "is_mover"]].head(20), width='stretch', hide_index=True)
    else:
        st.info("Watchlist data unavailable.")



def _near_low_screener_chart(table: pd.DataFrame, title: str) -> go.Figure:
    if not isinstance(table, pd.DataFrame) or table.empty:
        return go.Figure()

    df = table.copy().head(10)
    if "score" in df:
        df = df.sort_values("score", ascending=True)

    colors = ["#66dca3" if pd.notna(x) and x >= 60 else "#f6cd61" for x in df.get("score", pd.Series(dtype=float))]
    fig = go.Figure(
        go.Bar(
            x=df.get("score", pd.Series(dtype=float)),
            y=df.get("ticker", pd.Series(dtype=str)),
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:.0f}" if pd.notna(x) else "n/a" for x in df.get("score", pd.Series(dtype=float))],
            textposition="outside",
            name="Composite Score",
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Score", range=[0, 100], gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=320,
    )
    return _apply_chart_theme(fig)


def _render_near_52w_low_panel(screener: dict) -> None:
    st.subheader("Near 52-Week Lows With Momentum + Relative Strength")

    coverage = screener.get("coverage", pd.DataFrame())
    if isinstance(coverage, pd.DataFrame) and not coverage.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Indexes", str(len(coverage)))
        with c2:
            st.metric("Universe", f"{int(coverage['universe_size'].sum()):,}" if "universe_size" in coverage else "n/a")
        with c3:
            st.metric("Scanned", f"{int(coverage['scanned'].sum()):,}" if "scanned" in coverage else "n/a")
        with c4:
            st.metric("Qualified", f"{int(coverage['qualified'].sum()):,}" if "qualified" in coverage else "n/a")
        with c5:
            st.metric("Strict", f"{int(coverage['strict_qualified'].sum()):,}" if "strict_qualified" in coverage else "n/a")

    labels = screener.get("index_labels", {}) if isinstance(screener.get("index_labels", {}), dict) else {}
    results = screener.get("results", {}) if isinstance(screener.get("results", {}), dict) else {}
    top_n = int(screener.get("top_n", 10) or 10)

    order = ["russell2000", "sp500", "nasdaq100"]
    available = [k for k in order if isinstance(results.get(k), pd.DataFrame)]

    if not available:
        st.info("Screener data unavailable.")
        return

    if isinstance(coverage, pd.DataFrame) and "used_relaxed" in coverage.columns and bool(coverage["used_relaxed"].fillna(False).any()):
        st.info("Some indexes have zero strict matches right now, so fallback ranking is shown.")
    if isinstance(coverage, pd.DataFrame) and "used_broad" in coverage.columns and bool(coverage["used_broad"].fillna(False).any()):
        st.info("For indexes with no names inside the 10% band, the panel shows nearest lows outside the band as context.")

    tabs = st.tabs([str(labels.get(k, k)) for k in available])
    for tab, key in zip(tabs, available):
        with tab:
            table = results.get(key, pd.DataFrame())
            if not isinstance(table, pd.DataFrame) or table.empty:
                st.info("No candidates met the filter right now.")
                continue

            st.plotly_chart(_near_low_screener_chart(table, f"{labels.get(key, key)} Top {top_n} Candidates"), width='stretch')

            show = table.copy().head(top_n)
            if "last" in show:
                show["last"] = show["last"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
            for col in ["distance_to_52w_low", "ret_5d", "ret_1m", "ret_3m", "rs_1m", "rs_3m"]:
                if col in show:
                    show[col] = show[col].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
            if "score" in show:
                show["score"] = show["score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
            if "strict_match" in show:
                show["strict_match"] = show["strict_match"].map(lambda x: "Yes" if bool(x) else "No")

            display_cols = ["ticker", "name", "last", "distance_to_52w_low", "ret_5d", "ret_1m", "ret_3m", "rs_1m", "score"]
            if "strict_match" in show:
                display_cols.append("strict_match")

            st.dataframe(
                show[display_cols],
                width='stretch',
                hide_index=True,
            )

    notes = str(screener.get("notes", ""))
    if notes:
        st.caption(notes)

def _render_alert_routing_panel(routing_status: dict) -> None:
    st.subheader("Alert Routing")
    enabled = bool(routing_status.get("enabled", False))
    destinations = routing_status.get("destinations", [])
    dest_text = ", ".join(destinations) if destinations else "none"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Routing Enabled", "Yes" if enabled else "No")
    with c2:
        st.metric("Sent This Run", str(routing_status.get("sent_count", 0)))
    with c3:
        st.metric("Skipped", str(routing_status.get("skipped_count", 0)))

    st.markdown(f"**Destinations:** {dest_text}")
    st.markdown(f"**Min Severity:** {routing_status.get('min_severity', 'n/a')} | **Cooldown:** {routing_status.get('cooldown_min', 'n/a')} min")

    if routing_status.get("error_count", 0):
        for err in routing_status.get("errors", []):
            st.warning(str(err))

def _signal_quality_hit_chart(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return go.Figure()

    df = summary.copy().head(10)
    df["label"] = df["source"].astype(str) + " | " + df["direction"].astype(str)
    colors = ["#56d392" if pd.notna(v) and v >= 50 else "#ff7d7d" for v in df["hit_rate_5d"]]

    fig = go.Figure(
        go.Bar(
            x=df["hit_rate_5d"],
            y=df["label"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:.0f}%" if pd.notna(x) else "n/a" for x in df["hit_rate_5d"]],
            textposition="outside",
            name="5D Hit Rate",
        )
    )
    fig.add_vline(x=50, line_dash="dot", line_color="rgba(246,205,97,0.9)")
    fig.update_layout(
        title="Signal Quality (5D Hit Rate)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Hit Rate %", range=[0, 100], gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=320,
    )
    return _apply_chart_theme(fig)


def _render_signal_quality_panel(signal_quality: dict) -> None:
    st.subheader("Signal Quality Tracker")
    overall = signal_quality.get("overall_hit_rate_5d", np.nan)
    edge = signal_quality.get("edge_state", "Unavailable")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall 5D Hit Rate", f"{overall:.1f}%" if pd.notna(overall) else "n/a")
        st.metric("Edge State", str(edge))
    with c2:
        st.caption(str(signal_quality.get("notes", "")))

    summary = signal_quality.get("summary", pd.DataFrame())
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        st.plotly_chart(_signal_quality_hit_chart(summary), width='stretch')
        show = summary.copy()
        for col in ["hit_rate_1d", "hit_rate_3d", "hit_rate_5d", "avg_aligned_5d", "avg_max_adverse_5d"]:
            if col in show:
                show[col] = show[col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "n/a")
        if "avg_trigger_score" in show:
            show["avg_trigger_score"] = show["avg_trigger_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "n/a")
        st.dataframe(
            show[
                [
                    "source",
                    "direction",
                    "signals",
                    "evaluated_5d",
                    "hit_rate_1d",
                    "hit_rate_3d",
                    "hit_rate_5d",
                    "avg_aligned_5d",
                    "avg_max_adverse_5d",
                    "avg_trigger_score",
                ]
            ],
            width='stretch',
            hide_index=True,
        )

    trades = signal_quality.get("trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        show = trades.copy()
        if "trigger_date" in show.columns:
            show["trigger_date"] = pd.to_datetime(show["trigger_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ["ret_1d", "ret_3d", "ret_5d", "aligned_1d", "aligned_3d", "aligned_5d", "max_adverse_5d"]:
            if col in show:
                show[col] = show[col].map(lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "n/a")
        st.markdown("**Recent Trigger Outcomes**")
        st.dataframe(
            show[
                [
                    "trigger_date",
                    "source",
                    "sector",
                    "ticker",
                    "direction",
                    "state",
                    "ret_1d",
                    "ret_3d",
                    "ret_5d",
                    "aligned_5d",
                    "max_adverse_5d",
                ]
            ].head(30),
            width='stretch',
            hide_index=True,
        )


def _sector_heatmap_chart(sector_flow: dict) -> go.Figure:
    table = sector_flow.get("heatmap_table", pd.DataFrame())
    if not isinstance(table, pd.DataFrame) or table.empty:
        return go.Figure()

    df = table.copy().sort_values("daily", ascending=False)
    z = df[["daily", "weekly", "monthly"]].to_numpy() * 100
    text = [[f"{val:+.2f}%" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Daily", "Weekly", "Monthly"],
            y=df["sector"],
            colorscale=[[0.0, "#b94444"], [0.45, "#3a5677"], [0.5, "#314f70"], [1.0, "#39b777"]],
            zmid=0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title="Sector Heatmap (1D / 1W / 1M)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=440,
    )
    return _apply_chart_theme(fig)


def _render_sector_flow_panel(sector_flow: dict) -> None:
    st.subheader("Sector Heatmap + Rotation Flow")
    flow_score = sector_flow.get("flow_shift_score", np.nan)
    flow_regime = sector_flow.get("flow_regime", "Unavailable")
    turnover = sector_flow.get("turnover_ratio", np.nan)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Flow Regime", str(flow_regime))
    with c2:
        st.metric("Flow Shift Score", f"{flow_score:.0f}/100" if pd.notna(flow_score) else "n/a")
    with c3:
        st.metric("Leader Turnover", f"{turnover * 100:.0f}%" if pd.notna(turnover) else "n/a")

    st.plotly_chart(_sector_heatmap_chart(sector_flow), width='stretch')

    lt = sector_flow.get("leaders_today", pd.DataFrame())
    lp = sector_flow.get("leaders_prev", pd.DataFrame())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Leaders Today**")
        if isinstance(lt, pd.DataFrame) and not lt.empty:
            show = lt.copy()
            show["daily"] = show["daily"].map(lambda x: f"{x * 100:+.2f}%")
            st.dataframe(show, width='stretch', hide_index=True)
        else:
            st.info("Unavailable.")
    with col2:
        st.markdown("**Top Leaders Prior Session**")
        if isinstance(lp, pd.DataFrame) and not lp.empty:
            show = lp.copy()
            show["daily"] = show["daily"].map(lambda x: f"{x * 100:+.2f}%")
            st.dataframe(show, width='stretch', hide_index=True)
        else:
            st.info("Unavailable.")

    st.caption(str(sector_flow.get("notes", "")))


def _rotation_flow_sankey_chart(edges: pd.DataFrame, title: str) -> go.Figure:
    if not isinstance(edges, pd.DataFrame) or edges.empty:
        return go.Figure()
    needed = {"source_node", "target_node", "weight", "status", "source", "target"}
    if not needed.issubset(set(edges.columns)):
        return go.Figure()

    df = edges.copy()
    nodes = list(dict.fromkeys(df["source_node"].astype(str).tolist() + df["target_node"].astype(str).tolist()))
    idx_map = {name: i for i, name in enumerate(nodes)}

    status_colors = {
        "Retained": "rgba(86, 211, 146, 0.70)",
        "New": "rgba(125, 179, 255, 0.70)",
        "Dropped": "rgba(255, 125, 125, 0.70)",
    }
    link_colors = [status_colors.get(str(s), "rgba(140,170,205,0.55)") for s in df["status"]]
    custom = [f"{src} -> {tgt}<br>Status: {status}" for src, tgt, status in zip(df["source"], df["target"], df["status"])]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=nodes,
                color="rgba(12, 32, 52, 0.95)",
                line=dict(color="rgba(140,170,205,0.40)", width=1),
                pad=16,
                thickness=14,
            ),
            link=dict(
                source=[idx_map[str(x)] for x in df["source_node"]],
                target=[idx_map[str(x)] for x in df["target_node"]],
                value=[float(x) if pd.notna(x) else 0.0 for x in df["weight"]],
                color=link_colors,
                customdata=custom,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=36, b=8),
        font=dict(color="#ffffff", family="Space Grotesk", size=12),
        height=380,
    )
    return fig


def _render_rotation_flow_map_panel(rotation_flow_map: dict) -> None:
    st.subheader("Sector / Factor Rotation Flow Map")

    sector_turn = _as_float(rotation_flow_map.get("sector_turnover"))
    factor_turn = _as_float(rotation_flow_map.get("factor_turnover"))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Sector Leader Turnover", f"{sector_turn * 100:.0f}%" if pd.notna(sector_turn) else "n/a")
    with c2:
        st.metric("Factor Leader Turnover", f"{factor_turn * 100:.0f}%" if pd.notna(factor_turn) else "n/a")

    sector_edges = rotation_flow_map.get("sector_edges", pd.DataFrame())
    factor_edges = rotation_flow_map.get("factor_edges", pd.DataFrame())
    tab1, tab2 = st.tabs(["Sectors", "Factors"])

    with tab1:
        if isinstance(sector_edges, pd.DataFrame) and not sector_edges.empty:
            st.plotly_chart(_rotation_flow_sankey_chart(sector_edges, "Sector Leadership Flow (Prev -> Today)"), width="stretch")
            show = sector_edges.copy()
            if "rank_change" in show.columns:
                show["rank_change"] = show["rank_change"].map(lambda x: f"{int(x):+d}" if pd.notna(x) else "n/a")
            st.dataframe(
                show[["source", "target", "status", "prev_rank", "today_rank", "rank_change"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Sector flow edges unavailable.")

    with tab2:
        if isinstance(factor_edges, pd.DataFrame) and not factor_edges.empty:
            st.plotly_chart(_rotation_flow_sankey_chart(factor_edges, "Factor Leadership Flow (Prev -> Today)"), width="stretch")
            show = factor_edges.copy()
            if "rank_change" in show.columns:
                show["rank_change"] = show["rank_change"].map(lambda x: f"{int(x):+d}" if pd.notna(x) else "n/a")
            st.dataframe(
                show[["source", "target", "status", "prev_rank", "today_rank", "rank_change"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Factor flow edges unavailable.")

    st.caption(str(rotation_flow_map.get("notes", "")))


def _render_headlines(headlines: list[dict]) -> None:
    st.subheader("Market Headlines")
    if not headlines:
        st.info("Headline feed unavailable.")
        return

    for item in headlines[:8]:
        title = html.escape(str(item.get("title", "n/a")))
        publisher = html.escape(str(item.get("publisher", "Unknown")))
        published = item.get("published")
        ts = published.strftime("%Y-%m-%d %I:%M %p ET") if isinstance(published, datetime) else "time n/a"
        link = item.get("link")

        if link:
            safe_link = html.escape(str(link), quote=True)
            st.markdown(
                f"<div class='headline-item'><a class='headline-link' href='{safe_link}' target='_blank'>{title}</a><div class='mono'>{publisher} | {ts}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='headline-item'><div style='font-weight:600'>{title}</div><div class='mono'>{publisher} | {ts}</div></div>",
                unsafe_allow_html=True,
            )


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _setup_tier(score: float) -> str:
    if score >= 75:
        return "A (high conviction)"
    if score >= 60:
        return "B (tradeable)"
    if score >= 45:
        return "C (watch)"
    return "D (pass)"


def _has_confirmed_capitulation(log: pd.DataFrame, direction: str) -> bool:
    if not isinstance(log, pd.DataFrame) or log.empty:
        return False
    required = {"direction", "state"}
    if not required.issubset(set(log.columns)):
        return False
    show = log.copy()
    if "trigger_date" in show.columns:
        show["trigger_date"] = pd.to_datetime(show["trigger_date"], errors="coerce")
        show = show.sort_values("trigger_date")
    show = show.tail(20)
    want = str(direction).strip().lower()
    return bool(
        (
            show["direction"].astype(str).str.lower().str.contains(want, na=False)
            & show["state"].astype(str).str.lower().eq("confirmed")
        ).any()
    )


def _format_trigger_date(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "n/a"
    return ts.strftime("%Y-%m-%d")


def _last_condition_timestamp(history_df: pd.DataFrame, mask: pd.Series) -> object:
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return None
    if not isinstance(mask, pd.Series) or mask.empty:
        return None

    if "timestamp_et" in history_df.columns:
        time_col = "timestamp_et"
    elif "ts_utc" in history_df.columns:
        time_col = "ts_utc"
    else:
        return None

    work = history_df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    aligned_mask = mask.reindex(work.index, fill_value=False).astype(bool)
    matched = work.loc[aligned_mask & work[time_col].notna(), time_col]
    if matched.empty:
        return None
    return matched.max()


def _last_cap_trigger_date(log: pd.DataFrame, direction: str) -> object:
    if not isinstance(log, pd.DataFrame) or log.empty:
        return None
    required = {"direction", "trigger_date"}
    if not required.issubset(set(log.columns)):
        return None
    show = log.copy()
    show["trigger_date"] = pd.to_datetime(show["trigger_date"], errors="coerce")
    want = str(direction).strip().lower()
    filt = show["direction"].astype(str).str.lower().str.contains(want, na=False)
    matched = show.loc[filt & show["trigger_date"].notna(), "trigger_date"]
    if matched.empty:
        return None
    return matched.max()


def _build_actionable_playbook(payload: dict, history_df: pd.DataFrame) -> pd.DataFrame:
    sentiment = payload.get("sentiment", {})
    volume = payload.get("volume_profile", {})
    rotation = payload.get("rotation_signal", {})
    cta = payload.get("cta_proxy", {})
    liquidity = payload.get("liquidity_monitor", {})
    capitulation = payload.get("capitulation", {})
    put_skew = payload.get("put_skew", {})
    snapshot = payload.get("snapshot", {})

    sent = _as_float(sentiment.get("composite_score"))
    breadth = _as_float(sentiment.get("breadth_ratio"))
    vol_ratio = _as_float(volume.get("ratio"))
    rot_d = _as_float(rotation.get("spread_daily"))
    cta_n = _as_float(cta.get("net_score"))
    liq = _as_float(liquidity.get("liquidity_stress_score"))
    real_z = _as_float(liquidity.get("real_rate_proxy_z"))
    down_cap = _as_float(capitulation.get("downside_score"))
    up_cap = _as_float(capitulation.get("upside_score"))
    vix = _as_float(snapshot.get("^VIX", {}).get("last"))

    cap_log = capitulation.get("signal_log", pd.DataFrame())
    down_confirmed = _has_confirmed_capitulation(cap_log, "downside")
    up_confirmed = _has_confirmed_capitulation(cap_log, "upside")
    skew_regime = str(put_skew.get("regime", "")).lower()

    h = history_df.copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
    trend_long_date = None
    trend_short_date = None
    if not h.empty:
        for col in ["sentiment_score", "breadth_ratio", "rotation_daily", "volume_ratio", "cta_net"]:
            if col in h.columns:
                h[col] = pd.to_numeric(h[col], errors="coerce")
        long_mask = pd.Series(False, index=h.index)
        short_mask = pd.Series(False, index=h.index)
        needed_cols = {"sentiment_score", "breadth_ratio", "rotation_daily"}
        if needed_cols.issubset(set(h.columns)):
            long_mask = (
                (h["sentiment_score"] >= 20)
                & (h["breadth_ratio"] >= 0.55)
                & (h["rotation_daily"] >= 0.0)
            )
            short_mask = (
                (h["sentiment_score"] <= -20)
                & (h["breadth_ratio"] <= 0.45)
                & (h["rotation_daily"] <= 0.0)
            )
            if "volume_ratio" in h.columns:
                long_mask = long_mask & (h["volume_ratio"] >= 1.0)
                short_mask = short_mask & (h["volume_ratio"] >= 1.0)
        trend_long_date = _last_condition_timestamp(h, long_mask)
        trend_short_date = _last_condition_timestamp(h, short_mask)

    cap_down_date = _last_cap_trigger_date(cap_log, "downside")
    cap_up_date = _last_cap_trigger_date(cap_log, "upside")

    trend_long = 30.0
    trend_long += 22 if pd.notna(sent) and sent >= 20 else (-12 if pd.notna(sent) and sent <= -20 else 0)
    trend_long += 14 if pd.notna(breadth) and breadth >= 0.55 else (-8 if pd.notna(breadth) and breadth <= 0.45 else 0)
    trend_long += 14 if pd.notna(rot_d) and rot_d >= 0.002 else (-8 if pd.notna(rot_d) and rot_d <= -0.002 else 0)
    trend_long += 10 if pd.notna(cta_n) and cta_n >= 20 else (-7 if pd.notna(cta_n) and cta_n <= -20 else 0)
    trend_long += 12 if pd.notna(liq) and liq <= 45 else (-10 if pd.notna(liq) and liq >= 65 else 0)
    trend_long += 6 if pd.notna(real_z) and real_z <= 0.50 else (-5 if pd.notna(real_z) and real_z >= 1.20 else 0)
    trend_long += 4 if pd.notna(vol_ratio) and vol_ratio >= 1.0 else (-2 if pd.notna(vol_ratio) and vol_ratio < 0.80 else 0)

    trend_short = 30.0
    trend_short += 22 if pd.notna(sent) and sent <= -20 else (-12 if pd.notna(sent) and sent >= 20 else 0)
    trend_short += 14 if pd.notna(breadth) and breadth <= 0.45 else (-8 if pd.notna(breadth) and breadth >= 0.55 else 0)
    trend_short += 14 if pd.notna(rot_d) and rot_d <= -0.002 else (-8 if pd.notna(rot_d) and rot_d >= 0.002 else 0)
    trend_short += 10 if pd.notna(cta_n) and cta_n <= -20 else (-7 if pd.notna(cta_n) and cta_n >= 20 else 0)
    trend_short += 12 if pd.notna(liq) and liq >= 55 else (-10 if pd.notna(liq) and liq <= 35 else 0)
    trend_short += 6 if pd.notna(real_z) and real_z >= 0.80 else (-5 if pd.notna(real_z) and real_z <= -0.30 else 0)
    trend_short += 4 if pd.notna(vol_ratio) and vol_ratio >= 1.0 else (-2 if pd.notna(vol_ratio) and vol_ratio < 0.80 else 0)

    cap_long = 20.0
    cap_long += 28 if pd.notna(down_cap) and down_cap >= 60 else (10 if pd.notna(down_cap) and down_cap >= 45 else -8)
    cap_long += 16 if down_confirmed else 0
    cap_long += 10 if pd.notna(sent) and sent <= -20 else 0
    cap_long += 8 if pd.notna(vix) and vix >= 22 else 0
    cap_long += 8 if ("crash" in skew_regime or "downside" in skew_regime) else 0
    cap_long += 6 if pd.notna(cta_n) and cta_n <= -50 else 0
    cap_long += -10 if pd.notna(liq) and liq >= 75 else 0

    ex_short = 20.0
    ex_short += 28 if pd.notna(up_cap) and up_cap >= 60 else (10 if pd.notna(up_cap) and up_cap >= 45 else -8)
    ex_short += 16 if up_confirmed else 0
    ex_short += 10 if pd.notna(sent) and sent >= 20 else 0
    ex_short += 8 if pd.notna(vix) and vix <= 16 else 0
    ex_short += 8 if ("upside call demand" in skew_regime) else 0
    ex_short += 6 if pd.notna(cta_n) and cta_n >= 50 else 0
    ex_short += -10 if pd.notna(liq) and liq <= 25 else 0

    rows = [
        {
            "playbook": "Trend Continuation Long",
            "bias": "Long",
            "score": _clip_score(trend_long),
            "why_now": f"Sent {sent:+.0f} | Breadth {breadth * 100:.0f}% | Rotation {_format_pct(rot_d)}",
            "trigger": "Break/hold above intraday high with breadth >= 55%.",
            "last_trigger_date": _format_trigger_date(trend_long_date),
            "invalidation": "Sentiment falls below +5 or cyclical-defensive spread flips negative.",
        },
        {
            "playbook": "Trend Continuation Short",
            "bias": "Short",
            "score": _clip_score(trend_short),
            "why_now": f"Sent {sent:+.0f} | Breadth {breadth * 100:.0f}% | Rotation {_format_pct(rot_d)}",
            "trigger": "Break/hold below intraday low with breadth <= 45%.",
            "last_trigger_date": _format_trigger_date(trend_short_date),
            "invalidation": "Sentiment rises above -5 or rotation turns positive.",
        },
        {
            "playbook": "Downside Capitulation Reversal",
            "bias": "Long",
            "score": _clip_score(cap_long),
            "why_now": f"Cap score {down_cap:.0f} | VIX {vix:.1f} | Confirmed {down_confirmed}",
            "trigger": "Downside trigger >= 60 plus confirmed reversal within 1-3 bars.",
            "last_trigger_date": _format_trigger_date(cap_down_date),
            "invalidation": "No follow-through after confirmation or severe liquidity stress keeps rising.",
        },
        {
            "playbook": "Upside Exhaustion Fade",
            "bias": "Short",
            "score": _clip_score(ex_short),
            "why_now": f"Exhaust score {up_cap:.0f} | VIX {vix:.1f} | Confirmed {up_confirmed}",
            "trigger": "Upside trigger >= 60 plus confirmed reversal within 1-3 bars.",
            "last_trigger_date": _format_trigger_date(cap_up_date),
            "invalidation": "No downside follow-through after confirmation or breadth re-accelerates.",
        },
    ]

    table = pd.DataFrame(rows)
    table["tier"] = table["score"].map(_setup_tier)
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    return table


def _signal_shift_chart(events: pd.DataFrame) -> go.Figure:
    if not isinstance(events, pd.DataFrame) or events.empty:
        return go.Figure()

    df = events.copy().head(10).sort_values("shock", ascending=True)
    colors = ["#56d392" if x >= 0 else "#ff7d7d" for x in df["shock"]]

    fig = go.Figure(
        go.Bar(
            x=df["shock"],
            y=df["metric"],
            orientation="h",
            marker=dict(color=colors),
            text=df["delta_text"],
            textposition="outside",
            name="Shock",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(140,170,205,0.7)")
    fig.update_layout(
        title="Latest Signal Shifts",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        xaxis=dict(title="Shock (z-score proxy)", gridcolor="rgba(140,170,205,0.18)"),
        yaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=320,
    )
    return _apply_chart_theme(fig)


def _build_signal_change_snapshot(history_df: pd.DataFrame) -> dict:
    if not isinstance(history_df, pd.DataFrame) or history_df.empty or len(history_df) < 2:
        return {"pulse": "Unavailable", "pulse_score": np.nan, "events": pd.DataFrame(), "current_time": None}

    df = history_df.copy()
    if "timestamp_et" in df.columns:
        df = df.sort_values("timestamp_et")
    elif "ts_utc" in df.columns:
        df = df.sort_values("ts_utc")
    df = df.tail(72).reset_index(drop=True)

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    specs = [
        ("sentiment_score", "Sentiment", 12.0, 1, "pts"),
        ("breadth_ratio", "Breadth", 0.06, 1, "ratio"),
        ("volume_ratio", "Volume Ratio", 0.15, 1, "x"),
        ("cta_net", "CTA Proxy", 10.0, 1, "pts"),
        ("rotation_daily", "Rotation Spread", 0.0035, 1, "pct"),
        ("vix", "VIX", 1.20, -1, "pts"),
        ("spx_change", "S&P 500", 0.0025, 1, "pct"),
        ("ndx_change", "Nasdaq", 0.0030, 1, "pct"),
    ]

    rows: list[dict] = []
    pulse_score = 0.0
    shift_count = 0

    for col, label, threshold, orientation, unit in specs:
        if col not in df.columns:
            continue
        cur_val = _as_float(curr.get(col))
        prev_val = _as_float(prev.get(col))
        if pd.isna(cur_val) or pd.isna(prev_val):
            continue

        delta = float(cur_val - prev_val)
        diff_series = pd.to_numeric(df[col], errors="coerce").diff().dropna()
        sigma = _as_float(diff_series.std(ddof=0)) if not diff_series.empty else np.nan
        z_score = delta / sigma if pd.notna(sigma) and sigma > 1e-9 else np.nan
        intensity = abs(delta) / threshold if threshold > 0 else np.nan
        is_shift = bool(abs(delta) >= threshold or (pd.notna(z_score) and abs(z_score) >= 1.5))

        risk_impulse = orientation * float(np.clip(delta / threshold, -3.0, 3.0))
        if is_shift:
            pulse_score += risk_impulse
            shift_count += 1

        if unit == "pct":
            delta_text = f"{delta * 100:+.2f}%"
        elif unit == "x":
            delta_text = f"{delta:+.2f}x"
        elif unit == "ratio":
            delta_text = f"{delta * 100:+.1f} pp"
        else:
            delta_text = f"{delta:+.2f}"

        if not is_shift:
            state = "Steady"
        else:
            state = "Risk-on impulse" if risk_impulse > 0 else "Risk-off impulse"

        shock = z_score if pd.notna(z_score) else (np.sign(delta) * intensity)
        rows.append(
            {
                "metric": label,
                "current": cur_val,
                "previous": prev_val,
                "delta": delta,
                "delta_text": delta_text,
                "z_score": z_score,
                "shock": float(shock) if pd.notna(shock) else 0.0,
                "state": state,
                "is_shift": is_shift,
            }
        )

    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["is_shift", "shock"], ascending=[False, False]).reset_index(drop=True)

    if shift_count == 0:
        pulse = "Stable / no major shift"
    elif pulse_score >= 2.0:
        pulse = "Risk-on impulse"
    elif pulse_score <= -2.0:
        pulse = "Risk-off impulse"
    else:
        pulse = "Mixed impulse"

    current_time = curr.get("timestamp_et") if "timestamp_et" in curr else curr.get("ts_utc")
    return {
        "pulse": pulse,
        "pulse_score": pulse_score,
        "shift_count": shift_count,
        "events": events,
        "current_time": current_time,
    }


def _bucket_sentiment(value: float) -> str:
    if pd.isna(value):
        return "sent:unknown"
    if value >= 25:
        return "sent:strong_on"
    if value >= 5:
        return "sent:on"
    if value <= -25:
        return "sent:strong_off"
    if value <= -5:
        return "sent:off"
    return "sent:neutral"


def _bucket_volume(value: float) -> str:
    if pd.isna(value):
        return "vol:unknown"
    if value >= 1.15:
        return "vol:heavy"
    if value <= 0.90:
        return "vol:light"
    return "vol:normal"


def _bucket_vix(value: float) -> str:
    if pd.isna(value):
        return "vix:unknown"
    if value >= 22:
        return "vix:high"
    if value <= 16:
        return "vix:low"
    return "vix:mid"


def _bucket_breadth(value: float) -> str:
    if pd.isna(value):
        return "breadth:unknown"
    if value >= 0.60:
        return "breadth:strong"
    if value <= 0.40:
        return "breadth:weak"
    return "breadth:mixed"


def _build_regime_conditioned_probabilities(payload: dict, history_df: pd.DataFrame, horizon_bars: int = 4, min_samples: int = 24) -> dict:
    needed = {"spx_change", "ndx_change", "sentiment_score", "volume_ratio", "vix", "breadth_ratio"}
    if not isinstance(history_df, pd.DataFrame) or history_df.empty or not needed.issubset(set(history_df.columns)):
        return {"available": False, "reason": "Not enough stored history for conditioned probabilities."}

    df = history_df.copy()
    if "timestamp_et" in df.columns:
        df = df.sort_values("timestamp_et")
    elif "ts_utc" in df.columns:
        df = df.sort_values("ts_utc")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["state_full"] = (
        df["sentiment_score"].map(_bucket_sentiment)
        + " | "
        + df["volume_ratio"].map(_bucket_volume)
        + " | "
        + df["vix"].map(_bucket_vix)
        + " | "
        + df["breadth_ratio"].map(_bucket_breadth)
    )
    df["state_coarse"] = df["sentiment_score"].map(_bucket_sentiment) + " | " + df["volume_ratio"].map(_bucket_volume)

    df["fwd_spx"] = df["spx_change"].shift(-horizon_bars) - df["spx_change"]
    df["fwd_ndx"] = df["ndx_change"].shift(-horizon_bars) - df["ndx_change"]
    valid = df.dropna(subset=["fwd_spx", "fwd_ndx", "state_full", "state_coarse"]).copy()
    if len(valid) < max(30, min_samples):
        return {"available": False, "reason": "Need more snapshots before probability estimates become reliable."}

    def _aggregate(state_col: str) -> pd.DataFrame:
        grouped_rows = []
        for state, grp in valid.groupby(state_col):
            samples = int(len(grp))
            if samples < 8:
                continue
            grouped_rows.append(
                {
                    "state": str(state),
                    "samples": samples,
                    "spx_up_prob": float((grp["fwd_spx"] > 0).mean() * 100.0),
                    "ndx_up_prob": float((grp["fwd_ndx"] > 0).mean() * 100.0),
                    "spx_median_move": float(grp["fwd_spx"].median() * 100.0),
                    "ndx_median_move": float(grp["fwd_ndx"].median() * 100.0),
                }
            )
        out = pd.DataFrame(grouped_rows)
        if not out.empty:
            out = out.sort_values("samples", ascending=False).reset_index(drop=True)
        return out

    full_probs = _aggregate("state_full")
    coarse_probs = _aggregate("state_coarse")

    base_spx = float((valid["fwd_spx"] > 0).mean() * 100.0)
    base_ndx = float((valid["fwd_ndx"] > 0).mean() * 100.0)

    current_sent = _as_float(payload.get("sentiment", {}).get("composite_score"))
    current_vol = _as_float(payload.get("volume_profile", {}).get("ratio"))
    current_vix = _as_float(payload.get("snapshot", {}).get("^VIX", {}).get("last"))
    current_breadth = _as_float(payload.get("sentiment", {}).get("breadth_ratio"))
    current_full = (
        f"{_bucket_sentiment(current_sent)} | {_bucket_volume(current_vol)} | {_bucket_vix(current_vix)} | {_bucket_breadth(current_breadth)}"
    )
    current_coarse = f"{_bucket_sentiment(current_sent)} | {_bucket_volume(current_vol)}"

    current = pd.DataFrame()
    conditioning = "full"
    if not full_probs.empty:
        current = full_probs[full_probs["state"] == current_full].copy()
    if current.empty or int(current["samples"].iloc[0]) < min_samples:
        conditioning = "coarse"
        if not coarse_probs.empty:
            current = coarse_probs[coarse_probs["state"] == current_coarse].copy()

    if current.empty:
        return {
            "available": False,
            "reason": "Current regime state has too few historical matches so far.",
            "baseline_spx": base_spx,
            "baseline_ndx": base_ndx,
        }

    current_row = current.iloc[0].to_dict()
    edge = coarse_probs.copy() if not coarse_probs.empty else full_probs.copy()
    if not edge.empty:
        edge["spx_edge"] = edge["spx_up_prob"] - base_spx
        edge["ndx_edge"] = edge["ndx_up_prob"] - base_ndx
        edge["edge_abs"] = edge[["spx_edge", "ndx_edge"]].abs().max(axis=1)
        edge = edge[edge["samples"] >= min_samples].sort_values("edge_abs", ascending=False).head(10).reset_index(drop=True)

    return {
        "available": True,
        "conditioning": conditioning,
        "horizon_bars": int(horizon_bars),
        "current_state": current_row.get("state", "n/a"),
        "current": current_row,
        "baseline_spx": base_spx,
        "baseline_ndx": base_ndx,
        "samples_total": int(len(valid)),
        "edge_table": edge,
    }


def _regime_probability_chart(prob: dict) -> go.Figure:
    if not isinstance(prob, dict) or not prob.get("available", False):
        return go.Figure()

    current = prob.get("current", {})
    base_spx = _as_float(prob.get("baseline_spx"))
    base_ndx = _as_float(prob.get("baseline_ndx"))
    cur_spx = _as_float(current.get("spx_up_prob"))
    cur_ndx = _as_float(current.get("ndx_up_prob"))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["S&P 500", "Nasdaq"],
            y=[base_spx, base_ndx],
            name="Baseline",
            marker=dict(color="rgba(125, 179, 255, 0.65)"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=["S&P 500", "Nasdaq"],
            y=[cur_spx, cur_ndx],
            name="Current Regime Match",
            marker=dict(color="rgba(86, 211, 146, 0.85)"),
        )
    )
    fig.update_layout(
        title="Probability of Higher Print Over Forward Horizon",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=35, b=10),
        yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(140,170,205,0.18)"),
        xaxis=dict(title=""),
        font=dict(color="#ffffff", family="Space Grotesk"),
        height=300,
    )
    return _apply_chart_theme(fig)


def _render_actionability_panel(payload: dict, history_df: pd.DataFrame, history_prob_df: pd.DataFrame) -> None:
    st.subheader("Actionability Lab")
    st.caption("Structured trade planning from current regime, signal shifts, and empirical state probabilities.")

    tab1, tab2, tab3 = st.tabs(["Playbook", "Change Detection", "Regime Probabilities"])

    with tab1:
        table = _build_actionable_playbook(payload, history_df)
        if table.empty:
            st.info("Playbook unavailable.")
        else:
            top = table.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Top Setup", str(top.get("playbook", "n/a")))
            with c2:
                st.metric("Bias", str(top.get("bias", "n/a")))
            with c3:
                st.metric("Score", f"{float(top.get('score', np.nan)):.0f}/100")
            with c4:
                st.metric("Tier", str(top.get("tier", "n/a")))
            with c5:
                st.metric("Last Trigger", str(top.get("last_trigger_date", "n/a")))

            show = table.copy()
            show["score"] = show["score"].map(lambda x: f"{x:.0f}")
            st.dataframe(
                show[["playbook", "bias", "score", "tier", "last_trigger_date", "why_now", "trigger", "invalidation"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Trigger dates show the most recent model trigger match. For trend setups, date reflects the last snapshot matching regime trigger conditions."
            )

    with tab2:
        shift = _build_signal_change_snapshot(history_df)
        pulse = shift.get("pulse", "Unavailable")
        pulse_score = _as_float(shift.get("pulse_score"))
        shift_count = int(shift.get("shift_count", 0))
        current_time = shift.get("current_time")
        time_text = (
            pd.to_datetime(current_time, errors="coerce").strftime("%Y-%m-%d %I:%M %p")
            if current_time is not None and pd.notna(pd.to_datetime(current_time, errors="coerce"))
            else "n/a"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Pulse State", str(pulse))
        with c2:
            st.metric("Pulse Score", f"{pulse_score:+.2f}" if pd.notna(pulse_score) else "n/a")
        with c3:
            st.metric("Active Shifts", str(shift_count), delta=f"as of {time_text}")

        events = shift.get("events", pd.DataFrame())
        if isinstance(events, pd.DataFrame) and not events.empty:
            flagged = events[events["is_shift"]].copy()
            if flagged.empty:
                st.info("No metrics crossed shift thresholds on the latest snapshot.")
            else:
                st.plotly_chart(_signal_shift_chart(flagged), width="stretch")
                show = flagged.copy()
                show["z_score"] = show["z_score"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "n/a")
                st.dataframe(
                    show[["metric", "delta_text", "z_score", "state"]].head(12),
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("Change-detection history unavailable.")

    with tab3:
        prob = _build_regime_conditioned_probabilities(payload, history_prob_df)
        if not bool(prob.get("available", False)):
            st.info(str(prob.get("reason", "Regime probabilities unavailable.")))
        else:
            current = prob.get("current", {})
            conditioning = str(prob.get("conditioning", "n/a"))
            horizon = int(prob.get("horizon_bars", 0))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Conditioning", conditioning.title())
            with c2:
                st.metric("Samples", str(int(current.get("samples", 0))))
            with c3:
                st.metric("S&P Up Prob", f"{float(current.get('spx_up_prob', np.nan)):.1f}%")
            with c4:
                st.metric("Nasdaq Up Prob", f"{float(current.get('ndx_up_prob', np.nan)):.1f}%")

            st.plotly_chart(_regime_probability_chart(prob), width="stretch")
            st.caption(
                f"Forward horizon uses {horizon} snapshots. Current state: {prob.get('current_state', 'n/a')}"
            )

            edge = prob.get("edge_table", pd.DataFrame())
            if isinstance(edge, pd.DataFrame) and not edge.empty:
                show = edge.copy()
                for col in ["spx_up_prob", "ndx_up_prob", "spx_edge", "ndx_edge"]:
                    if col in show.columns:
                        show[col] = show[col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "n/a")
                for col in ["spx_median_move", "ndx_median_move"]:
                    if col in show.columns:
                        show[col] = show[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "n/a")
                st.markdown("**Highest-Edge Historical States**")
                st.dataframe(
                    show[
                        [
                            "state",
                            "samples",
                            "spx_up_prob",
                            "ndx_up_prob",
                            "spx_edge",
                            "ndx_edge",
                            "spx_median_move",
                            "ndx_median_move",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )


def _render_history_panel(history_df: pd.DataFrame) -> None:
    st.subheader("Historical Signal Trace (Last 72h)")
    if history_df.empty:
        st.info("History will populate after a few refresh cycles.")
        return

    st.plotly_chart(_history_chart(history_df), width='stretch')
    cols = st.columns(4)
    with cols[0]:
        avg_vol = history_df["volume_ratio"].dropna().mean() if "volume_ratio" in history_df else np.nan
        st.metric("Avg Volume Ratio", f"{avg_vol:.2f}x" if pd.notna(avg_vol) else "n/a")
    with cols[1]:
        avg_sent = history_df["sentiment_score"].dropna().mean() if "sentiment_score" in history_df else np.nan
        st.metric("Avg Sentiment", f"{avg_sent:+.1f}" if pd.notna(avg_sent) else "n/a")
    with cols[2]:
        max_vix = history_df["vix"].dropna().max() if "vix" in history_df else np.nan
        st.metric("Max VIX", f"{max_vix:.2f}" if pd.notna(max_vix) else "n/a")
    with cols[3]:
        last_ts = history_df["timestamp_et"].iloc[-1]
        st.metric("Snapshots", f"{len(history_df)}", delta=last_ts.strftime("%m-%d %I:%M %p"))


def _as_float(value: object) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _section_visible(selected_section: str, section_key: str) -> bool:
    return selected_section == "__all__" or selected_section == section_key


def _render_sidebar_navigation() -> str:
    if "selected_section" not in st.session_state:
        st.session_state["selected_section"] = "__all__"

    st.sidebar.markdown("### Navigation")

    quick_targets = [
        ("Overview", "overview"),
        ("Actionability", "actionability"),
        ("Near Lows", "near_52w_low"),
        ("Liquidity", "liquidity"),
        ("Yield Curve", "yield_curve"),
        ("Headlines", "headlines"),
    ]
    q1, q2 = st.sidebar.columns(2)
    for idx, (label, key) in enumerate(quick_targets):
        col = q1 if idx % 2 == 0 else q2
        with col:
            if st.button(label, key=f"quick_nav_{key}", width="stretch"):
                st.session_state["selected_section"] = key

    if st.sidebar.button("Full Dashboard", key="quick_nav_all", width="stretch"):
        st.session_state["selected_section"] = "__all__"

    options = ["Full Dashboard"] + [label for _, label in SECTION_NAV_ITEMS]
    selected_key = st.session_state.get("selected_section", "__all__")
    selected_label = "Full Dashboard" if selected_key == "__all__" else SECTION_LABEL_BY_KEY.get(selected_key, "Full Dashboard")
    default_index = options.index(selected_label) if selected_label in options else 0

    choice = st.sidebar.selectbox("View section", options, index=default_index)
    st.session_state["selected_section"] = "__all__" if choice == "Full Dashboard" else SECTION_KEY_BY_LABEL.get(choice, "__all__")

    return st.session_state["selected_section"]


def _render_quick_status_bar(payload: dict, routing_status: dict) -> None:
    sentiment = payload.get("sentiment", {})
    volume_profile = payload.get("volume_profile", {})
    liquidity = payload.get("liquidity_monitor", {})
    yield_curve = payload.get("yield_curve", {})

    sent_score = _as_float(sentiment.get("composite_score"))
    vol_ratio = _as_float(volume_profile.get("ratio"))
    liq_score = _as_float(liquidity.get("liquidity_score"))

    alerts = payload.get("alerts", [])
    alert_count = len(alerts) if isinstance(alerts, list) else 0
    routed_count = int(routing_status.get("sent_count", 0)) if isinstance(routing_status, dict) else 0
    error_count = int(routing_status.get("error_count", 0)) if isinstance(routing_status, dict) else 0

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("#### Dashboard Pulse")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Sentiment",
            str(sentiment.get("regime", "n/a")),
            delta=f"{sent_score:+.1f}" if pd.notna(sent_score) else "n/a",
        )
    with c2:
        st.metric(
            "Volume",
            str(volume_profile.get("regime", "n/a")),
            delta=f"{vol_ratio:.2f}x" if pd.notna(vol_ratio) else "n/a",
        )
    with c3:
        st.metric(
            "Liquidity",
            str(liquidity.get("liquidity_regime", "n/a")),
            delta=f"Stress {liq_score:.0f}" if pd.notna(liq_score) else "n/a",
        )
    with c4:
        st.metric(
            "Yield Curve",
            str(yield_curve.get("curve_state", "n/a")),
            delta=str(yield_curve.get("week_trend", "n/a")),
        )
    with c5:
        st.metric(
            "Alerts",
            str(alert_count),
            delta=f"Routed {routed_count} | Err {error_count}",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _inject_css()

    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_seconds = st.sidebar.slider("Refresh interval (sec)", min_value=30, max_value=300, value=60, step=15)
    manual_refresh = st.sidebar.button("Refresh now", width="stretch")
    selected_section = _render_sidebar_navigation()

    if "near_low_requested" not in st.session_state:
        st.session_state["near_low_requested"] = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Near-Low Screener")
    load_near_low = st.sidebar.button("Load 52W Low Screener", width="stretch")
    refresh_near_low = st.sidebar.button(
        "Refresh 52W Low Screener",
        width="stretch",
        disabled=not bool(st.session_state.get("near_low_requested", False)),
    )
    unload_near_low = st.sidebar.button(
        "Unload 52W Low Screener",
        width="stretch",
        disabled=not bool(st.session_state.get("near_low_requested", False)),
    )

    if load_near_low:
        st.session_state["near_low_requested"] = True
    if refresh_near_low:
        _get_near_52w_low_screener_payload.clear()
        st.session_state["near_low_requested"] = True
    if unload_near_low:
        st.session_state["near_low_requested"] = False

    st.sidebar.markdown(
        f"<div class='mono'>52W Screener: {'Loaded' if st.session_state.get('near_low_requested', False) else 'Not loaded'}</div>",
        unsafe_allow_html=True,
    )

    if manual_refresh:
        st.cache_data.clear()

    payload = _get_payload()

    try:
        persist_payload(payload)
    except Exception as exc:
        st.sidebar.warning(f"History write skipped: {exc}")

    try:
        routing_status = route_alerts(payload)
    except Exception as exc:
        routing_status = {
            "enabled": False,
            "destinations": [],
            "sent_count": 0,
            "skipped_count": 0,
            "error_count": 1,
            "errors": [str(exc)],
            "min_severity": "n/a",
            "cooldown_min": "n/a",
        }
        st.sidebar.warning(f"Alert routing skipped: {exc}")

    history_df = load_snapshot_history(hours=72)
    history_prob_df = load_snapshot_history(hours=24 * 120)
    alert_history = load_alert_history(limit=80)

    updated_at = payload.get("updated_at")
    updated_text = updated_at.strftime("%Y-%m-%d %I:%M:%S %p ET") if isinstance(updated_at, datetime) else "n/a"
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Snapshot")
    st.sidebar.markdown(f"<div class='mono'>Last refresh: {updated_text}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='mono'>Headlines: {len(payload.get('headlines', []))}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='mono'>Active alerts: {len(payload.get('alerts', []))}</div>", unsafe_allow_html=True)

    _render_header(payload["updated_at"])
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    _render_market_metrics(payload["snapshot"])
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    _render_quick_status_bar(payload, routing_status)

    if selected_section != "__all__":
        section_label = html.escape(SECTION_LABEL_BY_KEY.get(selected_section, selected_section))
        st.markdown(
            f"<div class='focus-banner'><strong>Focus Mode:</strong> {section_label}. Use sidebar navigation to jump or return to Full Dashboard.</div>",
            unsafe_allow_html=True,
        )

    if _section_visible(selected_section, "overview"):
        left, right = st.columns([1.6, 1.0])
        with left:
            _render_narrative(payload.get("narrative", []))
        with right:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            _render_sentiment_panel(payload.get("sentiment", {}))
            st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "actionability"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_actionability_panel(payload, history_df, history_prob_df)
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "alerts"):
        _render_alerts(payload.get("alerts", []), alert_history)

    if _section_visible(selected_section, "regime_cross"):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            _render_regime_panel(payload.get("regime_engine", {}))
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            _render_cross_asset_panel(payload.get("cross_asset_confirmation", {}))
            st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "event_watch"):
        c3, c4 = st.columns([1, 1])
        with c3:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            _render_event_risk_panel(payload.get("event_risk", {}))
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            _render_watchlist_panel(payload.get("watchlist", {}))
            st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "near_52w_low"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        if not bool(st.session_state.get("near_low_requested", False)):
            st.info("Press 'Load 52W Low Screener' in the sidebar to fetch this dataset on demand.")
        else:
            with st.spinner("Loading near-52-week-low screener..."):
                near_low_payload = _get_near_52w_low_screener_payload()
            _render_near_52w_low_panel(near_low_payload)
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "liquidity"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_liquidity_real_rate_panel(payload.get("liquidity_monitor", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "vol_structure"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_volatility_structure_panel(payload.get("volatility_structure", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "put_skew"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_put_skew_panel(payload.get("put_skew", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "capitulation"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_capitulation_panel(payload.get("capitulation", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "sector_capitulation"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_sector_capitulation_panel(payload.get("sector_capitulation", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "signal_quality"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_signal_quality_panel(payload.get("signal_quality", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "volume"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_volume_panel(payload.get("volume_profile", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "rotation"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_rotation_panel(payload.get("rotation_signal", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "sector_flow"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_sector_flow_panel(payload.get("sector_flow", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "rotation_flow_map"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_rotation_flow_map_panel(payload.get("rotation_flow_map", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "factor_baskets"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_factor_basket_panel(payload.get("factor_baskets", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "yield_curve"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_yield_curve_panel(payload.get("yield_curve", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "sectors"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_sector_panels(payload.get("sector_returns", pd.DataFrame()))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "cta"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_cta_panel(payload.get("cta_proxy", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "alert_routing"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_alert_routing_panel(routing_status)
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "history"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_history_panel(history_df)
        st.markdown("</div>", unsafe_allow_html=True)

    if _section_visible(selected_section, "headlines"):
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_headlines(payload.get("headlines", []))
        st.markdown("</div>", unsafe_allow_html=True)

    if auto_refresh:
        st.sidebar.markdown(f"<div class='mono'>Next refresh in {refresh_seconds}s</div>", unsafe_allow_html=True)
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()


