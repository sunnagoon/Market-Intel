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
from market_intel import build_dashboard_payload


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
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30, show_spinner=False)
def _get_payload() -> dict:
    return build_dashboard_payload()


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
            st.dataframe(
                comp[["ticker", "ratio", "observed_volume", "expected_volume"]],
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


def main() -> None:
    _inject_css()
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_seconds = st.sidebar.slider("Refresh interval (sec)", min_value=30, max_value=300, value=60, step=15)
    manual_refresh = st.sidebar.button("Refresh now")

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
    alert_history = load_alert_history(limit=80)

    _render_header(payload["updated_at"])
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    _render_market_metrics(payload["snapshot"])
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1.0])
    with left:
        _render_narrative(payload["narrative"])
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_sentiment_panel(payload["sentiment"])
        st.markdown("</div>", unsafe_allow_html=True)

    _render_alerts(payload.get("alerts", []), alert_history)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_regime_panel(payload.get("regime_engine", {}))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_cross_asset_panel(payload.get("cross_asset_confirmation", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_event_risk_panel(payload.get("event_risk", {}))
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        _render_watchlist_panel(payload.get("watchlist", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_liquidity_real_rate_panel(payload.get("liquidity_monitor", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_put_skew_panel(payload.get("put_skew", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_capitulation_panel(payload.get("capitulation", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_sector_capitulation_panel(payload.get("sector_capitulation", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_signal_quality_panel(payload.get("signal_quality", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_volume_panel(payload["volume_profile"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_rotation_panel(payload.get("rotation_signal", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_sector_flow_panel(payload.get("sector_flow", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_factor_basket_panel(payload.get("factor_baskets", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_yield_curve_panel(payload.get("yield_curve", {}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_sector_panels(payload["sector_returns"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_cta_panel(payload["cta_proxy"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_alert_routing_panel(routing_status)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_history_panel(history_df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    _render_headlines(payload["headlines"])
    st.markdown("</div>", unsafe_allow_html=True)

    if auto_refresh:
        st.sidebar.markdown(f"<div class='mono'>Next refresh in {refresh_seconds}s</div>", unsafe_allow_html=True)
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()


