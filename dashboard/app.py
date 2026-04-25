"""
Supply Chain Price Dashboard — composite risk score + 13-week forecast view.

Data sources (in priority order):
  1. data/forecasts/CORN/weekly_forecast.csv   (from 7.inference.py)
  2. data/materialized/CORN/training_weekly.parquet
  3. Synthetic fallback so the dashboard always renders for demo purposes

Run locally:
    python dashboard/app.py

Deployed on Render / Fly.io via gunicorn:
    gunicorn 'dashboard.app:server'

Environment variables:
    PORT        — listen port (default 8050)
    DASH_DEBUG  — set to "true" for hot-reload in dev
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_historical() -> pd.DataFrame | None:
    path = REPO_ROOT / "data" / "materialized" / "CORN" / "training_weekly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start").tail(104)  # last 2 years


def _load_forecast() -> pd.DataFrame | None:
    path = REPO_ROOT / "data" / "forecasts" / "CORN" / "weekly_forecast.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["forecast_date", "target_week_start"])


def _load_metadata() -> dict | None:
    path = REPO_ROOT / "models" / "corn" / "metadata.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _synthetic_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate demo data when the real pipeline hasn't been run yet."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=104, freq="W-MON")
    prices = 5.5 + 0.5 * np.sin(2 * np.pi * np.arange(104) / 52)
    prices += np.cumsum(rng.normal(0, 0.04, 104))
    prices = np.clip(prices, 4.0, 8.0)

    hist = pd.DataFrame({"week_start": dates, "price_avg_weekly": prices})

    last_date = dates[-1]
    last_price = float(prices[-1])
    horizons = list(range(1, 14))
    drift = np.cumsum(rng.normal(0.01, 0.025, 13))
    p50 = np.clip(last_price + drift, 3.0, 10.0)
    spread = np.linspace(0.12, 0.50, 13)

    fc = pd.DataFrame(
        {
            "forecast_date": last_date,
            "horizon_weeks": horizons,
            "target_week_start": [last_date + pd.Timedelta(weeks=h) for h in horizons],
            "p10": p50 - spread,
            "p50": p50,
            "p90": p50 + spread,
        }
    )
    return hist, fc


# ---------------------------------------------------------------------------
# Risk score computation
# ---------------------------------------------------------------------------


def compute_risk(hist: pd.DataFrame, fc: pd.DataFrame) -> dict:
    prices = hist["price_avg_weekly"].values[-12:]

    # Coefficient of variation → volatility score
    vol = float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0
    vol_score = int(min(100, vol * 600))

    # Forecast band width at h=13 relative to p50
    last = fc.iloc[-1]
    band_pct = (last["p90"] - last["p10"]) / last["p50"] if last["p50"] > 0 else 0.0
    unc_score = int(min(100, band_pct * 180))

    # Absolute price momentum over last 12 weeks
    mom = abs(float(prices[-1]) / float(prices[0]) - 1) if float(prices[0]) > 0 else 0.0
    mom_score = int(min(100, mom * 250))

    composite = int(0.40 * vol_score + 0.40 * unc_score + 0.20 * mom_score)
    level = "LOW" if composite < 33 else ("MODERATE" if composite < 66 else "HIGH")
    color = (
        "#27ae60" if composite < 33 else ("#e67e22" if composite < 66 else "#e74c3c")
    )

    return {
        "composite": composite,
        "level": level,
        "color": color,
        "volatility": vol_score,
        "uncertainty": unc_score,
        "momentum": mom_score,
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def build_forecast_fig(hist: pd.DataFrame, fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist["week_start"],
            y=hist["price_avg_weekly"],
            name="Historical",
            line={"color": "#2980b9", "width": 2.5},
            mode="lines",
        )
    )

    # Confidence band (p10–p90)
    x_band = pd.concat([fc["target_week_start"], fc["target_week_start"].iloc[::-1]])
    y_band = pd.concat([fc["p90"], fc["p10"].iloc[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor="rgba(231,76,60,0.15)",
            line={"color": "rgba(0,0,0,0)"},
            name="80 % CI (p10–p90)",
            hoverinfo="skip",
        )
    )

    # Forecast median
    fig.add_trace(
        go.Scatter(
            x=fc["target_week_start"],
            y=fc["p50"],
            name="Forecast p50",
            line={"color": "#e74c3c", "width": 2.5, "dash": "dash"},
            mode="lines+markers",
            marker={"size": 5},
        )
    )

    # p10 / p90 boundaries
    for col, label, dash in [("p10", "p10", "dot"), ("p90", "p90", "dot")]:
        fig.add_trace(
            go.Scatter(
                x=fc["target_week_start"],
                y=fc[col],
                name=label,
                line={"color": "#e74c3c", "width": 1, "dash": dash},
                mode="lines",
                opacity=0.5,
            )
        )

    # "Now" marker — use add_shape to avoid plotly version edge cases with timestamps
    forecast_date_str = str(pd.Timestamp(fc["forecast_date"].iloc[0]).date())
    fig.add_shape(
        type="line",
        x0=forecast_date_str,
        x1=forecast_date_str,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dot", "color": "gray", "width": 1.5},
    )
    fig.add_annotation(
        x=forecast_date_str,
        y=0.97,
        yref="paper",
        text="Forecast origin",
        showarrow=False,
        font={"color": "gray", "size": 11},
        xanchor="left",
    )

    fig.update_layout(
        title="13-Week Corn Price Forecast (XGBoost)",
        xaxis_title="Date",
        yaxis_title="Price ($/bushel)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        template="plotly_white",
        hovermode="x unified",
        height=480,
        margin={"t": 80},
    )
    return fig


def build_gauge_fig(risk: dict) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk["composite"],
            title={
                "text": (
                    f"Composite Risk Score<br>"
                    f"<span style='font-size:20px;color:{risk['color']}'>"
                    f"{risk['level']}</span>"
                )
            },
            number={"suffix": " / 100", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": risk["color"], "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "lightgray",
                "steps": [
                    {"range": [0, 33], "color": "#d5f5e3"},
                    {"range": [33, 66], "color": "#fef9e7"},
                    {"range": [66, 100], "color": "#fadbd8"},
                ],
            },
        )
    )
    fig.update_layout(height=340, margin={"t": 60, "b": 10}, template="plotly_white")
    return fig


def build_risk_breakdown(risk: dict) -> html.Div:
    components = [
        ("Price Volatility", risk["volatility"]),
        ("Forecast Uncertainty", risk["uncertainty"]),
        ("Price Momentum", risk["momentum"]),
    ]
    color_for = lambda v: "#27ae60" if v < 33 else ("#e67e22" if v < 66 else "#e74c3c")  # noqa: E731

    bars = []
    for label, val in components:
        bars.append(
            html.Div(
                [
                    html.Div(
                        label,
                        style={
                            "width": "180px",
                            "fontWeight": "600",
                            "fontSize": "14px",
                        },
                    ),
                    html.Div(
                        style={
                            "flex": "1",
                            "background": "#ecf0f1",
                            "borderRadius": "4px",
                            "height": "22px",
                            "margin": "0 12px",
                            "overflow": "hidden",
                        },
                        children=[
                            html.Div(
                                style={
                                    "width": f"{val}%",
                                    "background": color_for(val),
                                    "height": "100%",
                                    "borderRadius": "4px",
                                    "transition": "width 0.4s",
                                }
                            )
                        ],
                    ),
                    html.Div(
                        f"{val}/100",
                        style={
                            "width": "60px",
                            "textAlign": "right",
                            "fontSize": "14px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "14px",
                },
            )
        )

    return html.Div(
        [html.H3("Risk Component Breakdown", style={"marginTop": "0"}), *bars],
        style={
            "background": "white",
            "borderRadius": "8px",
            "padding": "24px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
            "marginTop": "16px",
        },
    )


def build_forecast_table(fc: pd.DataFrame) -> html.Table:
    header = html.Tr(
        [
            html.Th("Horizon", style={"padding": "10px 14px"}),
            html.Th("Week of", style={"padding": "10px 14px"}),
            html.Th("p10  ($/bu)", style={"padding": "10px 14px"}),
            html.Th("p50  ($/bu)", style={"padding": "10px 14px"}),
            html.Th("p90  ($/bu)", style={"padding": "10px 14px"}),
            html.Th("Spread", style={"padding": "10px 14px"}),
        ],
        style={"background": "#2c3e50", "color": "white"},
    )
    rows = []
    for _, row in fc.iterrows():
        spread = row["p90"] - row["p10"]
        rows.append(
            html.Tr(
                [
                    html.Td(
                        f"Week +{int(row['horizon_weeks'])}",
                        style={"padding": "8px 14px"},
                    ),
                    html.Td(
                        pd.Timestamp(row["target_week_start"]).strftime("%b %d, %Y"),
                        style={"padding": "8px 14px"},
                    ),
                    html.Td(
                        f"${row['p10']:.3f}",
                        style={"padding": "8px 14px", "color": "#2980b9"},
                    ),
                    html.Td(
                        f"${row['p50']:.3f}",
                        style={"padding": "8px 14px", "fontWeight": "700"},
                    ),
                    html.Td(
                        f"${row['p90']:.3f}",
                        style={"padding": "8px 14px", "color": "#e74c3c"},
                    ),
                    html.Td(
                        f"±${spread / 2:.3f}",
                        style={"padding": "8px 14px", "color": "#7f8c8d"},
                    ),
                ],
                style={"borderBottom": "1px solid #ecf0f1"},
            )
        )
    return html.Table(
        [header, html.Tbody(rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "background": "white",
            "borderRadius": "8px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
            "overflow": "hidden",
        },
    )


# ---------------------------------------------------------------------------
# Assemble app
# ---------------------------------------------------------------------------

hist_df = _load_historical()
forecast_df = _load_forecast()
if hist_df is None or forecast_df is None:
    hist_df, forecast_df = _synthetic_pair()

meta = _load_metadata()
train_end = (
    meta.get("train_end_date", "")[:10]
    if meta
    else str(hist_df["week_start"].max().date())
)
is_demo = (
    "⚠️ demo data — run pipeline for live results" if _load_historical() is None else ""
)

risk = compute_risk(hist_df, forecast_df)
forecast_fig = build_forecast_fig(hist_df, forecast_df)
gauge_fig = build_gauge_fig(risk)

CARD = {
    "background": "white",
    "borderRadius": "8px",
    "padding": "24px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "marginBottom": "20px",
}

app = Dash(__name__, title="Supply Chain Dashboard")
server = app.server  # Flask instance exposed for gunicorn

app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, sans-serif",
        "background": "#f4f6f9",
        "minHeight": "100vh",
    },
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": "linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%)",
                "padding": "24px 32px",
                "color": "white",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div(
                    [
                        html.H1(
                            "🌽 Supply Chain Price Dashboard",
                            style={
                                "margin": "0",
                                "fontSize": "26px",
                                "fontWeight": "700",
                            },
                        ),
                        html.P(
                            f"CORN · XGBoost 13-Week Forecast · Model trained through {train_end}"
                            + (f"  {is_demo}" if is_demo else ""),
                            style={
                                "margin": "6px 0 0 0",
                                "opacity": "0.85",
                                "fontSize": "14px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            f"{risk['composite']}",
                            style={
                                "fontSize": "42px",
                                "fontWeight": "900",
                                "color": risk["color"],
                                "lineHeight": "1",
                            },
                        ),
                        html.Div(
                            "Risk Score",
                            style={
                                "fontSize": "11px",
                                "opacity": "0.8",
                                "marginTop": "2px",
                            },
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
            ],
        ),
        # ── Tabs ────────────────────────────────────────────────────────────
        dcc.Tabs(
            style={"margin": "20px 32px 0"},
            colors={"border": "#ddd", "primary": "#2980b9", "background": "#f4f6f9"},
            children=[
                # Forecast tab
                dcc.Tab(
                    label="📈  13-Week Forecast",
                    style={"padding": "12px 20px", "fontWeight": "600"},
                    selected_style={
                        "padding": "12px 20px",
                        "fontWeight": "700",
                        "borderTop": "3px solid #2980b9",
                    },
                    children=[
                        html.Div(
                            style={"padding": "20px 32px"},
                            children=[
                                html.Div(dcc.Graph(figure=forecast_fig), style=CARD),
                                html.H3(
                                    "Weekly Forecast Detail",
                                    style={"margin": "0 0 12px"},
                                ),
                                build_forecast_table(forecast_df),
                            ],
                        )
                    ],
                ),
                # Risk tab
                dcc.Tab(
                    label="⚠️  Composite Risk Score",
                    style={"padding": "12px 20px", "fontWeight": "600"},
                    selected_style={
                        "padding": "12px 20px",
                        "fontWeight": "700",
                        "borderTop": "3px solid #e74c3c",
                    },
                    children=[
                        html.Div(
                            style={"padding": "20px 32px"},
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(figure=gauge_fig),
                                            style={**CARD, "flex": "1"},
                                        ),
                                        html.Div(
                                            build_risk_breakdown(risk),
                                            style={"flex": "1", "marginLeft": "20px"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "0"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "How the score is computed",
                                            style={"marginTop": "0"},
                                        ),
                                        html.P(
                                            "Composite = 40% × Volatility + 40% × Uncertainty + 20% × Momentum. "
                                            "Volatility is the 12-week coefficient of variation of the spot price. "
                                            "Uncertainty is the relative band width (p90−p10)/p50 at horizon 13. "
                                            "Momentum is the absolute 12-week price change rate.",
                                            style={
                                                "color": "#555",
                                                "lineHeight": "1.6",
                                            },
                                        ),
                                    ],
                                    style={**CARD, "marginTop": "20px"},
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        # ── Footer ──────────────────────────────────────────────────────────
        html.Div(
            "Supply Chain Analytics Pipeline · XGBoost · Dash · Plotly",
            style={
                "textAlign": "center",
                "padding": "24px",
                "color": "#aaa",
                "fontSize": "12px",
                "marginTop": "20px",
            },
        ),
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
