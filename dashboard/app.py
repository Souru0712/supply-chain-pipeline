"""
Supply Chain Intelligence Dashboard — 7 analytical tabs.

Tabs
----
  1. 📈  13-Week Forecast        — LightGBM price fan chart + detail table
  2. ⚠️  Composite Risk (simple) — volatility / uncertainty / momentum gauge
  3. 🔮  Enhanced Forecast+SHAP  — directional accuracy + SHAP feature attribution
  4. 💰  Margin & Cost Model      — total landed cost stack + scenario waterfall
  5. 🚨  Disruption Detection     — 4-component early-warning score + 12w trend
  6. 📅  Procurement Calendar     — LP-optimised 52-week purchase schedule
  7. 🎯  Risk Dashboard           — 4-dimension radar + composite gauge

Data sources (in priority order per tab):
  data/forecasts/CORN/weekly_forecast.csv
  data/forecasts/CORN/shap_summary.parquet
  data/forecasts/CORN/margin_model.parquet + scenario_analysis.json
  data/forecasts/CORN/disruption_score.json + disruption_history.parquet
  data/forecasts/CORN/procurement_schedule.parquet + seasonal_curve.json
  data/forecasts/CORN/risk_score.json + risk_history.parquet
  data/materialized/CORN/training_weekly.parquet

Run locally:
    python dashboard/app.py

Deployed via gunicorn:
    gunicorn 'dashboard.app:server'
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
FC_DIR = REPO_ROOT / "data" / "forecasts" / "CORN"
MAT_DIR = REPO_ROOT / "data" / "materialized" / "CORN"


# ---------------------------------------------------------------------------
# Data loaders (all return None gracefully when files missing)
# ---------------------------------------------------------------------------


def _load(path: Path, reader="parquet"):
    if not path.exists():
        return None
    try:
        if reader == "parquet":
            return pd.read_parquet(path)
        if reader == "csv":
            return pd.read_csv(path)
        if reader == "json":
            with open(path) as fh:
                return json.load(fh)
    except Exception:
        return None


def _load_historical() -> pd.DataFrame | None:
    df = _load(MAT_DIR / "training_weekly.parquet")
    if df is None:
        return None
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start").tail(104)


def _load_forecast() -> pd.DataFrame | None:
    df = _load(FC_DIR / "weekly_forecast.csv", "csv")
    if df is None:
        return None
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["target_week_start"] = pd.to_datetime(df["target_week_start"])
    return df


def _load_metadata() -> dict | None:
    path = REPO_ROOT / "models" / "corn" / "metadata.json"
    return _load(path, "json")


def _synthetic_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=104, freq="W-MON")
    prices = 5.5 + 0.5 * np.sin(2 * np.pi * np.arange(104) / 52)
    prices += np.cumsum(rng.normal(0, 0.04, 104))
    prices = np.clip(prices, 4.0, 8.0)
    hist = pd.DataFrame({"week_start": dates, "price_avg_weekly": prices})
    last_date, last_price = dates[-1], float(prices[-1])
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
# Shared styling
# ---------------------------------------------------------------------------

CARD = {
    "background": "white",
    "borderRadius": "8px",
    "padding": "24px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "marginBottom": "20px",
}
TAB_STYLE = {"padding": "12px 20px", "fontWeight": "600"}
TAB_SELECTED = {
    "padding": "12px 20px",
    "fontWeight": "700",
    "borderTop": "3px solid #2980b9",
}
PAD = {"padding": "20px 32px"}

color_for = lambda v: "#27ae60" if v < 33 else ("#e67e22" if v < 66 else "#e74c3c")  # noqa


# ---------------------------------------------------------------------------
# Tab 1 & 2 helpers (existing logic)
# ---------------------------------------------------------------------------


def compute_risk(hist: pd.DataFrame, fc: pd.DataFrame) -> dict:
    prices = hist["price_avg_weekly"].values[-12:]
    vol = float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0
    vol_score = int(min(100, vol * 600))
    last = fc.iloc[-1]
    band_pct = (last["p90"] - last["p10"]) / last["p50"] if last["p50"] > 0 else 0.0
    unc_score = int(min(100, band_pct * 180))
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


def build_forecast_fig(hist: pd.DataFrame, fc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["week_start"],
            y=hist["price_avg_weekly"],
            name="Historical",
            line={"color": "#2980b9", "width": 2.5},
            mode="lines",
        )
    )
    x_band = pd.concat([fc["target_week_start"], fc["target_week_start"].iloc[::-1]])
    y_band = pd.concat([fc["p90"], fc["p10"].iloc[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor="rgba(231,76,60,0.15)",
            line={"color": "rgba(0,0,0,0)"},
            name="80% CI",
            hoverinfo="skip",
        )
    )
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
    for col, label in [("p10", "p10"), ("p90", "p90")]:
        fig.add_trace(
            go.Scatter(
                x=fc["target_week_start"],
                y=fc[col],
                name=label,
                line={"color": "#e74c3c", "width": 1, "dash": "dot"},
                mode="lines",
                opacity=0.5,
            )
        )
    fd = str(pd.Timestamp(fc["forecast_date"].iloc[0]).date())
    fig.add_shape(
        type="line",
        x0=fd,
        x1=fd,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dot", "color": "gray", "width": 1.5},
    )
    fig.update_layout(
        title="13-Week Corn Price Forecast (LightGBM)",
        xaxis_title="Date",
        yaxis_title="Price ($/bushel)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
        margin={"t": 80},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig


def build_gauge_fig(value: float, title: str, level: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={
                "text": f"{title}<br><span style='font-size:18px;color:{color}'>{level}</span>"
            },
            number={"suffix": " / 100", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 33], "color": "#d5f5e3"},
                    {"range": [33, 66], "color": "#fef9e7"},
                    {"range": [66, 100], "color": "#fadbd8"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin={"t": 60, "b": 10}, template="plotly_white")
    return fig


def _bar_row(label: str, value: int) -> html.Div:
    return html.Div(
        [
            html.Div(
                label, style={"width": "180px", "fontWeight": "600", "fontSize": "14px"}
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
                            "width": f"{value}%",
                            "background": color_for(value),
                            "height": "100%",
                            "borderRadius": "4px",
                        }
                    )
                ],
            ),
            html.Div(
                f"{value}/100",
                style={"width": "60px", "textAlign": "right", "fontSize": "14px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "14px"},
    )


def build_forecast_table(fc: pd.DataFrame) -> html.Table:
    header = html.Tr(
        [
            html.Th(h, style={"padding": "10px 14px"})
            for h in [
                "Horizon",
                "Week of",
                "p10 ($/bu)",
                "p50 ($/bu)",
                "p90 ($/bu)",
                "Spread",
            ]
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
# Tab 3 — Enhanced Forecast + SHAP
# ---------------------------------------------------------------------------


def build_shap_tab(hist: pd.DataFrame, fc: pd.DataFrame) -> html.Div:
    shap_df = _load(FC_DIR / "shap_summary.parquet")

    children = [html.Div(dcc.Graph(figure=build_forecast_fig(hist, fc)), style=CARD)]

    if shap_df is not None and "mean_importance" in shap_df.columns:
        top = shap_df["mean_importance"].dropna().head(12)
        fig = go.Figure(
            go.Bar(
                x=top.values[::-1],
                y=top.index[::-1],
                orientation="h",
                marker_color="#2980b9",
                text=[f"{v:.5f}" for v in top.values[::-1]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Top Feature Importances — Mean |SHAP| (median model, all horizons)",
            xaxis_title="Mean |SHAP value|",
            template="plotly_white",
            height=420,
            margin={"t": 60, "l": 200},
        )
        children.append(html.Div(dcc.Graph(figure=fig), style=CARD))

    # Directional accuracy note from metadata_v2
    meta2 = _load(REPO_ROOT / "models" / "corn" / "metadata_v2.json", "json")
    if meta2:
        dir_rows = []
        for hr in meta2.get("per_horizon", []):
            q50 = (hr.get("quantiles") or {}).get("q50") or {}
            da = q50.get("directional_accuracy")
            if da is not None:
                dir_rows.append({"horizon": hr["horizon"], "directional_accuracy": da})
        if dir_rows:
            da_df = pd.DataFrame(dir_rows)
            fig2 = go.Figure(
                go.Bar(
                    x=da_df["horizon"],
                    y=(da_df["directional_accuracy"] * 100).round(1),
                    marker_color="#27ae60",
                    text=(da_df["directional_accuracy"] * 100).round(1),
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )
            )
            fig2.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                annotation_text="50% (random guess)",
            )
            fig2.update_layout(
                title="Directional Accuracy by Horizon (% correct up/down calls)",
                xaxis_title="Horizon (weeks)",
                yaxis_title="Directional Accuracy (%)",
                yaxis_range=[0, 105],
                template="plotly_white",
                height=360,
                margin={"t": 60},
            )
            children.append(html.Div(dcc.Graph(figure=fig2), style=CARD))

    return html.Div(children, style=PAD)


# ---------------------------------------------------------------------------
# Tab 4 — Margin & Cost Model
# ---------------------------------------------------------------------------


def build_margin_tab() -> html.Div:
    margin_df = _load(FC_DIR / "margin_model.parquet")
    scenarios = _load(FC_DIR / "scenario_analysis.json", "json")
    children = []

    if margin_df is not None:
        margin_df["week_start"] = pd.to_datetime(margin_df["week_start"])
        recent = margin_df.tail(52)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent["week_start"],
                y=recent["price_avg_weekly"],
                name="Commodity Price",
                stackgroup="cost",
                line={"color": "#2980b9"},
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=recent["week_start"],
                y=recent["transport_cost"],
                name="Transport",
                stackgroup="cost",
                mode="lines",
                line={"color": "#e67e22"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=recent["week_start"],
                y=recent["energy_cost"],
                name="Energy",
                stackgroup="cost",
                mode="lines",
                line={"color": "#8e44ad"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=recent["week_start"],
                y=recent["carry_cost"],
                name="Carry Cost",
                stackgroup="cost",
                mode="lines",
                line={"color": "#e74c3c"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=recent["week_start"],
                y=recent["implied_sell_price"],
                name="Implied Sell Price",
                mode="lines",
                line={"color": "#27ae60", "width": 2.5, "dash": "dash"},
            )
        )
        fig.update_layout(
            title="Total Landed Cost Stack — Last 52 Weeks",
            xaxis_title="Date",
            yaxis_title="USD / bushel",
            template="plotly_white",
            height=460,
            margin={"t": 60},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        children.append(html.Div(dcc.Graph(figure=fig), style=CARD))

        # Cost breakdown cards
        last = recent.iloc[-1]
        tlc = float(last["total_landed_cost"])
        cards = []
        for label, col, color in [
            ("Commodity", "price_avg_weekly", "#2980b9"),
            ("Transport", "transport_cost", "#e67e22"),
            ("Energy", "energy_cost", "#8e44ad"),
            ("Carry", "carry_cost", "#e74c3c"),
        ]:
            val = float(last[col])
            pct = val / tlc * 100 if tlc > 0 else 0
            cards.append(
                html.Div(
                    [
                        html.Div(
                            label,
                            style={
                                "fontSize": "12px",
                                "color": "#888",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            f"${val:.4f}",
                            style={
                                "fontSize": "22px",
                                "fontWeight": "700",
                                "color": color,
                            },
                        ),
                        html.Div(
                            f"{pct:.1f}% of TLC",
                            style={"fontSize": "12px", "color": "#aaa"},
                        ),
                    ],
                    style={
                        "background": "white",
                        "borderRadius": "8px",
                        "padding": "16px 20px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                        "flex": "1",
                        "borderTop": f"4px solid {color}",
                    },
                )
            )
        children.append(
            html.Div(
                cards, style={"display": "flex", "gap": "16px", "marginBottom": "20px"}
            )
        )

    if scenarios:
        # Waterfall chart: base TLC + shock impacts
        base = scenarios["base_tlc"]
        sc = scenarios["scenarios"]
        bar_labels, bar_vals, bar_colors = ["Base TLC"], [base], ["#2980b9"]
        for driver, shocks in sc.items():
            # Use +20% shock
            shock_20 = next((s for s in shocks if s["shock_pct"] == 20), None)
            if shock_20:
                bar_labels.append(f"{driver} +20%")
                bar_vals.append(base + shock_20["delta_tlc"])
                bar_colors.append("#e74c3c" if shock_20["delta_tlc"] > 0 else "#27ae60")

        fig2 = go.Figure(
            go.Bar(
                x=bar_labels,
                y=bar_vals,
                marker_color=bar_colors,
                text=[f"${v:.4f}" for v in bar_vals],
                textposition="outside",
            )
        )
        fig2.add_hline(
            y=base,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Base ${base:.4f}/bu",
        )
        fig2.update_layout(
            title="Scenario Analysis — Total Landed Cost under +20% Cost Shocks",
            yaxis_title="Total Landed Cost ($/bu)",
            template="plotly_white",
            height=380,
            margin={"t": 60},
        )
        children.append(html.Div(dcc.Graph(figure=fig2), style=CARD))

    if not children:
        children = [
            html.Div(
                "Run `python scripts/9.margin_model.py` to generate margin data.",
                style={**CARD, "color": "#888", "textAlign": "center"},
            )
        ]
    return html.Div(children, style=PAD)


# ---------------------------------------------------------------------------
# Tab 5 — Disruption Detection
# ---------------------------------------------------------------------------


def build_disruption_tab() -> html.Div:
    snapshot = _load(FC_DIR / "disruption_score.json", "json")
    hist_df = _load(FC_DIR / "disruption_history.parquet")
    children = []

    score = snapshot["composite_score"] if snapshot else 50.0
    alert = (
        snapshot["alert"]
        if snapshot
        else {"level": "UNKNOWN", "color": "#888", "action": "Run disruption script"}
    )

    # Gauge
    gauge = build_gauge_fig(score, "Disruption Score", alert["level"], alert["color"])
    children.append(
        html.Div(
            [
                html.Div(dcc.Graph(figure=gauge), style={**CARD, "flex": "1"}),
                html.Div(
                    [
                        html.H3(
                            "Alert", style={"marginTop": "0", "color": alert["color"]}
                        ),
                        html.P(
                            alert["action"],
                            style={"fontSize": "16px", "fontWeight": "600"},
                        ),
                        html.Hr(),
                        html.H4("Component Scores"),
                        *(
                            [
                                _bar_row(k.replace("_", " ").title(), int(v))
                                for k, v in snapshot["components"].items()
                            ]
                            if snapshot
                            else []
                        ),
                    ],
                    style={**CARD, "flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "20px"},
        )
    )

    if hist_df is not None and "disruption_score" in hist_df.columns:
        hist_df["week_start"] = pd.to_datetime(hist_df["week_start"])
        fig = go.Figure()
        for comp, color, dash in [
            ("score_supply_deviation", "#e67e22", "dot"),
            ("score_price_gap", "#8e44ad", "dot"),
            ("score_price_momentum", "#2980b9", "dot"),
            ("score_volatility_spike", "#e74c3c", "dot"),
        ]:
            if comp in hist_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df["week_start"],
                        y=hist_df[comp],
                        name=comp.replace("score_", "").replace("_", " ").title(),
                        line={"dash": dash, "width": 1},
                        opacity=0.7,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=hist_df["week_start"],
                y=hist_df["disruption_score"],
                name="Composite",
                line={"color": "black", "width": 2.5},
            )
        )
        for thresh, label, color in [
            (60, "Alert", "#fbc02d"),
            (75, "Review", "#f57c00"),
            (90, "Emergency", "#d32f2f"),
        ]:
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )
        fig.update_layout(
            title="Disruption Score History — All Components",
            xaxis_title="Date",
            yaxis_title="Score (0–100)",
            yaxis_range=[0, 105],
            template="plotly_white",
            height=440,
            margin={"t": 60},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        children.append(html.Div(dcc.Graph(figure=fig), style=CARD))

    return html.Div(children, style=PAD)


# ---------------------------------------------------------------------------
# Tab 6 — Procurement Calendar
# ---------------------------------------------------------------------------


def build_procurement_tab() -> html.Div:
    sched = _load(FC_DIR / "procurement_schedule.parquet")
    curve_meta = _load(FC_DIR / "seasonal_curve.json", "json")
    children = []

    if sched is not None:
        summary = curve_meta.get("optimiser_summary", {}) if curve_meta else {}
        if summary:
            stats = [
                ("Annual Volume", f"{summary.get('annual_volume', 0):,.0f} bu"),
                ("Optimised Cost", f"${summary.get('total_cost', 0):,.2f}"),
                ("Naive Cost", f"${summary.get('naive_cost', 0):,.2f}"),
                ("Est. Saving", f"{summary.get('saving_pct', 0):.2f}%"),
            ]
            cards = []
            for label, val in stats:
                color = "#27ae60" if "Saving" in label else "#2980b9"
                cards.append(
                    html.Div(
                        [
                            html.Div(
                                label, style={"fontSize": "12px", "color": "#888"}
                            ),
                            html.Div(
                                val,
                                style={
                                    "fontSize": "22px",
                                    "fontWeight": "700",
                                    "color": color,
                                },
                            ),
                        ],
                        style={
                            "background": "white",
                            "borderRadius": "8px",
                            "padding": "16px 20px",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                            "flex": "1",
                            "borderTop": f"4px solid {color}",
                        },
                    )
                )
            children.append(
                html.Div(
                    cards,
                    style={"display": "flex", "gap": "16px", "marginBottom": "20px"},
                )
            )

        # Price curve + purchase schedule
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sched["week_index"],
                y=sched["p75_price"],
                name="p75 band",
                line={"color": "rgba(231,76,60,0.3)"},
                fill=None,
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sched["week_index"],
                y=sched["p25_price"],
                name="p25 band",
                fill="tonexty",
                fillcolor="rgba(231,76,60,0.1)",
                line={"color": "rgba(231,76,60,0.3)"},
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sched["week_index"],
                y=sched["expected_price"],
                name="Expected Price",
                line={"color": "#e74c3c", "width": 2.5},
                mode="lines",
            )
        )

        # Opportunistic weeks
        opp = (
            sched[sched["opportunistic_flag"] == True]
            if "opportunistic_flag" in sched.columns
            else pd.DataFrame()
        )  # noqa
        if not opp.empty:
            fig.add_trace(
                go.Scatter(
                    x=opp["week_index"],
                    y=opp["expected_price"],
                    name="Opportunistic Buy",
                    mode="markers",
                    marker={"color": "#27ae60", "size": 12, "symbol": "star"},
                )
            )

        fig.update_layout(
            title="52-Week Forward Price Curve with Seasonal Bands",
            xaxis_title="Week Forward",
            yaxis_title="Expected Price ($/bu)",
            template="plotly_white",
            height=380,
            margin={"t": 60},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        children.append(html.Div(dcc.Graph(figure=fig), style=CARD))

        # Purchase schedule bar chart
        annual_vol = sched["recommended_purchase"].sum()
        fig2 = go.Figure()
        bar_colors = [
            "#27ae60" if row.get("opportunistic_flag", False) else "#2980b9"
            for _, row in sched.iterrows()
        ]
        fig2.add_trace(
            go.Bar(
                x=sched["week_index"],
                y=sched["recommended_purchase"] / annual_vol * 100,
                name="Purchase %",
                marker_color=bar_colors,
            )
        )
        fig2.update_layout(
            title="Recommended Weekly Purchase Allocation (% of Annual Volume)<br>"
            "<sup>Green = opportunistic buy window</sup>",
            xaxis_title="Week Forward",
            yaxis_title="% of Annual Volume",
            template="plotly_white",
            height=320,
            margin={"t": 80},
        )
        children.append(html.Div(dcc.Graph(figure=fig2), style=CARD))

    if not children:
        children = [
            html.Div(
                "Run `python scripts/11.procurement_optimizer.py` to generate schedule.",
                style={**CARD, "color": "#888", "textAlign": "center"},
            )
        ]
    return html.Div(children, style=PAD)


# ---------------------------------------------------------------------------
# Tab 7 — Risk Dashboard
# ---------------------------------------------------------------------------


def build_risk_tab() -> html.Div:
    snapshot = _load(FC_DIR / "risk_score.json", "json")
    hist_df = _load(FC_DIR / "risk_history.parquet")
    children = []

    score = snapshot["composite_risk"] if snapshot else 50.0
    cls = (
        snapshot["classification"]
        if snapshot
        else {"level": "UNKNOWN", "color": "#888", "action": "Run risk script"}
    )

    # Gauge + radar side by side
    gauge = build_gauge_fig(score, "Composite Risk Score", cls["level"], cls["color"])

    radar_fig = go.Figure()
    if snapshot and "radar" in snapshot:
        cats = [r["dimension"].replace("_", " ").title() for r in snapshot["radar"]]
        vals = [r["score"] for r in snapshot["radar"]]
        cats_closed = cats + [cats[0]]
        vals_closed = vals + [vals[0]]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                name="Risk Score",
                line={"color": cls["color"]},
            )
        )
        radar_fig.update_layout(
            polar={"radialaxis": {"range": [0, 100]}},
            title="Risk Dimensions",
            template="plotly_white",
            height=320,
            margin={"t": 60},
        )

    dim_info = []
    if snapshot and "dimensions" in snapshot:
        weights = snapshot.get("weights", {})
        for dim, val in snapshot["dimensions"].items():
            w = weights.get(dim, 0.25)
            dim_info.append(
                _bar_row(f"{dim.replace('_', ' ').title()} ({w:.0%})", int(val))
            )

    children.append(
        html.Div(
            [
                html.Div(dcc.Graph(figure=gauge), style={**CARD, "flex": "1"}),
                html.Div(dcc.Graph(figure=radar_fig), style={**CARD, "flex": "1"}),
            ],
            style={"display": "flex", "gap": "20px"},
        )
    )

    if dim_info:
        children.append(
            html.Div(
                [
                    html.H3("Dimension Scores", style={"marginTop": "0"}),
                    html.P(
                        cls["action"],
                        style={"color": cls["color"], "fontWeight": "600"},
                    ),
                    *dim_info,
                ],
                style=CARD,
            )
        )

    if hist_df is not None and "composite_risk" in hist_df.columns:
        hist_df["week_start"] = pd.to_datetime(hist_df["week_start"])
        fig = go.Figure()
        dim_cols = ["supply_risk", "cost_risk", "logistics_risk", "demand_risk"]
        colors_map = {
            "supply_risk": "#e67e22",
            "cost_risk": "#e74c3c",
            "logistics_risk": "#8e44ad",
            "demand_risk": "#2980b9",
        }
        for col in dim_cols:
            if col in hist_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_df["week_start"],
                        y=hist_df[col],
                        name=col.replace("_", " ").title(),
                        line={"color": colors_map[col], "width": 1.5, "dash": "dot"},
                        opacity=0.7,
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=hist_df["week_start"],
                y=hist_df["composite_risk"],
                name="Composite Risk",
                line={"color": "black", "width": 2.5},
            )
        )
        for thresh, label, color in [
            (40, "Moderate", "#fbc02d"),
            (60, "High", "#f57c00"),
            (80, "Critical", "#d32f2f"),
        ]:
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )
        fig.update_layout(
            title="Risk Score History — All Dimensions",
            xaxis_title="Date",
            yaxis_title="Score (0–100)",
            yaxis_range=[0, 105],
            template="plotly_white",
            height=440,
            margin={"t": 60},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        children.append(html.Div(dcc.Graph(figure=fig), style=CARD))

    return html.Div(children, style=PAD)


# ---------------------------------------------------------------------------
# Load data and assemble layout
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
    " ⚠️ demo data — run pipeline for live results" if _load_historical() is None else ""
)

risk = compute_risk(hist_df, forecast_df)
forecast_fig = build_forecast_fig(hist_df, forecast_df)
gauge_fig = build_gauge_fig(
    risk["composite"], "Composite Risk Score", risk["level"], risk["color"]
)

app = Dash(__name__, title="Supply Chain Intelligence Dashboard")
server = app.server

app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, sans-serif",
        "background": "#f4f6f9",
        "minHeight": "100vh",
    },
    children=[
        # Header
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
                            "🌽 Supply Chain Intelligence Dashboard",
                            style={
                                "margin": "0",
                                "fontSize": "26px",
                                "fontWeight": "700",
                            },
                        ),
                        html.P(
                            f"CORN · LightGBM · Trained through {train_end}{is_demo}",
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
        # Tabs
        dcc.Tabs(
            style={"margin": "20px 32px 0"},
            colors={"border": "#ddd", "primary": "#2980b9", "background": "#f4f6f9"},
            children=[
                dcc.Tab(
                    label="📈  13-Week Forecast",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #2980b9"},
                    children=[
                        html.Div(
                            style=PAD,
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
                dcc.Tab(
                    label="⚠️  Risk Score",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #e74c3c"},
                    children=[
                        html.Div(
                            style=PAD,
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(figure=gauge_fig),
                                            style={**CARD, "flex": "1"},
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Risk Components",
                                                    style={"marginTop": "0"},
                                                ),
                                                _bar_row(
                                                    "Price Volatility",
                                                    risk["volatility"],
                                                ),
                                                _bar_row(
                                                    "Forecast Uncertainty",
                                                    risk["uncertainty"],
                                                ),
                                                _bar_row(
                                                    "Price Momentum", risk["momentum"]
                                                ),
                                            ],
                                            style={**CARD, "flex": "1"},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Methodology", style={"marginTop": "0"}
                                        ),
                                        html.P(
                                            "Composite = 40% × Volatility + 40% × Uncertainty + 20% × Momentum. "
                                            "Volatility: 12-week CoV of spot price. "
                                            "Uncertainty: relative band width (p90−p10)/p50 at h=13. "
                                            "Momentum: absolute 12-week price change rate.",
                                            style={
                                                "color": "#555",
                                                "lineHeight": "1.6",
                                            },
                                        ),
                                    ],
                                    style=CARD,
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="🔮  SHAP Attribution",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #8e44ad"},
                    children=[build_shap_tab(hist_df, forecast_df)],
                ),
                dcc.Tab(
                    label="💰  Margin & Cost",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #e67e22"},
                    children=[build_margin_tab()],
                ),
                dcc.Tab(
                    label="🚨  Disruption Alert",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #e74c3c"},
                    children=[build_disruption_tab()],
                ),
                dcc.Tab(
                    label="📅  Procurement",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #27ae60"},
                    children=[build_procurement_tab()],
                ),
                dcc.Tab(
                    label="🎯  Risk Dashboard",
                    style=TAB_STYLE,
                    selected_style={**TAB_SELECTED, "borderTop": "3px solid #c0392b"},
                    children=[build_risk_tab()],
                ),
            ],
        ),
        html.Div(
            "Supply Chain Analytics Pipeline · LightGBM · SHAP · Dash · Plotly",
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
