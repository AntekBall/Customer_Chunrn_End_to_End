"""Portfolio Overview, Customer Scoring and Financial Simulator dashboard.

Start the API first, then run `uv run python -m src.frontend.app`.
CHURN_API_URL defaults to http://127.0.0.1:8000. This frontend performs no model
loading or direct dataset reads. Use separate processes for the API and Dash.
"""

import logging
import os
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from pydantic import ValidationError

from src.data.dashboard import PortfolioFilters
from src.data.portfolio import REVENUE_SEGMENTS, RISK_SEGMENTS
from src.frontend.categories import CATEGORY_OPTIONS
from src.frontend.financial import financial_layout, register_financial_callbacks
from src.frontend.client import APIError, PortfolioClient, Summary
from src.frontend.scoring import register_scoring_callbacks, scoring_layout


logger = logging.getLogger(__name__)


def _options(labels):
    return [{"label": label, "value": label} for label in labels]


def _dropdown(label, component_id, options):
    return html.Div([
        html.Label(label, htmlFor=component_id),
        dcc.Dropdown(
            id=component_id, options=options, placeholder="All", clearable=True,
        ),
    ], className="filter-field")


def _number(value: float | int | None, *, percent: bool = False) -> str:
    if value is None:
        return "Not defined"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def _kpis(summary: Summary):
    metrics = summary.overview
    cards = (
        ("CUSTOMERS IN VIEW", f"{metrics.customers:,}", "Snapshot members, not verified active users"),
        ("MONTHLY CHARGE TOTAL", _number(metrics.monthly_charge_total), "Charge proxy; not verified MRR"),
        ("MEAN PREDICTED CHURN", _number(metrics.mean_predicted_churn, percent=True),
         "Model estimate; no established forecast horizon"),
        ("RISK-WEIGHTED CHARGES", _number(metrics.risk_weighted_monthly_charges),
         "Sum of probability x charge; not realized loss"),
    )
    return [
        html.Article([
            html.P(label, className="kpi-label"),
            html.H2(value),
            html.P(description, className="muted"),
        ], className="kpi-card") for label, value, description in cards
    ]


def _heatmap(summary: Summary, metric: str) -> go.Figure:
    fields = {
        "customers": ("customers", "Customers"),
        "monthly_charge_total": ("monthly_charge_total", "Monthly charge total"),
        "mean_predicted_churn": ("mean_predicted_churn", "Mean predicted churn"),
        "risk_weighted_monthly_charges": (
            "risk_weighted_monthly_charges", "Risk-weighted monthly charges",
        ),
    }
    if metric not in fields:
        raise ValueError(f"Unknown heatmap metric: {metric}")
    field, title = fields[metric]
    cells = {(row.revenue_segment, row.risk_segment): row for row in summary.segments}
    z = [
        [getattr(cells[revenue, risk], field) for risk in RISK_SEGMENTS]
        for revenue in REVENUE_SEGMENTS
    ]
    text = [
        [
            f"{value:,}" if field == "customers" else _number(
                value, percent=field == "mean_predicted_churn",
            ) for value in row
        ] for row in z
    ]
    figure = go.Figure(go.Heatmap(
        x=list(RISK_SEGMENTS), y=list(REVENUE_SEGMENTS), z=z,
        text=text, texttemplate="%{text}", colorscale="Teal",
        hovertemplate="%{y} / %{x}<br>" + title + ": %{text}<extra></extra>",
        colorbar={"title": {"text": title}},
        hoverongaps=False,
    ))
    if field == "mean_predicted_churn":
        figure.update_traces(colorbar_tickformat=".0%")
    figure.update_layout(
        template="plotly_white", margin={"t": 20, "b": 30, "l": 10, "r": 10},
        height=370, font={"family": "Segoe UI, sans-serif", "color": "#20334b"},
        xaxis={"side": "bottom", "title": "Fixed training-reference risk band"},
        yaxis={"autorange": "reversed"},
    )
    return figure


TABLE_COLUMNS = [
    {"name": label, "id": column} for column, label in (
        ("revenue_segment", "Revenue band"), ("risk_segment", "Risk band"),
        ("customers", "Customers"), ("customer_share", "Customer share"),
        ("monthly_charge_total", "Charge total"),
        ("monthly_charge_share", "Charge share"),
        ("mean_monthly_charge", "Mean charge"),
        ("mean_predicted_churn", "Mean predicted churn"),
        ("risk_weighted_monthly_charges", "Risk-weighted charges"),
    )
]


def _table_rows(summary: Summary) -> list[dict]:
    cells = {(row.revenue_segment, row.risk_segment): row for row in summary.segments}
    result = []
    for revenue in REVENUE_SEGMENTS:
        for risk in RISK_SEGMENTS:
            row = cells[revenue, risk].model_dump()
            row["customers"] = f"{row['customers']:,}"
            for name in ("customer_share", "monthly_charge_share", "mean_predicted_churn"):
                row[name] = _number(row[name], percent=True)
            for name in (
                "monthly_charge_total", "mean_monthly_charge", "risk_weighted_monthly_charges",
            ):
                row[name] = _number(row[name])
            result.append(row)
    return result


def _portfolio_layout():
    return html.Main([
        html.Header([
            html.Div([
                html.P("CUSTOMER INTELLIGENCE / SNAPSHOT ANALYTICS", className="eyebrow"),
                html.H1("Portfolio overview"),
                html.P(
                    "Explore revenue exposure and predicted churn with transparent, "
                    "fixed-reference segments.", className="subtitle",
                ),
            ]),
            html.Span("Synthetic data", className="badge"),
        ], className="page-header"),
        html.Div(
            "Descriptive estimates, not measured campaign returns. Training scores are "
            "in-sample; the snapshot includes training, validation and test customers.",
            className="notice",
        ),
        html.Section([
            html.Div([
                _dropdown("Source partition", "partition", _options(["train", "validation", "test"])),
                _dropdown("Revenue band", "revenue", _options(REVENUE_SEGMENTS)),
                _dropdown("Risk band", "risk", _options(RISK_SEGMENTS)),
                _dropdown("Internet service", "internet", [
                    {"label": "Has internet", "value": "yes"},
                    {"label": "No internet", "value": "no"},
                ]),
                _dropdown("Contract", "contract", CATEGORY_OPTIONS["contract"]),
                _dropdown("Payment method", "payment", CATEGORY_OPTIONS["payment_method"]),
            ], className="filter-grid"),
            html.Div([
                html.Button("Apply filters / refresh", id="apply", n_clicks=0),
                html.Button("Reset filters", id="reset", n_clicks=0, className="secondary"),
                html.Span("Changes take effect when you apply filters.", className="muted"),
            ], className="filter-actions"),
        ], className="panel"),
        dcc.Store(id="summary-data"),
        dcc.Loading(html.Div(id="request-status", role="status"), type="circle"),
        html.Div(id="error-message", role="alert"),
        html.Div(id="snapshot-message", className="result-note", role="status"),
        html.Section(id="kpis", className="kpi-grid"),
        html.Section([
            html.Div([
                html.Div([
                    html.H3("Revenue x risk"),
                    html.P(
                        "Training quantiles define the bands. Filtering never refits them.",
                        className="muted",
                    ),
                ]),
                dcc.Dropdown(
                    id="heatmap-metric", value="customers", clearable=False,
                    options=[
                        {"label": label, "value": value} for value, label in (
                            ("customers", "Customer count"),
                            ("monthly_charge_total", "Monthly charge total"),
                            ("mean_predicted_churn", "Mean predicted churn"),
                            ("risk_weighted_monthly_charges", "Risk-weighted charges"),
                        )
                    ],
                    className="metric-select",
                ),
            ], className="panel-heading"),
            dcc.Graph(id="segment-heatmap", config={"displayModeBar": False}),
        ], className="panel"),
        html.Section([
            html.H3("Segment detail"),
            html.P(
                "Shares use the current filtered population. All 12 cells are retained; "
                "undefined means and shares are shown as 'Not defined'.",
                className="muted",
            ),
            dash_table.DataTable(
                id="segment-table", columns=TABLE_COLUMNS, data=[],
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "Segoe UI, sans-serif", "fontSize": 13,
                    "padding": "13px", "textAlign": "right", "minWidth": "110px",
                    "border": "none", "borderBottom": "1px solid #e8edf2",
                },
                style_cell_conditional=[
                    {"if": {"column_id": column}, "textAlign": "left"}
                    for column in ("revenue_segment", "risk_segment")
                ],
                style_header={"fontWeight": "600", "backgroundColor": "#f2f6fa"},
                style_data_conditional=[{
                    "if": {"row_index": "odd"}, "backgroundColor": "#f9fbfd",
                }],
            ),
        ], className="panel"),
        html.Footer(id="provenance", className="provenance"),
    ], className="shell")


def _layout():
    return html.Div([
        dcc.Tabs(id="dashboard-tab", value="portfolio", children=[
            dcc.Tab(label="Portfolio Overview", value="portfolio"),
            dcc.Tab(label="Customer Scoring", value="scoring"),
            dcc.Tab(label="Financial Simulator", value="financial"),
        ], className="dashboard-tabs"),
        html.Div(_portfolio_layout(), id="portfolio-panel"),
        html.Div(scoring_layout(), id="scoring-panel", style={"display": "none"}),
        html.Div(financial_layout(), id="financial-panel", style={"display": "none"}),
    ])


def create_app(client: PortfolioClient | None = None) -> Dash:
    api = client if client is not None else PortfolioClient(
        os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")
    )
    app = Dash(
        __name__, title="Churn Engine | Portfolio",
        assets_folder=str(Path(__file__).with_name("assets")),
        update_title="Loading portfolio...",
    )
    app.layout = _layout

    @app.callback(
        Output("portfolio-panel", "style"), Output("scoring-panel", "style"),
        Output("financial-panel", "style"),
        Input("dashboard-tab", "value"),
    )
    def switch_tab(tab):
        return (
            {"display": "block" if tab == "portfolio" else "none"},
            {"display": "block" if tab == "scoring" else "none"},
            {"display": "block" if tab == "financial" else "none"},
        )

    @app.callback(
        Output("partition", "value"), Output("revenue", "value"),
        Output("risk", "value"), Output("internet", "value"),
        Output("contract", "value"), Output("payment", "value"),
        Input("reset", "n_clicks"), prevent_initial_call=True,
    )
    def reset_filters(_clicks):
        return None, None, None, None, None, None

    @app.callback(
        Output("summary-data", "data"), Output("error-message", "children"),
        Output("request-status", "children"),
        Input("apply", "n_clicks"),
        State("partition", "value"), State("revenue", "value"),
        State("risk", "value"), State("internet", "value"),
        State("contract", "value"), State("payment", "value"),
    )
    def load_summary(_clicks, partition, revenue, risk, internet, contract, payment):
        try:
            filters = PortfolioFilters(
                partition=partition, revenue_segment=revenue, risk_segment=risk,
                has_internet_service={"yes": True, "no": False, None: None}[internet],
                contract=contract.strip() or None if contract else None,
                payment_method=payment.strip() or None if payment else None,
            )
            summary = api.summary(filters)
        except ValidationError:
            logger.warning("Invalid portfolio filter selection.", exc_info=True)
            return None, html.Div("Invalid filters. Reset and try again.", className="error"), ""
        except APIError as error:
            logger.warning("Portfolio request failed: %s", error, exc_info=True)
            return None, html.Div(str(error), className="error"), ""
        return summary.model_dump(mode="json"), "", "Portfolio loaded."

    @app.callback(
        Output("kpis", "children"), Output("segment-heatmap", "figure"),
        Output("segment-table", "data"), Output("snapshot-message", "children"),
        Output("provenance", "children"),
        Input("summary-data", "data"), Input("heatmap-metric", "value"),
    )
    def render_summary(data, metric):
        if data is None:
            figure = go.Figure()
            figure.update_layout(
                template="plotly_white", height=370,
                annotations=[{
                    "text": "No portfolio loaded. Apply filters to load or retry.",
                    "showarrow": False, "xref": "paper", "yref": "paper", "x": .5, "y": .5,
                }],
            )
            return [], figure, [], "", ""
        summary = Summary.model_validate(data)
        note = (
            "No customers match the applied filters. Counts and totals are zero; "
            "means and shares are undefined."
            if summary.overview.customers == 0
            else f"Showing {summary.overview.customers:,} customers for the applied filters."
        )
        provenance = (
            f"Model: {summary.model_run_id} | Freeze: {summary.freeze_run_id} | "
            f"Final test reference: {summary.final_test_run_id}. {summary.interpretation}"
        )
        return _kpis(summary), _heatmap(summary, metric), _table_rows(summary), note, provenance

    register_scoring_callbacks(app, api)
    register_financial_callbacks(app, api)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=8050, debug=False)