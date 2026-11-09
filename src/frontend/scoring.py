"""Editable customer scorecard; all inference and SHAP computation use the API.

The demo profile is illustrative, never fetched from a customer snapshot. Blank
nullable fields are sent as null. Editing any input clears the previous result;
no probability is presented as current until the user explicitly scores again.
"""

import logging
from numbers import Real

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dash_table, dcc, html

from src.data.loader import NULLABLE_COLUMNS, SCHEMA
from src.frontend.categories import CATEGORY_OPTIONS
from src.frontend.client import APIError, Explanation, PortfolioClient
from src.models.explain import CustomerExplanation, waterfall_figure
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
)
from src.models.predict import ModelProvenance, validate_customers


logger = logging.getLogger(__name__)
FORM_COLUMNS = ("customer_id", *FEATURE_COLUMNS)

DEMO_CUSTOMER = {
    "customer_id": "illustrative-demo",
    "age": 43, "annual_income": 55000., "dependents": 1, "tenure": 24,
    "monthlycharges": 85., "totalcharges": 2040., "num_services": 3,
    "customer_satisfaction": 7., "num_complaints": 1., "num_service_calls": 2,
    "late_payments": 0, "avg_monthly_gb": 45., "days_since_last_interaction": 30,
    "credit_score": None, "gender": "Female", "education": "college",
    "marital_status": "married", "contract": "two_year",
    "payment_method": "electronic_check",
    "paperless_billing": 1, "has_phone_service": 1, "has_internet_service": 1,
    "has_online_security": 0, "has_online_backup": 0, "has_device_protection": 1,
    "has_tech_support": 1, "has_streaming_tv": 1, "has_streaming_movies": 1,
}


def _label(column: str) -> str:
    labels = {
        "customer_id": "Customer ID", "monthlycharges": "Monthly charges",
        "totalcharges": "Total charges", "avg_monthly_gb": "Average monthly GB",
    }
    return labels.get(column, column.replace("_", " ").capitalize())


def _field(column: str):
    component_id = f"score-{column}"
    if column in BINARY_COLUMNS:
        control = dcc.Dropdown(
            id=component_id, options=[
                {"label": "Yes", "value": 1}, {"label": "No", "value": 0},
            ], placeholder="Choose yes or no", clearable=True,
        )
    elif column in CATEGORICAL_COLUMNS:
        control = dcc.Dropdown(
            id=component_id, options=CATEGORY_OPTIONS[column],
            placeholder="Choose an option", clearable=True,
        )
    elif column in NUMERIC_COLUMNS:
        control = dcc.Input(
            id=component_id, type="number", min=0,
            step=1 if SCHEMA[column] == "BIGINT" else "any",
            placeholder="Optional (null)" if column in NULLABLE_COLUMNS else "Required",
        )
    else:
        control = dcc.Input(
            id=component_id, type="text",
            placeholder="Required",
        )
    return html.Div([
        html.Label(
            _label(column) + (" (optional)" if column in NULLABLE_COLUMNS else ""),
            htmlFor=component_id,
        ),
        control,
    ], className="filter-field")


def scoring_layout():
    groups = (
        ("Customer identity & categories", ("customer_id", *CATEGORICAL_COLUMNS)),
        ("Usage, value & interactions", NUMERIC_COLUMNS),
        ("Services & billing flags", BINARY_COLUMNS),
    )
    return html.Main([
        html.Header([
            html.Div([
                html.P("CUSTOMER INTELLIGENCE / MODEL EXPLANATIONS", className="eyebrow"),
                html.H1("Customer scoring"),
                html.P(
                    "Score an entered profile and inspect the factors behind its prediction.",
                    className="subtitle",
                ),
            ]),
            html.Span("Frozen CatBoost", className="badge"),
        ], className="page-header"),
        html.Div(
            "Predictions have no established time horizon. SHAP explains the model, "
            "not causal effects. No contact recommendation or campaign selection is made.",
            className="notice",
        ),
        html.Section([
            html.H3("Customer profile"),
            html.P(
                "Enter all required fields. Only the five optional numeric fields may "
                "be blank. Choose categories from the dropdowns. "
                "Count fields require whole numbers.",
                className="muted",
            ),
            html.Div([
                html.Button(
                    "Load illustrative demo", id="score-demo", n_clicks=0,
                    className="secondary",
                ),
                html.P(
                    "The demo is a made-up profile, not a portfolio customer.",
                    className="muted",
                ),
            ], className="filter-actions"),
            *[
                html.Fieldset([
                    html.Legend(title),
                    html.Div([_field(column) for column in columns], className="filter-grid"),
                ], className="score-fieldset") for title, columns in groups
            ],
            html.Button("Score & explain", id="score-submit", n_clicks=0),
        ], className="panel"),
        dcc.Loading(
            html.Div([
                html.Div(id="score-error", role="alert"),
                html.Div(id="score-status", role="status", className="result-note"),
                html.Section(id="score-card"),
                dcc.Graph(id="score-waterfall", figure=blank_waterfall(),
                          config={"displayModeBar": False}),
                html.Section([
                    html.H3("All feature contributions"),
                    html.P(
                        "Positive pushes toward churn; negative pushes away. Units are "
                        "log-odds, not probability points. The waterfall groups omitted "
                        "features so its full sum still reconstructs the model output.",
                        className="muted",
                    ),
                    dash_table.DataTable(
                        id="score-contributions", data=[],
                        columns=[
                            {"name": label, "id": column} for column, label in (
                                ("feature", "Feature"), ("feature_value", "Entered value"),
                                ("shap_log_odds", "SHAP (log-odds)"),
                            )
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "Segoe UI, sans-serif", "textAlign": "left",
                            "padding": "12px", "border": "none",
                            "borderBottom": "1px solid #e8edf2",
                        },
                        style_header={"fontWeight": "600", "backgroundColor": "#f2f6fa"},
                    ),
                ], className="panel"),
                html.Footer(id="score-provenance", className="provenance"),
            ]),
            type="circle",
        ),
    ], className="shell")


def customer_payload(values: list[object]) -> dict[str, object]:
    """Validate form data using the inference contract, then emit JSON API types."""
    customer = dict(zip(FORM_COLUMNS, values, strict=True))
    for column in ("customer_id", *CATEGORICAL_COLUMNS):
        value = customer[column]
        if isinstance(value, str):
            customer[column] = value.strip()
    validate_customers(pd.DataFrame([customer]))
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        value = customer[column]
        if value is not None and isinstance(value, Real):
            customer[column] = (
                int(value) if SCHEMA[column] in {"BIGINT", "BOOLEAN"} else float(value)
            )
    return customer


def blank_waterfall() -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white", height=280,
        annotations=[{
            "text": "Score a profile to see its explanation.",
            "showarrow": False, "xref": "paper", "yref": "paper", "x": .5, "y": .5,
        }],
    )
    return figure


def explanation_figure(explanation: Explanation) -> go.Figure:
    local = CustomerExplanation(
        customer_id=explanation.customer_id,
        churn_probability=explanation.churn_probability,
        base_log_odds=explanation.base_log_odds,
        raw_log_odds=explanation.raw_log_odds,
        contributions=pd.DataFrame([row.model_dump() for row in explanation.contributions]),
        provenance=ModelProvenance(
            model_uri="api-served", source_finalist_run_id=explanation.model_run_id,
            freeze_run_id=explanation.freeze_run_id,
            final_test_run_id=explanation.final_test_run_id,
        ),
    )
    figure = waterfall_figure(local, top_n=10)
    trace = figure.data[0]
    figure.update_traces(orientation="h", x=list(trace.y), y=list(trace.x))
    figure.update_layout(
        template="plotly_white", height=620,
        margin={"l": 160, "r": 30, "t": 80, "b": 65},
        title={"text": f"Churn explanation | {explanation.churn_probability:.2%}", "font": {"size": 16}},
        xaxis={"title": "Churn log-odds (not probability points)", "automargin": True},
        yaxis={"title": "", "autorange": "reversed", "automargin": True},
    )
    return figure


def register_scoring_callbacks(app: Dash, api: PortfolioClient) -> None:
    @app.callback(
        [Output(f"score-{column}", "value") for column in FORM_COLUMNS],
        Input("score-demo", "n_clicks"), prevent_initial_call=True,
    )
    def demo(_clicks):
        return [DEMO_CUSTOMER[column] for column in FORM_COLUMNS]

    @app.callback(
        Output("score-card", "children"), Output("score-waterfall", "figure"),
        Output("score-contributions", "data"), Output("score-provenance", "children"),
        Output("score-error", "children"), Output("score-status", "children"),
        Input("score-submit", "n_clicks"),
        *[Input(f"score-{column}", "value") for column in FORM_COLUMNS],
        prevent_initial_call=True,
    )
    def score_customer(_clicks, *values):
        cleared = ([], blank_waterfall(), [], "")
        if ctx.triggered_id != "score-submit":
            return *cleared, "", "Profile changed. Click Score & explain to refresh the result."
        try:
            customer = customer_payload(list(values))
        except ValueError as error:
            logger.warning("Customer form validation failed: %s", error)
            return *cleared, html.Div(str(error), className="error"), ""
        try:
            result = api.explain(customer)
        except APIError as error:
            logger.warning("Customer explanation request failed: %s", error, exc_info=True)
            return *cleared, html.Div(str(error), className="error"), ""
        card = html.Article([
            html.P("PREDICTED CHURN PROBABILITY", className="kpi-label"),
            html.H2(f"{result.churn_probability:.2%}"),
            html.P(
                f"Entered profile: {result.customer_id}. No established prediction horizon; "
                "this is not a campaign eligibility decision.", className="muted",
            ),
        ], className="kpi-card")
        rows = sorted(
            result.contributions, key=lambda row: abs(row.shap_log_odds), reverse=True,
        )
        table = [
            {
                "feature": row.feature,
                "feature_value": "Missing (null)" if row.feature_value is None else row.feature_value,
                "shap_log_odds": f"{row.shap_log_odds:+.6f}",
            } for row in rows
        ]
        provenance = (
            f"Model: {result.model_run_id} | Freeze: {result.freeze_run_id} | "
            f"Final test reference: {result.final_test_run_id}. {result.interpretation}"
        )
        return card, explanation_figure(result), table, provenance, "", "Profile scored."
