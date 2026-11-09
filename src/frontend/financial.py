import logging

import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dash_table, dcc, html

from src.api.campaign import ScenarioInput, SimulationRequest
from src.data.dashboard import PortfolioFilters
from src.data.portfolio import REVENUE_SEGMENTS, RISK_SEGMENTS
from src.frontend.categories import CATEGORY_OPTIONS
from src.frontend.client import APIError, PortfolioClient


logger = logging.getLogger(__name__)
SCENARIO_FIELDS = (
    "contact_cost", "gross_margin", "retained_months",
    "prevention_probability", "capacity_fraction",
)
FILTER_FIELDS = (
    "partition", "revenue_segment", "risk_segment",
    "contract", "payment_method", "has_internet_service",
)


def _number(value, percent=False):
    if value is None:
        return "Not defined"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def _dropdown(name, label, options):
    return html.Div([
        html.Label(label, htmlFor=f"finance-{name}"),
        dcc.Dropdown(
            id=f"finance-{name}", options=options, placeholder="All", clearable=True,
        ),
    ], className="filter-field")


def _options(labels):
    return [{"label": label, "value": label} for label in labels]


def financial_layout():
    controls = (
        ("contact_cost", "Cost per contact", 2, 0, None, .5),
        ("gross_margin", "Gross margin (%)", 20, 0, 100, 1),
        ("retained_months", "Retained months", 6, .01, None, 1),
        ("prevention_probability", "Prevention probability (%)", 20, 0, 100, 1),
        ("capacity_fraction", "Contact capacity (%)", 10, 0, 100, 1),
    )
    return html.Main([
        html.Header([
            html.Div([
                html.P("CUSTOMER INTELLIGENCE / SCENARIO ECONOMICS", className="eyebrow"),
                html.H1("Financial simulator"),
                html.P(
                    "Explore a hypothetical retention campaign under explicit assumptions.",
                    className="subtitle",
                ),
            ]),
            html.Span("Scenario estimates", className="badge"),
        ], className="page-header"),
        html.Div(
            "Not measured or causal returns. Selection uses the full snapshot, including "
            "in-sample training scores. Snapshot membership does not establish active "
            "status or contact consent. Currency units follow the source charges.",
            className="notice",
        ),
        html.Section([
            html.H3("Campaign assumptions"),
            html.P(
                "Expected net value = churn probability x monthly charge x margin x "
                "retained months x prevention probability - contact cost. Only customers "
                "with net > 0 qualify; rank by net descending, then customer ID ascending.",
                className="muted",
            ),
            html.Div([
                html.Div([
                    html.Label(label, htmlFor=f"finance-{name}"),
                    dcc.Input(
                        id=f"finance-{name}", type="number", value=default,
                        min=minimum, max=maximum, step=step,
                    ),
                ], className="filter-field")
                for name, label, default, minimum, maximum, step in controls
            ], className="filter-grid"),
            html.H3("Display filters", className="finance-filter-heading"),
            html.P(
                "These narrow the displayed results only. They never reallocate global "
                "capacity or select a new campaign audience.",
                className="muted",
            ),
            html.Div([
                _dropdown("partition", "Source partition", _options(["train", "validation", "test"])),
                _dropdown("revenue_segment", "Revenue band", _options(REVENUE_SEGMENTS)),
                _dropdown("risk_segment", "Risk band", _options(RISK_SEGMENTS)),
                _dropdown("contract", "Contract", CATEGORY_OPTIONS["contract"]),
                _dropdown("payment_method", "Payment method", CATEGORY_OPTIONS["payment_method"]),
                _dropdown("has_internet_service", "Internet service", [
                    {"label": "Has internet", "value": "yes"},
                    {"label": "No internet", "value": "no"},
                ]),
            ], className="filter-grid"),
            html.Div([
                html.Button("Run simulation", id="finance-submit", n_clicks=0),
                html.Span(
                    "One scenario per click; editing controls clears the previous result.",
                    className="muted",
                ),
            ], className="filter-actions"),
        ], className="panel"),
        dcc.Loading(html.Div([
            html.Div(id="finance-error", role="alert"),
            html.Div(id="finance-status", role="status", className="result-note"),
            html.Div(id="finance-global"),
            html.Div(id="finance-displayed"),
            dcc.Graph(id="finance-matrix", figure=blank_matrix(), config={"displayModeBar": False}),
            html.Section([
                html.H3("Displayed campaign segments"),
                html.P(
                    "All 12 cells are retained. Empty-cell means and fractions are undefined.",
                    className="muted",
                ),
                dash_table.DataTable(
                    id="finance-table", data=[], columns=[
                        {"id": name, "name": label} for name, label in (
                            ("revenue_segment", "Revenue band"), ("risk_segment", "Risk band"),
                            ("customers", "Customers"), ("eligible_customers", "Eligible"),
                            ("selected_customers", "Selected"), ("selected_fraction", "Selected share"),
                            ("eligible_expected_net_value", "Eligible expected net"),
                            ("selected_expected_net_value", "Selected expected net"),
                            ("mean_expected_net_value", "Mean expected net / customer"),
                        )
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontFamily": "Segoe UI, sans-serif", "textAlign": "right",
                        "padding": "12px", "border": "none",
                        "borderBottom": "1px solid #e8edf2",
                    },
                    style_header={"fontWeight": "600", "backgroundColor": "#f2f6fa"},
                ),
            ], className="panel"),
            html.Footer(id="finance-provenance", className="provenance"),
        ]), type="circle"),
    ], className="shell")


def simulation_request(values) -> SimulationRequest:
    scenario_values = dict(zip(SCENARIO_FIELDS, values[:5], strict=True))
    for name in ("gross_margin", "prevention_probability", "capacity_fraction"):
        value = scenario_values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Enter numeric assumptions; percentages must be between 0 and 100.")
        scenario_values[name] = value / 100
    filters = dict(zip(FILTER_FIELDS, values[5:], strict=True))
    internet = filters["has_internet_service"]
    if internet not in (None, "yes", "no"):
        raise ValueError("Invalid internet-service filter.")
    filters["has_internet_service"] = {"yes": True, "no": False, None: None}[internet]
    return SimulationRequest(
        scenario=ScenarioInput.model_validate(scenario_values),
        display_filters=PortfolioFilters.model_validate(filters),
    )


def blank_matrix():
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white", height=360,
        annotations=[{
            "text": "Run a scenario to see selected expected net value.",
            "showarrow": False, "xref": "paper", "yref": "paper", "x": .5, "y": .5,
        }],
    )
    return figure


def _cards(totals, *, title, capacity=None, unused=None):
    metrics = [
        ("CUSTOMERS", f"{totals.customers:,}"),
        ("ELIGIBLE", f"{totals.eligible_customers:,}"),
        ("SELECTED", f"{totals.selected_customers:,}"),
        ("SCENARIO ROI", _number(totals.scenario_roi, percent=True)),
        ("EXPECTED RETAINED CONTRIBUTION", _number(totals.expected_retained_contribution)),
        ("CAMPAIGN COST", _number(totals.campaign_cost)),
        ("EXPECTED NET VALUE", _number(totals.expected_net_value)),
        ("EXPECTED PREVENTED CHURN", _number(totals.expected_prevented_churn)),
    ]
    return html.Section([
        html.H3(title),
        html.P(
            f"Global capacity: {capacity:,} | Unused capacity: {unused:,}"
            if capacity is not None else
            "Subset of the global audience, not a separately optimized campaign.",
            className="muted",
        ),
        html.Div([
            html.Article([
                html.P(label, className="kpi-label"), html.H2(value),
            ], className="kpi-card") for label, value in metrics
        ], className="kpi-grid"),
        html.P("ROI is undefined when campaign cost is zero.", className="muted"),
    ])


def register_financial_callbacks(app: Dash, api: PortfolioClient):
    @app.callback(
        Output("finance-global", "children"), Output("finance-displayed", "children"),
        Output("finance-matrix", "figure"), Output("finance-table", "data"),
        Output("finance-provenance", "children"), Output("finance-error", "children"),
        Output("finance-status", "children"),
        Input("finance-submit", "n_clicks"),
        *[Input(f"finance-{name}", "value") for name in (*SCENARIO_FIELDS, *FILTER_FIELDS)],
        prevent_initial_call=True,
    )
    def simulate(_clicks, *values):
        cleared = ([], [], blank_matrix(), [], "")
        if ctx.triggered_id != "finance-submit":
            return *cleared, "", "Controls changed. Run simulation to refresh the results."
        try:
            request = simulation_request(values)
        except ValueError as error:
            logger.warning("Invalid campaign form: %s", error)
            return *cleared, html.Div(str(error), className="error"), ""
        try:
            result = api.simulate(request)
        except APIError as error:
            logger.warning("Campaign simulation request failed: %s", error, exc_info=True)
            return *cleared, html.Div(str(error), className="error"), ""
        cells = {
            (row.revenue_segment, row.risk_segment): row for row in result.displayed_segments
        }
        ordered = [cells[revenue, risk] for revenue in REVENUE_SEGMENTS for risk in RISK_SEGMENTS]
        z = [
            [cells[revenue, risk].selected_expected_net_value for risk in RISK_SEGMENTS]
            for revenue in REVENUE_SEGMENTS
        ]
        figure = go.Figure(go.Heatmap(
            z=z, x=list(RISK_SEGMENTS), y=list(REVENUE_SEGMENTS),
            colorscale="Teal", text=[[_number(value) for value in row] for row in z],
            texttemplate="%{text}", colorbar={
                "title": {"text": "Expected net", "side": "bottom"},
                "orientation": "h", "y": -.4, "thickness": 12,
                "tickangle": 0, "nticks": 3,
            },
            hovertemplate="%{y} / %{x}<br>Selected expected net: %{text}<extra></extra>",
        ))
        figure.update_layout(
            template="plotly_white", height=460,
            title={"text": "Selected expected net<br>(displayed scenario estimate)", "font": {"size": 14}},
            yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
            xaxis={"tickfont": {"size": 10}},
            margin={"l": 95, "r": 10, "b": 150, "t": 70},
        )
        table = []
        for cell in ordered:
            row = cell.model_dump()
            for name in ("customers", "eligible_customers", "selected_customers"):
                row[name] = f"{row[name]:,}"
            for name in ("eligible_fraction", "selected_fraction"):
                row[name] = _number(row[name], percent=True)
            for name in (
                "eligible_expected_net_value", "selected_expected_net_value", "mean_expected_net_value",
            ):
                row[name] = _number(row[name])
            table.append(row)
        provenance = (
            f"Model: {result.model_run_id} | Freeze: {result.freeze_run_id} | "
            f"Final test reference: {result.final_test_run_id}. {result.interpretation}"
        )
        status = (
            "No customers match the display filters. Global campaign results are unchanged."
            if result.displayed_summary.customers == 0 else
            "Scenario calculated. Display filters did not reallocate the global selection."
        )
        return (
            _cards(result.global_summary, title="Global campaign",
                   capacity=result.capacity_customers, unused=result.unused_capacity),
            _cards(result.displayed_summary, title="Displayed subset"),
            figure, table, provenance, "", status,
        )
