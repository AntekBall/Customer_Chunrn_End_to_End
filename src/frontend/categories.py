"""Dropdown vocabulary for the synthetic telecom dataset.

Labels are presentation-only; values preserve the exact source spelling.
The API still accepts unseen nonblank categories for programmatic inference.
Update this vocabulary when supporting a different source dataset.
"""


CATEGORY_OPTIONS = {
    "gender": [
        {"label": "Female", "value": "Female"},
        {"label": "Male", "value": "Male"},
        {"label": "Other", "value": "Other"},
    ],
    "education": [
        {"label": "High school", "value": "high_school"},
        {"label": "College", "value": "college"},
        {"label": "Bachelor's degree", "value": "bachelor"},
        {"label": "Master's degree", "value": "master"},
        {"label": "PhD", "value": "phd"},
    ],
    "marital_status": [
        {"label": "Single", "value": "single"},
        {"label": "Married", "value": "married"},
        {"label": "Divorced", "value": "divorced"},
        {"label": "Widowed", "value": "widowed"},
    ],
    "contract": [
        {"label": "Month-to-month", "value": "month_to_month"},
        {"label": "One year", "value": "one_year"},
        {"label": "Two years", "value": "two_year"},
    ],
    "payment_method": [
        {"label": "Bank transfer", "value": "bank_transfer"},
        {"label": "Credit card", "value": "credit_card"},
        {"label": "Electronic check", "value": "electronic_check"},
        {"label": "Mailed check", "value": "mailed_check"},
    ],
}
