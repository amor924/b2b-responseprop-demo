from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


NUMERIC_COLS = [
    "company_size",
    "revenue",
    "tenure_months",
    "contract_remaining_months",
    "current_product_count",
    "avg_monthly_bill",
    "email_open_rate",
    "website_visits_30d",
    "prior_campaign_engagement",
    "legacy_infrastructure_flag",
    "multi_location_flag",
    "renewal_window_flag",
    "high_engagement_flag",
    "large_account_flag",
    "revenue_per_employee",
]

CATEGORICAL_COLS = [
    "industry",
    "cloud_provider",
    "company_size_bin",
    "website_visits_bin",
    "contract_remaining_bin",
]


def build_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_COLS),
        ]
    )

    model = LogisticRegression(
    penalty="l2",
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced"
)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return pipe
