from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

from .data import make_train_test


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main():
    repo_root = Path(__file__).resolve().parents[1]

    pipe = joblib.load(repo_root / "models" / "model.joblib")

    X_train, X_test, y_train, y_test = make_train_test()

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)
    ks_test = ks_statistic(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=20)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration Curve")
    plt.savefig(reports_dir / "calibration_curve.png")
    plt.close()

    # Save metrics
    metrics = {
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "ks_test": float(ks_test),
        "brier_test": float(brier),
        "test_response_rate": float(y_test.mean()),
    }
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(repo_root / "models" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Coefficients
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()
    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
    }).sort_values("coefficient", key=abs, ascending=False)

    coef_df.to_csv(reports_dir / "model_coefficients.csv", index=False)

    print("Evaluation complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
