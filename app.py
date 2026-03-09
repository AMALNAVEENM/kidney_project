from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    import pickle
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pickle is required to load model.pkl") from exc


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    min_value: float
    max_value: float
    step: float
    default: float
    help_text: str
    integer_only: bool = False


FEATURES: list[FeatureSpec] = [
    FeatureSpec("age", "Age (years)", 1, 120, 1, 45, "Patient age in years", integer_only=True),
    FeatureSpec("albumin", "Albumin (0-5)", 0, 5, 1, 1, "Urine albumin level", integer_only=True),
    FeatureSpec("sugar", "Sugar (0-5)", 0, 5, 1, 0, "Urine sugar level", integer_only=True),
    FeatureSpec(
        "blood_glucose_random",
        "Random Blood Glucose (mg/dL)",
        22,
        500,
        1,
        120,
        "Random blood glucose",
        integer_only=True,
    ),
    FeatureSpec("blood_urea", "Blood Urea (mg/dL)", 1, 400, 1, 50, "Blood urea level", integer_only=True),
    FeatureSpec(
        "serum_creatinine",
        "Serum Creatinine (mg/dL)",
        0.1,
        25,
        0.1,
        1.2,
        "Serum creatinine level",
    ),
    FeatureSpec("haemoglobin", "Hemoglobin (g/dL)", 3, 20, 0.1, 12.5, "Hemoglobin"),
    FeatureSpec(
        "packed_cell_volume",
        "Packed Cell Volume (%)",
        10,
        60,
        1,
        40,
        "Packed cell volume",
        integer_only=True,
    ),
    FeatureSpec(
        "white_blood_cell_count",
        "White Blood Cell Count (cells/cumm)",
        2000,
        30000,
        1,
        8000,
        "Total white blood cell count",
        integer_only=True,
    ),
    FeatureSpec(
        "hypertension",
        "Hypertension (0 = No, 1 = Yes)",
        0,
        1,
        1,
        0,
        "Known hypertension status",
        integer_only=True,
    ),
]


def load_model(model_path: Path) -> Any:
    with model_path.open("rb") as fh:
        return pickle.load(fh)


MODEL = load_model(MODEL_PATH)
MODEL_FEATURES = list(getattr(MODEL, "feature_names_in_", []))
if not MODEL_FEATURES:
    MODEL_FEATURES = [feature.key for feature in FEATURES]

FEATURE_BY_KEY = {feature.key: feature for feature in FEATURES}
MISSING_FEATURES = [name for name in MODEL_FEATURES if name not in FEATURE_BY_KEY]


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


MODEL_METRICS = load_metrics()


def parse_features(payload: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    errors: list[str] = []

    for feature_name in MODEL_FEATURES:
        feature = FEATURE_BY_KEY.get(feature_name)
        if feature is None:
            errors.append(f"Model requires unsupported feature: {feature_name}")
            continue

        raw = payload.get(feature.key)
        if raw is None or str(raw).strip() == "":
            errors.append(f"{feature.label} is required.")
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{feature.label} must be numeric.")
            continue

        if value < feature.min_value or value > feature.max_value:
            errors.append(
                f"{feature.label} must be between {feature.min_value} and {feature.max_value}."
            )
            continue

        if feature.integer_only and not float(value).is_integer():
            errors.append(f"{feature.label} must be a whole number.")
            continue

        values[feature.key] = value

    return values, errors


def predict(values: dict[str, float]) -> dict[str, Any]:
    ordered = [values[feature_name] for feature_name in MODEL_FEATURES]
    frame = pd.DataFrame([ordered], columns=MODEL_FEATURES)

    pred = int(MODEL.predict(frame)[0])
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(frame)[0]
        classes = list(getattr(MODEL, "classes_", [0, 1]))
        prob_by_class = {int(classes[i]): float(probs[i]) for i in range(len(classes))}
        ckd_probability = prob_by_class.get(0, float("nan"))
        non_ckd_probability = prob_by_class.get(1, float("nan"))
    else:
        ckd_probability = float("nan")
        non_ckd_probability = float("nan")

    # Training notebook maps: ckd -> 0, not ckd -> 1
    label = "CKD Detected" if pred == 0 else "No CKD Detected"
    confidence = max(ckd_probability, non_ckd_probability)
    risk_level = "High" if ckd_probability >= 0.7 else ("Moderate" if ckd_probability >= 0.4 else "Low")

    return {
        "prediction_raw": pred,
        "prediction_label": label,
        "ckd_probability": ckd_probability,
        "non_ckd_probability": non_ckd_probability,
        "confidence": confidence,
        "risk_level": risk_level,
    }


app = Flask(__name__)


@app.get("/")
def home():
    display_features = [FEATURE_BY_KEY[name] for name in MODEL_FEATURES if name in FEATURE_BY_KEY]
    return render_template("index.html", features=display_features, form_data={}, metrics=MODEL_METRICS)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if not MISSING_FEATURES else "degraded",
            "model_loaded": True,
            "feature_count": len(MODEL_FEATURES),
            "missing_features": MISSING_FEATURES,
        }
    )


@app.post("/predict")
def predict_form():
    form_data = request.form.to_dict()
    values, errors = parse_features(form_data)
    if MISSING_FEATURES:
        errors.append(f"Model is misconfigured. Missing metadata for: {', '.join(MISSING_FEATURES)}")
    if errors:
        display_features = [FEATURE_BY_KEY[name] for name in MODEL_FEATURES if name in FEATURE_BY_KEY]
        return (
            render_template(
                "index.html",
                features=display_features,
                form_data=form_data,
                errors=errors,
                metrics=MODEL_METRICS,
            ),
            400,
        )

    result = predict(values)
    display_features = [FEATURE_BY_KEY[name] for name in MODEL_FEATURES if name in FEATURE_BY_KEY]
    return render_template(
        "index.html",
        features=display_features,
        form_data=form_data,
        result=result,
        metrics=MODEL_METRICS,
    )


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    values, errors = parse_features(payload)
    if MISSING_FEATURES:
        errors.append(f"Model is misconfigured. Missing metadata for: {', '.join(MISSING_FEATURES)}")
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    return jsonify({"ok": True, "result": predict(values)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
