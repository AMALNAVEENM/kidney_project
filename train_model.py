from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import pickle
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pickle is required") from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "kidney_disease.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"


RENAME_COLUMNS = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "red_blood_cells",
    "pus_cell",
    "pus_cell_clumps",
    "bacteria",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "haemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
    "hypertension",
    "diabetes_mellitus",
    "coronary_artery_disease",
    "appetite",
    "peda_edema",
    "aanemia",
    "class",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.columns = RENAME_COLUMNS

    df["packed_cell_volume"] = pd.to_numeric(df["packed_cell_volume"], errors="coerce")
    df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"], errors="coerce")
    df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"], errors="coerce")

    df["diabetes_mellitus"] = df["diabetes_mellitus"].replace({"\\tno": "no", "\\tyes": "yes", " yes": "yes"})
    df["coronary_artery_disease"] = df["coronary_artery_disease"].replace({"\\tno": "no"})
    df["class"] = df["class"].replace({"ckd\\t": "ckd", "notckd": "not ckd"})

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.strip() if hasattr(df[column], "str") else df[column]

    num_cols = [col for col in df.columns if df[col].dtype != "object"]
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Manual label encoding to keep mappings stable.
    binary_maps = {
        "red_blood_cells": {"normal": 0, "abnormal": 1},
        "pus_cell": {"normal": 0, "abnormal": 1},
        "pus_cell_clumps": {"notpresent": 0, "present": 1},
        "bacteria": {"notpresent": 0, "present": 1},
        "hypertension": {"no": 0, "yes": 1},
        "diabetes_mellitus": {"no": 0, "yes": 1},
        "coronary_artery_disease": {"no": 0, "yes": 1},
        "appetite": {"good": 0, "poor": 1},
        "peda_edema": {"no": 0, "yes": 1},
        "aanemia": {"no": 0, "yes": 1},
        "class": {"ckd": 0, "not ckd": 1},
    }

    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    if df.isnull().sum().sum() > 0:
        raise ValueError("Null values remain after preprocessing.")

    return df


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    x = df.drop(columns=["class"])
    y = df["class"]

    selector = SelectKBest(score_func=chi2, k=10)
    x_selected = selector.fit_transform(x, y)
    selected_features = list(x.columns[selector.get_support()])
    x_selected = pd.DataFrame(x_selected, columns=selected_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        criterion="entropy",
        max_depth=11,
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=3,
        n_estimators=130,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, y_pred)

    with MODEL_PATH.open("wb") as fh:
        pickle.dump(model, fh)

    metrics = {
        "selected_features": selected_features,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Training complete. Test accuracy: {test_acc:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    train()
