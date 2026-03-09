from app import app


def run_checks() -> None:
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200, f"/health failed: {health.status_code}"
    body = health.get_json()
    assert body["model_loaded"] is True, "Model did not load"
    assert body["feature_count"] == 10, f"Unexpected feature count: {body['feature_count']}"

    valid_payload = {
        "age": 48,
        "albumin": 1,
        "sugar": 0,
        "blood_glucose_random": 121,
        "blood_urea": 36,
        "serum_creatinine": 1.2,
        "haemoglobin": 13.4,
        "packed_cell_volume": 41,
        "white_blood_cell_count": 7800,
        "hypertension": 0,
    }

    pred = client.post("/api/predict", json=valid_payload)
    assert pred.status_code == 200, f"/api/predict failed: {pred.status_code}"
    out = pred.get_json()
    assert out["ok"] is True, "Prediction response not ok"
    assert out["result"]["prediction_raw"] in (0, 1), "Unexpected class prediction"
    assert 0 <= out["result"]["ckd_probability"] <= 1, "CKD probability out of range"
    assert 0 <= out["result"]["non_ckd_probability"] <= 1, "Non-CKD probability out of range"

    invalid = client.post("/api/predict", json={"age": "abc"})
    assert invalid.status_code == 400, "Invalid payload should fail with 400"

    print("All checks passed.")


if __name__ == "__main__":
    run_checks()
