# Kidney Disease Prediction Web App

Professional Flask web app for CKD prediction using your trained `model.pkl`.

## Features
- Clean clinical-style UI
- Input validation
- Prediction + probability display
- JSON API endpoint for integration
- Health endpoint
- Reproducible model training pipeline
- Persisted training metrics

## End-to-End Workflow
```bash
pip install -r requirements.txt
python train_model.py
python verify_app.py
python app.py
```

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Verify
Run automated checks:
```bash
python verify_app.py
```

## Retrain Model
To retrain from dataset and regenerate artifacts:
```bash
python train_model.py
```

This writes:
- `model.pkl`
- `model_metrics.json`

## API
`POST /api/predict`

Example JSON body:
```json
{
  "age": 48,
  "albumin": 1,
  "sugar": 0,
  "blood_glucose_random": 121,
  "blood_urea": 36,
  "serum_creatinine": 1.2,
  "haemoglobin": 13.4,
  "packed_cell_volume": 41,
  "white_blood_cell_count": 7800,
  "hypertension": 0
}
```

## Clinical Note
This app is a machine-learning decision-support tool and not a medical diagnosis system. Final clinical decisions should be made by qualified professionals.
