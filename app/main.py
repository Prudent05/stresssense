from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path

from src.engine import predict_stress

app = FastAPI(title="StressSense API", version="0.2")

ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


class PredictRequest(BaseModel):
    # New: consumer-friendly inputs
    inputs: dict | None = None
    # Old: raw model features (we'll map a few if provided)
    features: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/assessment", response_class=HTMLResponse)
def assessment(request: Request):
    return templates.TemplateResponse("assessment.html", {"request": request})



@app.get("/what-if", response_class=HTMLResponse)
def what_if(request: Request):
    return templates.TemplateResponse("what-if-simulator.html", {"request": request})


def _features_to_inputs(features: dict) -> dict:
    """
    Backwards-compat: if someone sends old keys like B_1, D_39, S_3, P_2,
    convert them into consumer inputs.
    """
    inputs = {}

    # B_1 is expected 0–1 in your old UI mapping -> convert to 0–100
    if "B_1" in features and features["B_1"] is not None:
        try:
            inputs["utilization_pct"] = float(features["B_1"]) * 100.0
        except Exception:
            pass

    # D_39 already behaves like a count in your earlier mapping
    if "D_39" in features and features["D_39"] is not None:
        try:
            inputs["missed_payments"] = float(features["D_39"])
        except Exception:
            pass

    # S_3 expected 0–1 -> 0–100
    if "S_3" in features and features["S_3"] is not None:
        try:
            inputs["spending_volatility"] = float(features["S_3"]) * 100.0
        except Exception:
            pass

    # P_2 expected 0–1 -> 0–100
    if "P_2" in features and features["P_2"] is not None:
        try:
            inputs["income_stability"] = float(features["P_2"]) * 100.0
        except Exception:
            pass

    return inputs


@app.post("/predict")
def predict(req: PredictRequest):
    # Preferred: consumer inputs
    if req.inputs is not None:
        return predict_stress(req.inputs)

    # Backwards-compat: old "features" payload
    if req.features is not None:
        mapped = _features_to_inputs(req.features)
        # If mapping fails, still provide defaults in engine
        return predict_stress(mapped)

    # If nothing provided:
    return predict_stress({})
