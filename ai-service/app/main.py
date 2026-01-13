from app.insights_basic import get_basic_power_insights
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecast_gru import forecast_power
from app.anomaly_batch import check_recent_anomalies

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ai-service alive"}


@app.get("/forecast")
def forecast(device_id: str, horizon: int = 5):
    preds = forecast_power(
        device_id=device_id,
        lookback=12,
        horizon=horizon
    )

    return {
        "device_id": device_id,
        "forecast": preds.tolist()
    }


@app.get("/anomaly/recent")
def recent_anomalies(device_id: str, minutes: int = 30):
    anomalies = check_recent_anomalies(
        device_id=device_id,
        minutes=minutes
    )

    return {
        "device_id": device_id,
        "minutes": minutes,
        "anomalies": anomalies
    }

@app.get("/insights/basic")
def basic_insights(device_id: str, hours: int = 24):
    insights = get_basic_power_insights(
        device_id=device_id,
        hours=hours
    )

    return insights
