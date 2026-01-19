import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = os.path.join("..", "datasets", "smart_home_energy_consumption_large.csv")

OUT_RESULTS_JSON = "results.json"
OUT_MODEL = "gru_model.keras"
OUT_LOSS_PNG = "training_loss.png"
OUT_TEST_PNG = "test_prediction.png"

SEQ_LEN = 24          # past 24 hours -> predict next hour
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 5

Z_THRESHOLD = 2.0     # anomaly baseline threshold

np.random.seed(42)
tf.random.set_seed(42)


def to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Dataset: Date (YYYY-MM-DD), Time (HH:MM)
    dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    df = df.copy()
    df["dt"] = dt
    df = df.dropna(subset=["dt"])
    return df


def build_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make 'Whole Home' hourly energy series.
    We sum kWh within the hour. Then create features.
    """
    df = to_datetime(df)

    # Hour bucket
    df["hour"] = df["dt"].dt.floor("H")

    # Whole-home hourly energy (kWh)
    hourly = df.groupby("hour", as_index=False)["Energy Consumption (kWh)"].sum()

    # add temp: hourly average outdoor temperature (if present)
    if "Outdoor Temperature (°C)" in df.columns:
        temp_hourly = df.groupby("hour", as_index=False)["Outdoor Temperature (°C)"].mean()
        hourly = hourly.merge(temp_hourly, on="hour", how="left")
    else:
        hourly["Outdoor Temperature (°C)"] = np.nan

    hourly = hourly.sort_values("hour").reset_index(drop=True)

    # Resample to ensure continuous hourly timeline
    hourly = hourly.set_index("hour").asfreq("H")
    # Fill missing energy as 0 (no records that hour)
    hourly["Energy Consumption (kWh)"] = hourly["Energy Consumption (kWh)"].fillna(0.0)
    # Fill temp by forward fill then backfill (simple & defensible)
    hourly["Outdoor Temperature (°C)"] = hourly["Outdoor Temperature (°C)"].ffill().bfill()

    hourly = hourly.reset_index().rename(columns={"hour": "dt"})
    return hourly


def add_time_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["hour_of_day"] = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek

    # cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    return df


def zscore_anomaly(series: np.ndarray, z_th: float) -> np.ndarray:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0:
        sigma = 1.0
    z = (series - mu) / sigma
    return z > z_th


def standardize_train_only(train_arr: np.ndarray, full_arr: np.ndarray):
    """
    Fit mean/std on train, apply to full.
    """
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (full_arr - mean) / std, mean, std


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_val_test_split_time(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def mape(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def main():
    raw = pd.read_csv(CSV_PATH)
    hourly = build_hourly_series(raw)
    hourly = add_time_features(hourly)

    # Target: next-hour energy (kWh). We'll model kWh directly (more honest).
    # You can convert to kW later if needed.
    target = hourly["Energy Consumption (kWh)"].values.astype(np.float32)

    # Features: past energy + temp + time cycles
    feats = np.column_stack([
        hourly["Energy Consumption (kWh)"].values.astype(np.float32),
        hourly["Outdoor Temperature (°C)"].values.astype(np.float32),
        hourly["hour_sin"].values.astype(np.float32),
        hourly["hour_cos"].values.astype(np.float32),
        hourly["dow_sin"].values.astype(np.float32),
        hourly["dow_cos"].values.astype(np.float32),
    ])

    # Split point for scaling: use time-based split on raw timeline FIRST.
    # We'll build sequences after scaling.
    n = len(feats)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    feats_scaled, feat_mean, feat_std = standardize_train_only(feats[:train_end], feats)
    target_scaled, y_mean, y_std = standardize_train_only(target[:train_end].reshape(-1, 1), target.reshape(-1, 1))
    target_scaled = target_scaled.reshape(-1).astype(np.float32)

    # Build sequences
    X_seq, y_seq = make_sequences(feats_scaled.astype(np.float32), target_scaled.astype(np.float32), SEQ_LEN)

    # time-based split on sequence arrays (aligned)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_time(X_seq, y_seq)

    # Model
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, X_seq.shape[-1])),
        layers.GRU(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_loss", save_best_only=True)
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=cb
    )

    # Plot loss
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("GRU Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LOSS_PNG)
    plt.close()

    # Test predictions (scaled -> unscale)
    y_pred_test = model.predict(X_test).reshape(-1).astype(np.float32)

    # unscale
    y_test_un = (y_test.reshape(-1, 1) * y_std + y_mean).reshape(-1)
    y_pred_un = (y_pred_test.reshape(-1, 1) * y_std + y_mean).reshape(-1)

    test_mape = mape(y_test_un, y_pred_un)
    score = max(0.0, 100.0 - test_mape)

    # Plot test prediction (first 200 points)
    show_n = min(200, len(y_test_un))
    plt.figure()
    plt.plot(y_test_un[:show_n], label="Actual (kWh)")
    plt.plot(y_pred_un[:show_n], label="Predicted (kWh)")
    plt.xlabel("Time step (test)")
    plt.ylabel("Energy (kWh)")
    plt.title("Test: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_TEST_PNG)
    plt.close()

    # Anomaly baseline on hourly series (kWh)
    is_anom = zscore_anomaly(hourly["Energy Consumption (kWh)"].values.astype(np.float32), Z_THRESHOLD)
    anomaly_ratio = float(round(is_anom.mean() * 100.0, 2))

    # Capacity heuristic (use average kW-ish proxy: kWh per hour ~= kW)
    avg_kw_proxy = float(round(hourly["Energy Consumption (kWh)"].mean(), 2))
    ref_capacity_kw = 3.5
    system_status = "Dengeli" if avg_kw_proxy <= ref_capacity_kw else "Riskli"

    # Narrative blocks (academic)
    why_text = [
        "GRU modeli, zaman serilerinde geçmiş bağımlılıklarını yakalayabildiği için tercih edilmiştir.",
        "Saat ve gün döngüleri (sin/cos) ile periyodik tüketim davranışı modele aktarılmıştır.",
        "Dış sıcaklık, tüketimi etkileyen dışsal bir faktör olarak feature setine eklenmiştir."
    ]

    limitations = [
        "Veri setindeki kayıtlar cihaz bazlı olsa da bu çalışmada 'Tüm Ev' saatlik agregasyon kullanılmıştır.",
        "Bazı saatlerde veri yoğunluğu düşük olabileceği için eksik saatler basit doldurma yöntemleriyle ele alınmıştır.",
        "Daha iyi performans için ek dışsal değişkenler (iç sıcaklık, kullanıcı davranışı, tarifeler) gerekebilir."
    ]

    future = [
        "Cihaz bazlı forecasting (klima/ısıtıcı gibi) için ayrı modeller eğitmek.",
        "Anomali tespitinde ML tabanlı yöntemler (Isolation Forest / Autoencoder) eklemek.",
        "MQTT akışında online inference ile gerçek zamanlı tahmin/anomali üretmek."
    ]

    results = {
        "dataset": {
            "name": "Smart Home Energy Consumption",
            "source_file": os.path.basename(CSV_PATH),
            "total_records_raw": int(len(raw)),
            "total_points_hourly": int(len(hourly)),
            "aggregation": "Whole-home hourly sum (kWh)"
        },
        "forecasting": {
            "model": "GRU",
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "metric_primary": "MAPE (test)",
            "test_mape_percent": float(round(test_mape, 2)),
            "score_0_100": float(round(score, 2)),
            "artifacts": {
                "loss_plot": OUT_LOSS_PNG,
                "test_plot": OUT_TEST_PNG,
                "saved_model": OUT_MODEL
            }
        },
        "anomaly_analysis": {
            "type": "Statistical baseline (NOT ML)",
            "method": "z-score",
            "z_threshold": Z_THRESHOLD,
            "anomaly_ratio_percent": anomaly_ratio
        },
        "energy_efficiency": {
            "reference_capacity_kw": ref_capacity_kw,
            "avg_kw_proxy": avg_kw_proxy,
            "system_status": system_status
        },
        "explanations": {
            "why_this_method": why_text,
            "limitations": limitations,
            "future_work": future
        }
    }

    with open(OUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Eğitim tamamlandı.")
    print(f"✅ Test MAPE: {results['forecasting']['test_mape_percent']}% | Score: {results['forecasting']['score_0_100']}")
    print(f"✅ Yazıldı: {OUT_RESULTS_JSON}, {OUT_LOSS_PNG}, {OUT_TEST_PNG}, {OUT_MODEL}")


if __name__ == "__main__":
    main()
