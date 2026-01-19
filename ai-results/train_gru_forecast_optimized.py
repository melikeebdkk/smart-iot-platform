import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

from sklearn.ensemble import IsolationForest


# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = os.path.join("..", "datasets", "smart_home_energy_consumption_large.csv")

RUN_DIR = "runs_optimized"
os.makedirs(RUN_DIR, exist_ok=True)

SEQ_LEN_CANDIDATES = [24, 48, 72]  # grid search
BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 8

Z_THRESHOLD = 2.0
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ----------------------------
# Helpers
# ----------------------------
def to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    df = df.copy()
    df["dt"] = dt
    df = df.dropna(subset=["dt"])
    return df


def build_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whole-home hourly series:
    - sum kWh within hour
    - average outdoor temp within hour
    - fill missing hours
    """
    df = to_datetime(df)
    df["hour"] = df["dt"].dt.floor("H")

    hourly_kwh = df.groupby("hour", as_index=False)["Energy Consumption (kWh)"].sum()

    if "Outdoor Temperature (°C)" in df.columns:
        hourly_temp = df.groupby("hour", as_index=False)["Outdoor Temperature (°C)"].mean()
        hourly = hourly_kwh.merge(hourly_temp, on="hour", how="left")
    else:
        hourly = hourly_kwh.copy()
        hourly["Outdoor Temperature (°C)"] = np.nan

    hourly = hourly.sort_values("hour").reset_index(drop=True)
    hourly = hourly.set_index("hour").asfreq("H")

    hourly["Energy Consumption (kWh)"] = hourly["Energy Consumption (kWh)"].fillna(0.0)
    hourly["Outdoor Temperature (°C)"] = hourly["Outdoor Temperature (°C)"].ffill().bfill()

    hourly = hourly.reset_index().rename(columns={"hour": "dt"})
    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"] = out["dt"].dt.hour
    out["day_of_week"] = out["dt"].dt.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    return out


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Static features for time t (prediction time):
    lag1, lag24, lag168 + rolling mean/std
    """
    out = df.copy()
    y = out["Energy Consumption (kWh)"]

    out["lag1"] = y.shift(1)
    out["lag24"] = y.shift(24)
    out["lag168"] = y.shift(168)

    out["roll_mean_24"] = y.rolling(24).mean()
    out["roll_std_24"] = y.rolling(24).std()
    out["roll_mean_168"] = y.rolling(168).mean()

    # fill initial NaNs conservatively
    out = out.fillna(method="bfill").fillna(0.0)
    return out


def zscore_anomaly(series: np.ndarray, z_th: float) -> np.ndarray:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0:
        sigma = 1.0
    z = (series - mu) / sigma
    return z > z_th


def standardize_train_only(train_arr: np.ndarray, full_arr: np.ndarray):
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return (full_arr - mean) / std, mean, std


def make_sequences(seq_features: np.ndarray, static_features: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Predict y[i] using previous seq_len timesteps: X_seq = [i-seq_len ... i-1], static = static_features[i]
    """
    Xs, Ss, Ys = [], [], []
    for i in range(seq_len, len(seq_features)):
        Xs.append(seq_features[i - seq_len:i])
        Ss.append(static_features[i])
        Ys.append(y[i])
    return np.array(Xs), np.array(Ss), np.array(Ys)


def train_val_test_split_time(Xs, Ss, Ys, train_ratio=0.7, val_ratio=0.15):
    n = len(Xs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, S_train, y_train = Xs[:train_end], Ss[:train_end], Ys[:train_end]
    X_val, S_val, y_val = Xs[train_end:val_end], Ss[train_end:val_end], Ys[train_end:val_end]
    X_test, S_test, y_test = Xs[val_end:], Ss[val_end:], Ys[val_end:]

    return (X_train, S_train, y_train), (X_val, S_val, y_val), (X_test, S_test, y_test)


def mape(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def build_model(seq_len: int, seq_feat_dim: int, static_dim: int):
    # sequence branch
    seq_in = layers.Input(shape=(seq_len, seq_feat_dim), name="seq_in")
    x = layers.GRU(64, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(32)(x)

    # static branch
    stat_in = layers.Input(shape=(static_dim,), name="stat_in")
    s = layers.Dense(32, activation="relu")(stat_in)
    s = layers.Dropout(0.1)(s)

    # merge
    h = layers.Concatenate()([x, s])
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(1, name="y_out")(h)

    model = models.Model(inputs=[seq_in, stat_in], outputs=out)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return model


def plot_loss(history, out_png):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_test(y_true, y_pred, out_png, n=300):
    n = min(n, len(y_true))
    plt.figure()
    plt.plot(y_true[:n], label="Actual (kWh)")
    plt.plot(y_pred[:n], label="Predicted (kWh)")
    plt.xlabel("Test time step")
    plt.ylabel("Energy (kWh)")
    plt.title("Test: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    raw = pd.read_csv(CSV_PATH)
    hourly = build_hourly_series(raw)
    hourly = add_time_features(hourly)
    hourly = add_lag_rolling_features(hourly)

    # Base sequence features (per timestep)
    seq_feats = np.column_stack([
        hourly["Energy Consumption (kWh)"].values.astype(np.float32),
        hourly["Outdoor Temperature (°C)"].values.astype(np.float32),
        hourly["hour_sin"].values.astype(np.float32),
        hourly["hour_cos"].values.astype(np.float32),
        hourly["dow_sin"].values.astype(np.float32),
        hourly["dow_cos"].values.astype(np.float32),
    ])

    # Static features (for prediction time t)
    stat_feats = np.column_stack([
        hourly["lag1"].values.astype(np.float32),
        hourly["lag24"].values.astype(np.float32),
        hourly["lag168"].values.astype(np.float32),
        hourly["roll_mean_24"].values.astype(np.float32),
        hourly["roll_std_24"].values.astype(np.float32),
        hourly["roll_mean_168"].values.astype(np.float32),
        hourly["Outdoor Temperature (°C)"].values.astype(np.float32),
    ])

    # Target: log1p(kWh)
    y_kwh = hourly["Energy Consumption (kWh)"].values.astype(np.float32)
    y_log = np.log1p(y_kwh).astype(np.float32)

    # Train split indices on raw timeline (before sequencing)
    n = len(hourly)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Standardize features (fit only on train)
    seq_scaled, seq_mean, seq_std = standardize_train_only(seq_feats[:train_end], seq_feats)
    stat_scaled, stat_mean, stat_std = standardize_train_only(stat_feats[:train_end], stat_feats)

    # Standardize y_log (train only)
    y_scaled, y_mean, y_std = standardize_train_only(y_log[:train_end].reshape(-1, 1), y_log.reshape(-1, 1))
    y_scaled = y_scaled.reshape(-1).astype(np.float32)

    best = {
        "seq_len": None,
        "val_loss": float("inf"),
        "run_path": None
    }

    # Grid search over SEQ_LEN
    for seq_len in SEQ_LEN_CANDIDATES:
        run_name = f"seq{seq_len}"
        run_path = os.path.join(RUN_DIR, run_name)
        os.makedirs(run_path, exist_ok=True)

        # Make sequences
        Xs, Ss, Ys = make_sequences(seq_scaled.astype(np.float32), stat_scaled.astype(np.float32), y_scaled, seq_len)

        (X_train, S_train, y_train), (X_val, S_val, y_val), (X_test, S_test, y_test) = train_val_test_split_time(Xs, Ss, Ys)

        model = build_model(seq_len, Xs.shape[-1], Ss.shape[-1])

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
            callbacks.ModelCheckpoint(os.path.join(run_path, "best_model.keras"), monitor="val_loss", save_best_only=True)
        ]

        hist = model.fit(
            {"seq_in": X_train, "stat_in": S_train},
            y_train,
            validation_data=({"seq_in": X_val, "stat_in": S_val}, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=cb
        )

        # Save loss plot
        loss_png = os.path.join(run_path, "training_loss.png")
        plot_loss(hist, loss_png)

        # Evaluate on test (inverse scale -> inverse log)
        y_pred = model.predict({"seq_in": X_test, "stat_in": S_test}).reshape(-1).astype(np.float32)

        y_test_un = (y_test.reshape(-1, 1) * y_std + y_mean).reshape(-1)
        y_pred_un = (y_pred.reshape(-1, 1) * y_std + y_mean).reshape(-1)

        # back to kWh
        y_test_kwh = np.expm1(y_test_un)
        y_pred_kwh = np.expm1(y_pred_un)

        test_mape = mape(y_test_kwh, y_pred_kwh)
        score = max(0.0, 100.0 - test_mape)

        test_png = os.path.join(run_path, "test_prediction.png")
        plot_test(y_test_kwh, y_pred_kwh, test_png, n=300)

        # record metrics
        metrics = {
            "seq_len": seq_len,
            "test_mape_percent": float(round(test_mape, 2)),
            "score_0_100": float(round(score, 2)),
            "best_val_loss": float(np.min(hist.history["val_loss"]))
        }
        with open(os.path.join(run_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # update best
        if metrics["best_val_loss"] < best["val_loss"]:
            best["seq_len"] = seq_len
            best["val_loss"] = metrics["best_val_loss"]
            best["run_path"] = run_path

    # ---------------------------
    # Anomaly: baseline + ML (IsolationForest) on hourly kWh
    # ---------------------------
    series = y_kwh
    anom_z = zscore_anomaly(series.astype(np.float32), Z_THRESHOLD)
    anomaly_ratio_z = float(round(anom_z.mean() * 100.0, 2))

    # Isolation Forest trained on train portion
    train_series = series[:train_end].reshape(-1, 1)
    full_series = series.reshape(-1, 1)

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
    iso.fit(train_series)
    pred = iso.predict(full_series)  # -1 anomaly, 1 normal
    anom_iso = (pred == -1)
    anomaly_ratio_iso = float(round(anom_iso.mean() * 100.0, 2))

    # ---------------------------
    # Energy status heuristic
    # ---------------------------
    avg_kw_proxy = float(round(series.mean(), 2))  # kWh per hour ~ kW proxy
    ref_capacity_kw = 3.5
    system_status = "Dengeli" if avg_kw_proxy <= ref_capacity_kw else "Riskli"

    # ---------------------------
    # Assemble final results.json for dashboard
    # ---------------------------
    # Read best metrics
    with open(os.path.join(best["run_path"], "metrics.json"), "r", encoding="utf-8") as f:
        best_metrics = json.load(f)

    results = {
        "dataset": {
            "name": "Smart Home Energy Consumption",
            "source_file": os.path.basename(CSV_PATH),
            "total_records_raw": int(len(raw)),
            "total_points_hourly": int(len(hourly)),
            "aggregation": "Whole-home hourly sum (kWh)"
        },
        "forecasting": {
            "model": "Stacked GRU (Huber loss, log1p target, lag+rolling features)",
            "seq_len_candidates": SEQ_LEN_CANDIDATES,
            "chosen_seq_len": best["seq_len"],
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "metric_primary": "MAPE (test)",
            "test_mape_percent": best_metrics["test_mape_percent"],
            "score_0_100": best_metrics["score_0_100"],
            "artifacts": {
                "run_dir": best["run_path"],
                "loss_plot": os.path.join(best["run_path"], "training_loss.png"),
                "test_plot": os.path.join(best["run_path"], "test_prediction.png"),
                "model_file": os.path.join(best["run_path"], "best_model.keras"),
                "metrics_file": os.path.join(best["run_path"], "metrics.json")
            }
        },
        "anomaly_analysis": {
            "baseline": {
                "type": "Statistical baseline (NOT ML)",
                "method": "z-score",
                "z_threshold": Z_THRESHOLD,
                "anomaly_ratio_percent": anomaly_ratio_z
            },
            "ml_based": {
                "type": "ML-based anomaly detection",
                "method": "Isolation Forest",
                "anomaly_ratio_percent": anomaly_ratio_iso
            }
        },
        "energy_efficiency": {
            "reference_capacity_kw": ref_capacity_kw,
            "avg_kw_proxy": avg_kw_proxy,
            "system_status": system_status
        },
        "explanations": {
            "why_this_method": [
                "GRU, zaman serilerinde geçmiş bağımlılıklarını yakalayabildiği için tercih edilmiştir.",
                "log1p dönüşümü ile pik değerlerin etkisi azaltılarak daha stabil öğrenme sağlanmıştır.",
                "Huber loss, gürültülü enerji verilerinde outlier etkisini azaltır.",
                "Lag ve rolling istatistikler, periyodik davranışları yakalamada modele yardımcı olur.",
                "SEQ_LEN grid search ile doğrulama kaybı en düşük pencere seçilmiştir."
            ],
            "limitations": [
                "Bu çalışma 'Tüm Ev' saatlik agregasyon üzerinden yürütülmüştür; cihaz bazlı tahmin ayrıca modellenebilir.",
                "Eksik saatler basit doldurma yöntemleri ile ele alınmıştır.",
                "Daha iyi performans için iç ortam sensörleri (iç sıcaklık/nem), kullanıcı davranışı ve tarifeler gibi ek değişkenler faydalı olabilir."
            ],
            "future_work": [
                "Cihaz bazlı forecasting için ayrı modeller (klima/ısıtıcı gibi) eğitmek.",
                "Anomali tespitinde Autoencoder gibi derin öğrenme tabanlı yöntemleri eklemek.",
                "MQTT akışında online inference ile gerçek zamanlı tahmin/anomali üretmek."
            ]
        }
    }

    out_json = "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ OPTIMIZED eğitim bitti.")
    print(f"✅ Seçilen SEQ_LEN: {best['seq_len']} | Test MAPE: {best_metrics['test_mape_percent']}% | Score: {best_metrics['score_0_100']}")
    print(f"✅ Dashboard verisi yazıldı: {out_json}")
    print(f"✅ En iyi run klasörü: {best['run_path']}")


if __name__ == "__main__":
    main()
