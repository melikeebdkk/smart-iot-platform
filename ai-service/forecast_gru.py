import torch
import numpy as np
import psycopg2
from datetime import datetime, timedelta

from data.dataset_builder import load_power_series
from models.gru_model import GRUModel

# --- DB KAYIT FONKSİYONU ---
def save_forecast_to_db(device_id, preds):
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="iotdb",
            user="iotuser",
            password="iotpass"
        )
        cur = conn.cursor()

        now = datetime.utcnow()

        for i, p in enumerate(preds, 1):
            # Her bir tahmin adımını (T+1, T+2...) saatlik periyotlar olarak ekle
            t = now + timedelta(hours=i)
            cur.execute("""
                INSERT INTO energy_forecasts(device_id, forecast_time, predicted_power)
                VALUES (%s, %s, %s)
            """, (device_id, t, float(p[0])))


        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ {len(preds)} adet tahmin DB'ye başarıyla kaydedildi.")
    except Exception as e:
        print(f"❌ DB Kayıt Hatası: {e}")


def forecast_power(device_id, lookback=12, horizon=5):
    # Dataset'ten veriyi al
    X, y, scaler, df = load_power_series(
        device_id=device_id,
        lookback=lookback
    )

    # Son lookback pencereyi al
    last_window = X[-1]  # shape: (lookback, 1)
    current_input = torch.tensor(
        last_window.reshape(1, lookback, 1),
        dtype=torch.float32
    )

    # Modeli yükle
    model = GRUModel(input_size=1)
    model.load_state_dict(
        torch.load("models/gru_power_model.pt", map_location="cpu")
    )
    model.eval()

    predictions = []

    for _ in range(horizon):
        with torch.no_grad():
            pred = model(current_input)

        predictions.append(pred.item())

        # sliding window güncelle
        new_input = np.append(
            current_input.numpy()[0, 1:, 0],
            pred.item()
        )

        current_input = torch.tensor(
            new_input.reshape(1, lookback, 1),
            dtype=torch.float32
        )

    # Normalize'dan geri çevir
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_real = scaler.inverse_transform(predictions)

    # ⬇️ TAHMİNLERİ DB’YE YAZ
    save_forecast_to_db(device_id, predictions_real)

    return predictions_real.flatten()


# --- TEST ---
if __name__ == "__main__":
    devices = [
        "smart_plug_klima",
        "smart_plug_fridge",
        "smart_plug_tv",
        "smart_plug_light",
        "smart_plug_other"
    ]

    print("🔮 AI Forecast başlatıldı...")

    for device in devices:
        try:
            print(f"\n📡 {device} için tahmin hesaplanıyor...")

            preds = forecast_power(
                device_id=device,
                lookback=12,
                horizon=6
            )

            for i, p in enumerate(preds, 1):
                print(f"{device} → T+{i}: {p:.2f} W")

        except Exception as e:
            print(f"❌ {device} hata verdi: {e}")
