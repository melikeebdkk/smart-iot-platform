import pandas as pd
import json
import numpy as np

CSV_PATH = "../datasets/smart_home_energy_consumption_large.csv"

df = pd.read_csv(CSV_PATH)
total_records = len(df)

# ---------------------------
# 1) Dataset özeti
# ---------------------------
appliance_counts = df["Appliance Type"].value_counts().to_dict()

# ---------------------------
# 2) Anomali tespiti (istatistiksel z-score eşiği)
# - power_watt: kWh -> W (varsayım: kayıt başına ~1 saatlik tüketim)
# - z > 2.0 => anomali
# ---------------------------
df["power_watt"] = df["Energy Consumption (kWh)"] * 1000.0
mean_power = df["power_watt"].mean()
std_power = df["power_watt"].std(ddof=0)
df["z_score"] = (df["power_watt"] - mean_power) / (std_power if std_power != 0 else 1.0)

Z_THRESHOLD = 2.0
df["is_anomaly"] = df["z_score"] > Z_THRESHOLD

anomaly_count = int(df["is_anomaly"].sum())
anomaly_ratio = round((anomaly_count / total_records) * 100, 2)

anomaly_by_device = (
    df[df["is_anomaly"]]["Appliance Type"]
    .value_counts()
    .head(8)
    .to_dict()
)

# ---------------------------
# 3) Enerji tahmini performansı (baseline: rolling mean)
# Not: Burada amaç "kısa vadeli tahmin pipeline" göstermek.
# Metrik: MAPE (hata yüzdesi) -> Akademik standart.
# ---------------------------
df_sorted = df.sort_values(["Date", "Time"]).copy()
df_sorted["actual"] = df_sorted["power_watt"]

# Baseline tahmin: son 5 örneğin ortalaması
WINDOW = 5
df_sorted["predicted"] = df_sorted["actual"].rolling(window=WINDOW).mean()

forecast_df = df_sorted.dropna().copy()

# MAPE hesap (0'a bölünmeyi engelle)
eps = 1e-6
mape = np.mean(np.abs((forecast_df["actual"] - forecast_df["predicted"]) / (forecast_df["actual"] + eps))) * 100
mape = float(round(mape, 2))

# Grafiğe gerçek örnek veri (son 30 nokta)
sample_n = 30
tail = forecast_df.tail(sample_n)
forecast_series = {
    "labels": [f"t{i+1}" for i in range(len(tail))],
    "actual": [float(round(x, 2)) for x in tail["actual"].tolist()],
    "predicted": [float(round(x, 2)) for x in tail["predicted"].tolist()],
}

# ---------------------------
# 4) Yorumlayıcı AI (durum + tavsiye)
# Kapasite referansı: 3.5 kW (projede gösterdiğin referans)
# Ortalama güç (kW) ile kıyaslıyoruz.
# ---------------------------
avg_power_kw = float(round(df["power_watt"].mean() / 1000.0, 2))
reference_capacity_kw = 3.5

if avg_power_kw <= reference_capacity_kw:
    system_status = "Dengeli"
    recommendation = (
        "Enerji tüketimi referans kapasite sınırları içerisindedir. "
        "Sistem dengeli çalışmaktadır. Yüksek tüketimli cihazlar (klima/ısıtıcı) "
        "yoğun kullanım saatlerinde izlenmelidir."
    )
else:
    system_status = "Riskli"
    recommendation = (
        "Enerji tüketimi referans kapasitenin üzerindedir. "
        "Yüksek tüketimli cihazların kullanım süreleri kısaltılmalı ve "
        "yük dağılımı dengelenmelidir."
    )

# ---------------------------
# 5) AI modülleri (hocaya net anlatım)
# ---------------------------
ai_modules = [
    {
        "name": "Anomali Tespiti",
        "method": "İstatistiksel uç değer analizi (z-score > 2.0)",
        "output": "Anomali oranı, cihaz bazlı anomali dağılımı"
    },
    {
        "name": "Enerji Tahmini",
        "method": "Zaman serisi tahmini (baseline: rolling mean, GRU pipeline ile uyumlu)",
        "output": "MAPE hata oranı, gerçek-tahmin karşılaştırma serisi"
    },
    {
        "name": "Yorumlayıcı AI / Tavsiye",
        "method": "Kapasite bazlı karar + öneri üretimi (3.5 kW referans)",
        "output": "Sistem durumu (Dengeli/Riskli) + öneri metni"
    }
]

results = {
    "dataset": {
        "name": "Smart Home Energy Consumption",
        "total_records": int(total_records),
        "columns_used": [
            "Appliance Type",
            "Energy Consumption (kWh)",
            "Date",
            "Time",
            "Outdoor Temperature (°C)"
        ],
        "appliance_counts": appliance_counts
    },
    "anomaly_detection": {
        "method": "z-score",
        "z_threshold": Z_THRESHOLD,
        "anomaly_count": anomaly_count,
        "anomaly_ratio_percent": anomaly_ratio,
        "anomaly_by_device_top": anomaly_by_device
    },
    "forecasting": {
        "metric": "MAPE",
        "forecast_error_mape_percent": mape,
        "series_sample": forecast_series
    },
    "energy_efficiency": {
        "average_consumption_kw": avg_power_kw,
        "reference_capacity_kw": reference_capacity_kw,
        "system_status": system_status,
        "ai_recommendation": recommendation
    },
    "ai_modules": ai_modules
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("✅ AI analiz sonuçları güncellendi → results.json")

