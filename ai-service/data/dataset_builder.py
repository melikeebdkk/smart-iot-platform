import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

DB_URL = "postgresql://iotuser:iotpass@127.0.0.1:5432/iotdb"



def load_power_series(device_id, lookback=12, resample_rule="5min"):
    engine = create_engine(DB_URL)

    query = f"""
        SELECT time, power
        FROM telemetry
        WHERE device_id = '{device_id}'
        ORDER BY time ASC
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        raise ValueError("DB’den veri gelmedi")

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    df = df.resample(resample_rule).mean()
    df["power"] = df["power"].interpolate()

    values = df[["power"]].values

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(values_scaled) - lookback):
        X.append(values_scaled[i:i + lookback])
        y.append(values_scaled[i + lookback])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, df
