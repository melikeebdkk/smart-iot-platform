from data.dataset_builder import load_power_series

X, y, scaler, df = load_power_series(
    device_id="device_01",
    lookback=12
)

print("Dataframe shape:", df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
