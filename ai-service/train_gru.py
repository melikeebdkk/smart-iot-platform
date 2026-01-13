import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.dataset_builder import load_power_series
from models.gru_model import GRUModel


# Dataset yükle
X, y, scaler, df = load_power_series(
    device_id="device_01",
    lookback=12
)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# Model
model = GRUModel(input_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training
epochs = 20
for epoch in range(epochs):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")


# Modeli kaydet
torch.save(model.state_dict(), "models/gru_power_model.pt")
print("Model kaydedildi: models/gru_power_model.pt")
