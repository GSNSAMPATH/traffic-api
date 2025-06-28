# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib  # for saving scaler

# ==== Load and Preprocess Dataset ====
df = pd.read_csv("traffic-prediction-dataset.csv")  # Use your real path

scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)

# ==== Green Time Prediction Model ====
class GreenTimePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

X_all = df.values.reshape(-1, 1)
y_all = df.values.reshape(-1, 1) / 60.0

scaler_input = MinMaxScaler()
X_scaled = scaler_input.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2)

model = GreenTimePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print("Training GreenTimePredictor...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("GreenTimePredictor trained.")

# ==== Sequencer Model ====
class Sequencer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, state):
        return self.net(state)

sequencer = Sequencer()
seq_optimizer = optim.Adam(sequencer.parameters(), lr=0.001)
seq_criterion = nn.MSELoss()

print("Training Sequencer...")
for epoch in range(500):
    for _ in range(50):
        vehicle_counts = np.random.randint(10, 1000, size=6)
        green_status = np.random.randint(0, 2, size=6)

        inputs_scaled = scaler_input.transform(vehicle_counts.reshape(-1, 1))
        inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

        green_times = model(inputs_tensor).detach().numpy().flatten()

        state = np.concatenate([inputs_scaled.flatten(), green_times, green_status])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        priority_score = vehicle_counts + green_status * 200
        target_order = np.argsort(-priority_score)

        target = np.zeros(6)
        for i, idx in enumerate(target_order):
            target[idx] = 6 - i

        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        seq_optimizer.zero_grad()
        q_values = sequencer(state_tensor)
        loss = seq_criterion(q_values, target_tensor)
        loss.backward()
        seq_optimizer.step()

print("Sequencer trained.")

# ==== Save Everything ====
torch.save(model.state_dict(), "green_model.pth")
torch.save(sequencer.state_dict(), "sequencer_model.pth")
joblib.dump(scaler_input, "scaler_input.save")

print("Models and scaler saved.")
