# api_server.py
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# ==== Define Models Again ====
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

# ==== Load Models and Scaler ====
model = GreenTimePredictor()
model.load_state_dict(torch.load("green_model.pth"))
model.eval()

sequencer = Sequencer()
sequencer.load_state_dict(torch.load("sequencer_model.pth"))
sequencer.eval()

scaler_input = joblib.load("scaler_input.save")

# ==== FastAPI ====
app = FastAPI()

class TrafficInput(BaseModel):
    vehicle_counts: List[int]
    green_status: List[int]

@app.post("/predict")
def predict_traffic(data: TrafficInput):
    if len(data.vehicle_counts) != 6 or len(data.green_status) != 6:
        return {"error": "Must provide 6 vehicle counts and 6 green status values."}

    inputs_scaled = scaler_input.transform(np.array(data.vehicle_counts).reshape(-1, 1))
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

    with torch.no_grad():
        green_times = model(inputs_tensor).numpy().flatten()
        green_times_sec = green_times * 60

        total_time = sum(green_times_sec)
        if total_time > 0:
            scaled_green_times = [(gt / total_time) * 300 for gt in green_times_sec]
        else:
            scaled_green_times = [0.0] * 6

        state = np.concatenate([inputs_scaled.flatten(), green_times, data.green_status])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        q_values = sequencer(state_tensor).numpy().flatten()
        ordered_indices = list(np.argsort(-q_values))

        green_order = [f"Cross {i+1}" for i in ordered_indices]
        ordered_times = [round(scaled_green_times[i], 2) for i in ordered_indices]

        return {
            "green_order": green_order,
            "green_times_sec": ordered_times
        }
