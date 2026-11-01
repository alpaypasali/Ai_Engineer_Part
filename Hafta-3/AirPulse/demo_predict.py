
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Proje yolları
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from validation import validate_pm25_window


# Kaydedilmiş LSTM modelimizin mimarisi
class AirPulseLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])


def predict_pm25(timestamps, values):

    # 1. Girdi doğrulama
    validate_pm25_window(timestamps, values)

    # 2. Scaler dosyalarını yükle
    input_scaler = joblib.load(
        MODELS_DIR / "input_scaler.joblib"
    )

    target_scaler = joblib.load(
        MODELS_DIR / "target_scaler.joblib"
    )

    # 3. Model ağırlıklarını yükle
    model = AirPulseLSTM()

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "lstm_state_dict.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # 4. Girdiyi ölçeklendir
    values = np.asarray(values, dtype=np.float32)

    scaled = input_scaler.transform(
        values.reshape(-1, 1)
    )

    X = torch.tensor(
        scaled.reshape(1, 24, 1),
        dtype=torch.float32
    )

    # 5. Tahmin oluştur
    with torch.no_grad():
        prediction_scaled = model(X).numpy()

    # 6. Orijinal PM2.5 birimine dönüştür
    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )[0, 0]

    return float(prediction)


def main():

    # Gerçek veri setini yükle
    data_path = (
        PROJECT_DIR
        / "data"
        / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
    )

    df = pd.read_csv(data_path)

    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Veri setindeki en son geçerli 24 saatlik pencereyi bul
    for end in range(len(df), 23, -1):

        window = df.iloc[end-24:end]

        timestamps = window["timestamp"].tolist()
        values = window["PM2.5"].to_numpy()

        try:
            validate_pm25_window(timestamps, values)
            break

        except ValueError:
            continue

    else:
        raise RuntimeError(
            "Veri setinde geçerli 24 saatlik pencere bulunamadı."
        )

    # Tahmin
    prediction = predict_pm25(timestamps, values)

    input_end = timestamps[-1]
    forecast_time = input_end + pd.Timedelta(hours=1)

    print("=" * 42)
    print("                AIRPULSE")
    print("=" * 42)

    print("Model           : LSTM")
    print("Station         : Aotizhongxin")
    print("Sequence Length : 24 hours")

    print()
    print("Input End Time  :", input_end)
    print("Forecast Time   :", forecast_time)

    print()
    print(f"Predicted PM2.5 : {prediction:.2f} µg/m³")

    print()
    print("Status          : Prediction completed")
    print("Usage           : Educational demo only")

    print("=" * 42)


if __name__ == "__main__":
    main()
