from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os

app = Flask(__name__)

# Paths
MODEL_PATH = "model_lstm_best.h5"
SCALER_PATH = "scaler.pkl"

# Load model dan scaler
model = load_model(MODEL_PATH)
scaler = load(SCALER_PATH)

# Fungsi bantu
WINDOW_SIZE = 6

def create_prediction_sequence(last_window):
    predictions = []
    for _ in range(12):
        X_input = last_window.reshape(1, WINDOW_SIZE, 2)
        y_scaled = model.predict(X_input, verbose=0)

        dummy = np.zeros((1, 3))
        dummy[0, 0] = y_scaled[0, 0]
        dummy[0, 1:] = last_window[-1]
        y_inverse = max(0, scaler.inverse_transform(dummy)[0, 0])
        predictions.append(y_inverse)

        next_month = (last_window[-1, 0] % 12) + 1
        next_input = np.array([next_month, y_scaled[0, 0]])
        last_window = np.vstack([last_window[1:], next_input])
    return predictions

@app.route("/prediksi-2024", methods=["POST"])
def prediksi_2024():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values("Tanggal").reset_index(drop=True)

        df_train = df[df['Tanggal'].dt.year <= 2023].copy()
        df_val = df[df['Tanggal'].dt.year == 2024].copy()

        df_train['Bulan'] = df_train['Tanggal'].dt.month
        df_train['Lag_1'] = df_train['Penjualan_Bersih'].shift(1)
        df_train.dropna(inplace=True)

        features = ['Penjualan_Bersih', 'Bulan', 'Lag_1']
        scaled = scaler.transform(df_train[features])
        last_window = scaled[-WINDOW_SIZE:, 1:]  # Ambil Bulan & Lag_1

        future_preds = create_prediction_sequence(last_window)

        future_dates = pd.date_range("2024-01-01", periods=12, freq='MS')
        df_pred = pd.DataFrame({
            "Tanggal": future_dates,
            "Prediksi": future_preds
        })

        if not df_val.empty:
            df_val = df_val[['Tanggal', 'Penjualan_Bersih']]
            df_combined = pd.merge(df_val, df_pred, on='Tanggal', how='inner')
            mape = mean_absolute_percentage_error(df_combined['Penjualan_Bersih'], df_combined['Prediksi'])
            mae = mean_absolute_error(df_combined['Penjualan_Bersih'], df_combined['Prediksi'])
        else:
            mape = None
            mae = None

        return jsonify({
            "prediksi": df_pred.to_dict(orient="records"),
            "mape": f"{mape:.2%}" if mape else None,
            "mae": mae
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
