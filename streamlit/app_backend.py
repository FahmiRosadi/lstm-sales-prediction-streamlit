from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import os
from datetime import datetime

app = Flask(__name__)

# ====== Paths & Artifacts ======
MODEL_PATH = "model_lstm_best.h5"
SCALER_PATH = "scaler.pkl"

# Load model & scaler (sekali saat start)
model = load_model(MODEL_PATH, compile=False)
scaler = load(SCALER_PATH)

# ====== Konfigurasi umum ======
WINDOW_SIZE = 6                   # panjang jendela input ke LSTM
TARGET_STEPS = 12                 # jumlah bulan yang diproyeksikan (Jan–Des)

# ====== Util ======
def _ensure_df(records):
    """Ubah list-of-dicts atau dict menjadi DataFrame, validasi kolom pokok."""
    df = pd.DataFrame(records)
    if "Tanggal" not in df.columns or "Penjualan_Bersih" not in df.columns:
        raise ValueError("Payload wajib memuat kolom 'Tanggal' dan 'Penjualan_Bersih'.")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)
    return df

def _compute_metrics(df_actual_pred):
    """Hitung metrik jika tersedia kolom aktual & prediksi."""
    if df_actual_pred.empty:
        return None, None, None
    y_true = df_actual_pred["Penjualan_Bersih"].values
    y_pred = df_actual_pred["Prediksi"].values
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)  # 0..1
    except Exception:
        mape = None
    try:
        mae = mean_absolute_error(y_true, y_pred)
    except Exception:
        mae = None
    try:
        mse = mean_squared_error(y_true, y_pred)
    except Exception:
        mse = None
    return mape, mae, mse

def _scale_month_value(month_int):
    """
    Skala 'Bulan' menggunakan scaler yang sama dengan training.
    Kita buat dummy row [Penjualan_Bersih=0, Bulan=month, Lag_1=0], lalu transform.
    Ambil kolom index 1 (Bulan) dari hasil transformasi.
    """
    dummy = np.array([[0.0, float(month_int), 0.0]])
    scaled = scaler.transform(dummy)
    return scaled[0, 1]  # kolom 'Bulan' terskala

def _scaled_to_unscaled_target(y_scaled, ref_scaled_features):
    """
    Invers target (Penjualan_Bersih) dari nilai scaled (kolom 0).
    ref_scaled_features: array, minimal 2 nilai untuk melengkapi inverse_transform:
        [Bulan_scaled, Lag_1_scaled]
    """
    tmp = np.zeros((1, 3))
    tmp[0, 0] = y_scaled
    tmp[0, 1:] = ref_scaled_features  # bulan_scaled, lag1_scaled
    # inverse_transform mengembalikan [Penjualan_Bersih, Bulan, Lag_1] dalam domain asli
    y_inverse = scaler.inverse_transform(tmp)[0, 0]
    return float(max(0.0, y_inverse))

def _create_prediction_sequence(last_window_scaled, start_month=1, steps=TARGET_STEPS):
    """
    Auto-regressive forecasting:
    - last_window_scaled: np.ndarray shape (WINDOW_SIZE, 2) berisi [Bulan_scaled, Lag1_scaled]
    - start_month: bulan pertama tahun target (1..12)
    - steps: jumlah langkah (default 12)
    NOTE: Untuk menyederhanakan asumsi, kita gunakan y_scaled (kolom 0) sebagai proxy Lag_1_scaled berikutnya.
    """
    preds = []
    # Kita juga butuh bulan berjalan (real month int) agar dapat di-scale tiap langkah
    current_month = int(start_month)

    window = last_window_scaled.copy()

    for _ in range(steps):
        # Model input (1, WINDOW_SIZE, 2)
        X_input = window.reshape(1, WINDOW_SIZE, 2)
        y_scaled = model.predict(X_input, verbose=0)[0, 0]

        # Invers ke domain asli (Rp) menggunakan Bulan & Lag_1 yang terskala dari elemen terakhir window
        ref_features = window[-1, :]  # [Bulan_scaled, Lag1_scaled]
        y_unscaled = _scaled_to_unscaled_target(y_scaled, ref_features)
        preds.append(y_unscaled)

        # Siapkan next input:
        #  - Bulan_scaled untuk bulan berikutnya
        next_month = (current_month % 12) + 1
        bulan_scaled_next = _scale_month_value(next_month)

        #  - Lag_1_scaled: secara ketat, ini seharusnya skala kolom Lag_1.
        #    Kita sederhanakan dengan memakai y_scaled (kolom target terskala).
        lag1_scaled_next = y_scaled

        next_pair = np.array([bulan_scaled_next, lag1_scaled_next], dtype="float32")
        window = np.vstack([window[1:], next_pair])

        current_month = next_month

    return preds

def _prepare_last_window_scaled(df_train):
    """
    Dari df_train (<= tahun target-1), bentuk deret fitur dan ambil WINDOW_SIZE terakhir
    untuk [Bulan_scaled, Lag_1_scaled].
    """
    df_feat = df_train.copy()
    df_feat["Bulan"] = df_feat["Tanggal"].dt.month
    df_feat["Lag_1"] = df_feat["Penjualan_Bersih"].shift(1)
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Urutan fitur saat fit scaler (harus konsisten dengan training): 
    # ['Penjualan_Bersih', 'Bulan', 'Lag_1']
    features = ["Penjualan_Bersih", "Bulan", "Lag_1"]
    scaled_all = scaler.transform(df_feat[features])

    # Ambil WINDOW_SIZE baris terakhir kolom 1: (Bulan_scaled, Lag_1_scaled)
    last_window = scaled_all[-WINDOW_SIZE:, 1:]
    return last_window

def _predict_year_core(df_raw, target_year):
    """
    Inti prediksi untuk tahun tertentu.
    - df_raw: DataFrame berisi 'Tanggal' & 'Penjualan_Bersih' (2020..terbaru)
    - target_year: int, misalnya 2024, 2025, 2026
    Return dict: {prediksi, mape, mae, mse}
    """
    if not isinstance(target_year, int):
        raise ValueError("Parameter year harus integer, misal 2024.")

    # Sort & split train/val
    df = df_raw.sort_values("Tanggal").reset_index(drop=True)

    # Train memakai s/d 31 Des tahun sebelumnya
    train_mask = df["Tanggal"].dt.year <= (target_year - 1)
    df_train = df[train_mask].copy()
    if df_train.empty or len(df_train) < (WINDOW_SIZE + 1):
        raise ValueError(f"Data training tidak cukup hingga {target_year-1}. Minimal {WINDOW_SIZE+1} baris.")

    # Siapkan jendela terakhir dan lakukan forecasting 12 bulan Jan–Des tahun target
    last_window = _prepare_last_window_scaled(df_train)
    preds = _create_prediction_sequence(last_window, start_month=1, steps=TARGET_STEPS)

    future_dates = pd.date_range(f"{target_year}-01-01", periods=TARGET_STEPS, freq="MS")
    df_pred = pd.DataFrame({"Tanggal": future_dates, "Prediksi": preds})

    # Jika ada aktual tahun target di payload, hitung metrik
    df_val = df[df["Tanggal"].dt.year == target_year][["Tanggal", "Penjualan_Bersih"]].copy()
    if not df_val.empty:
        df_merge = pd.merge(df_val, df_pred, on="Tanggal", how="inner")
        mape, mae, mse = _compute_metrics(df_merge)
    else:
        mape = mae = mse = None

    # Format respons
    return {
        "prediksi": df_pred.to_dict(orient="records"),
        "mape": f"{mape:.2%}" if mape is not None else None,
        "mae": mae,
        "mse": mse,
        "year": target_year
    }

# ====== Endpoints ======

@app.route("/prediksi", methods=["POST"])
def prediksi_generic():
    """
    Endpoint generik:
    Payload bisa berupa:
      1) {"year": 2025, "data": [ { "Tanggal": "...", "Penjualan_Bersih": ... }, ... ]}
      2) [ { "Tanggal": "...", "Penjualan_Bersih": ... }, ... ]  -> default year=2024
    """
    try:
        payload = request.get_json()
        if isinstance(payload, dict) and "data" in payload:
            year = int(payload.get("year", 2024))
            df = _ensure_df(payload["data"])
        else:
            # langsung assume list-of-dicts dengan default year=2024
            year = 2024
            df = _ensure_df(payload)

        result = _predict_year_core(df, year)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/prediksi-2024", methods=["POST"])
def prediksi_2024():
    """Kompatibel dengan versi lama: kirim list-of-dicts, output 2024."""
    try:
        data = request.get_json()
        df = _ensure_df(data)
        result = _predict_year_core(df, 2024)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/prediksi-2025", methods=["POST"])
def prediksi_2025():
    """List-of-dicts => prediksi 2025."""
    try:
        data = request.get_json()
        df = _ensure_df(data)
        result = _predict_year_core(df, 2025)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/prediksi-2026", methods=["POST"])
def prediksi_2026():
    """List-of-dicts => prediksi 2026."""
    try:
        data = request.get_json()
        df = _ensure_df(data)
        result = _predict_year_core(df, 2026)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ====== Main ======
if __name__ == "__main__":
    # Bisa diubah ke host="0.0.0.0" bila ingin diakses dari luar container
    app.run(debug=True)
