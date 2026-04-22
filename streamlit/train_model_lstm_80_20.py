#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error,  mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns


# # Load File

# In[2]:


folder = r'C:\Users\User\Documents\Bismillah'
tahun_data = [2020, 2021, 2022, 2023, 2024]
list_df = []

for tahun in tahun_data:
    path = os.path.join(folder, f'{tahun} Summary.xlsx')
    df = pd.read_excel(path)
    df['Tahun'] = tahun
    list_df.append(df)

df_all = pd.concat(list_df, ignore_index=True)


# # EDA

# In[3]:


print("--- Dataset Info ---")
print(df_all.info(), '\n')

print("--- Missing Values per Column ---")
print(df_all.isnull().sum(), '\n')

print("--- Duplicate Rows ---")
print(df_all.duplicated().sum(), '\n')


if df_all.duplicated().any():
    df_all = df_all.drop_duplicates()
    print("Duplicates dropped, new shape:", df_all.shape)


numeric_cols = df_all.select_dtypes(include=['float64', 'int64'])
print("--- Correlation Matrix ---")
print(numeric_cols.corr(), '\n')


# In[4]:


plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Variabel Numerik")
plt.tight_layout()
plt.show()


# # Konversi Data Penjualan Ke Format Time Series (Long Format)

# In[5]:


bulan_nama = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Agt','Sep','Okt','Nov','Des']
records = []
for _, row in df_all.iterrows():
    for i, b in enumerate(bulan_nama, start=1):
        # Cek variasi nama kolom dengan atau tanpa spasi
        col1 = f'{b}Net Sales'
        col2 = f'{b} Net Sales'
        if col1 in df_all.columns:
            col = col1
        elif col2 in df_all.columns:
            col = col2
        else:
            col = None

        if col is not None:
            val = row.get(col, np.nan)
            if pd.notna(val):
                try:
                    clean_val = float(str(val).replace(',','').strip())
                    records.append({'Tahun': row['Tahun'], 'Bulan': i, 'Penjualan_Bersih': clean_val})
                except:
                    pass

df_ts = pd.DataFrame(records)
df_ts['Tanggal'] = pd.to_datetime(df_ts['Tahun'].astype(str) + '-' + df_ts['Bulan'].astype(str) + '-01')

df_monthly = df_ts.groupby('Tanggal')['Penjualan_Bersih'].sum().reset_index()

print("Missing in time series:", df_monthly.isnull().sum(), '\n')
print("Duplicates in TS:", df_monthly.duplicated().sum(), '\n')
print(df_monthly[['Tanggal', 'Penjualan_Bersih']].head(10))  # tampilkan 10 baris pertama


df_monthly['Bulan'] = df_monthly['Tanggal'].dt.month
df_monthly['Lag_1'] = df_monthly['Penjualan_Bersih'].shift(1)
df_monthly = df_monthly.dropna().reset_index(drop=True)


# In[6]:


df_monthly_sorted = df_monthly.sort_values('Tanggal').reset_index(drop=True)

train_size_80 = int(len(df_monthly_sorted) * 0.8)
train_80 = df_monthly_sorted.iloc[:train_size_80]
test_20 = df_monthly_sorted.iloc[train_size_80:]

print("Pembagian 80:20 berdasarkan urutan waktu:")
print("Jumlah data train:", len(train_80))
print("Jumlah data test:", len(test_20))


# # Preprocessing

# In[7]:


# Cek missing dan duplikasi
print("Missing in time series:", df_monthly.isnull().sum(), '\n')
print("Duplicates in TS:", df_monthly.duplicated().sum(), '\n')

# Buat fitur tambahan
df_monthly['Bulan'] = df_monthly['Tanggal'].dt.month
df_monthly['Lag_1'] = df_monthly['Penjualan_Bersih'].shift(1)
df_monthly = df_monthly.dropna().reset_index(drop=True)

# Tentukan fitur yang akan digunakan
features = ['Penjualan_Bersih', 'Bulan', 'Lag_1']

# Urutkan berdasarkan tanggal
df_monthly_sorted = df_monthly.sort_values('Tanggal').reset_index(drop=True)

# Bagi data 80:20 berdasarkan baris
split_index = int(len(df_monthly_sorted) * 0.8)
train_df = df_monthly_sorted.iloc[:split_index]
val_df   = df_monthly_sorted.iloc[split_index:]

# Scaling menggunakan MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df[features])  # Fit hanya pada data latih

train_80_scaled = scaler.transform(train_df[features])
test_20_scaled  = scaler.transform(val_df[features])

print("Preprocessing 80:20 selesai:")
print("Train shape:", train_80_scaled.shape)
print("Val shape  :", test_20_scaled.shape)


# In[8]:


# --- Fungsi membuat dataset time steps ---
def create_dataset(dataset, time_steps=12):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps, :])  # semua fitur
        y.append(dataset[i+time_steps, 0])    # target: Penjualan_Bersih (kolom ke-0)
    return np.array(X), np.array(y)

# Parameter time steps
time_steps = 12


# In[9]:


# Gunakan data hasil scaling dari pembagian 80:20
X_train_80, y_train_80 = create_dataset(train_80_scaled, time_steps)
X_test_20, y_test_20 = create_dataset(test_20_scaled, time_steps)

print("X_train_80 shape:", X_train_80.shape)
print("X_test_20 shape :", X_test_20.shape)

# Reshape jika hanya 2 dimensi (untuk LSTM butuh 3 dimensi)
if len(X_train_80.shape) == 2:
    X_train_80 = X_train_80.reshape((X_train_80.shape[0], X_train_80.shape[1], 1))
if len(X_test_20.shape) == 2:
    X_test_20 = X_test_20.reshape((X_test_20.shape[0], X_test_20.shape[1], 1))

# --- Fungsi membuat sequence sliding window (opsional) ---
def create_sequences(data, window=6):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, 1:])  # semua fitur kecuali target
        y.append(data[i + window, 0])   # target tetap Penjualan_Bersih
    return np.array(X), np.array(y)

# Buat sequence untuk train dan val (berdasarkan 80:20)
X_train_seq, y_train_seq = create_sequences(train_80_scaled, window=6)

# Untuk validasi: ambil 6 data terakhir dari train + test_20
X_val_seq, y_val_seq = create_sequences(np.vstack([train_80_scaled[-6:], test_20_scaled]), window=6)

print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_val_seq shape  :", X_val_seq.shape)
print("y_val_seq shape  :", y_val_seq.shape)

# Input shape untuk LSTM
input_shape_80 = (X_train_seq.shape[1], X_train_seq.shape[2])
print("Input shape untuk LSTM (80:20):", input_shape_80)


# # LSTM

# In[10]:


# Bangun model stacked LSTM untuk data 80:20
model = Sequential([
    Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# In[11]:


# Gunakan early stopping untuk mencegah overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=16,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stop],
    verbose=1
)
# Catat performa akhir
val_loss = history.history['val_loss'][-1]
print(f"Final Validation Loss (MSE): {val_loss:.4f}")


# In[12]:


plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# In[13]:


# Prediksi pada data validasi (test_20)
y_pred_20 = model.predict(X_val_seq)

# Buat dummy array agar bisa inverse_transform
# Harus sama jumlah kolomnya dengan saat scaler digunakan
dummy_val = np.zeros((len(y_val_seq), train_80_scaled.shape[1]))
dummy_pred = np.zeros((len(y_pred_20), train_80_scaled.shape[1]))

# Masukkan nilai aktual dan prediksi ke kolom target (kolom ke-0)
dummy_val[:, 0] = y_val_seq
dummy_pred[:, 0] = y_pred_20.flatten()

# Lakukan inverse transform
y_val_inv = scaler.inverse_transform(dummy_val)[:, 0]
y_pred_inv = scaler.inverse_transform(dummy_pred)[:, 0]

# Cetak hasil
print(" y_val (real):", y_val_inv[:5])
print(" y_pred (real):", y_pred_inv[:5])


# # Evaluasi

# In[14]:


# Hitung MSE
mse = mean_squared_error(y_val_inv, y_pred_inv)

# Hitung MAE
mae = mean_absolute_error(y_val_inv, y_pred_inv)

# Hitung MAPE
mape = mean_absolute_percentage_error(y_val_inv, y_pred_inv) * 100  # dalam persen

# Tampilkan hasil
print("Evaluasi Model (80:20) dalam Skala Asli:")
print(f"Mean Squared Error (MSE)     : {mse:.2f}")
print(f"Mean Absolute Error (MAE)    : {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[15]:


plt.figure(figsize=(12, 5))
plt.plot(y_val_inv, label='Aktual', marker='o')
plt.plot(y_pred_inv, label='Prediksi', marker='x')
plt.title('Perbandingan Aktual vs Prediksi (Validasi 80:20)')
plt.xlabel('Waktu')
plt.ylabel('Penjualan Bersih')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[16]:


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # validasi Testing

# In[17]:


# === Fungsi untuk mengembalikan skala asli dari prediksi ===
def inv_transform_80(y_preds, X_prev):
    if X_prev.ndim == 3:
        last_feats = X_prev[:, -1, :]
    else:
        last_feats = X_prev
    arr = np.hstack([y_preds.reshape(-1, 1), last_feats])
    return scaler.inverse_transform(arr)[:, 0]

# === Prediksi pada data validasi (test 20%) ===
X_val_full_80 = np.vstack([train_80_scaled[-6:], test_20_scaled])
X_val_window_feats_80 = np.array([
    X_val_full_80[i + 6 - 1, 1:]
    for i in range(len(test_20_scaled))
])

# Prediksi model
desired_steps = 3
batch_size_pred = max(1, len(X_val_seq) // desired_steps)
preds_val_80 = model.predict(X_val_seq, batch_size=batch_size_pred, verbose=1)

# Inverse transform
y_val_actual_80 = inv_transform_80(y_val_seq, X_val_window_feats_80)
y_val_pred_80   = inv_transform_80(preds_val_80.flatten(), X_val_window_feats_80)

# Tampilkan hasil
print("=== Hasil Prediksi Validasi 80:20 ===")
for i in range(len(y_val_actual_80)):
    print(f"Bulan ke-{i+1}: Aktual = {y_val_actual_80[i]:.2f}, Prediksi = {y_val_pred_80[i]:.2f}")

# === Prediksi 12 Bulan ke Depan Berdasarkan 80:20 ===
data_80_scaled_full = np.vstack([train_80_scaled, test_20_scaled])
last_window_80 = data_80_scaled_full[-6:, 1:]  # hanya ambil fitur Bulan & Lag_1

future_preds_80 = []

for i in range(12):
    X_input = last_window_80.reshape(1, 6, 2)  # (1, window, 2 fitur: Bulan & Lag_1)
    y_pred_scaled = model.predict(X_input)

    # Inverse ke skala asli
    y_pred_actual = inv_transform_80(y_pred_scaled.flatten(), last_window_80[-1, :].reshape(1, -1))[0]
    future_preds_80.append(y_pred_actual)

    # Geser window
    next_month = ((last_window_80[-1, 0] % 12) + 1)
    next_input = np.array([[next_month, y_pred_scaled[0][0]]])
    last_window_80 = np.vstack([last_window_80[1:], next_input])


# In[18]:


# Buat list nama bulan (jika diperlukan)
bulan_labels = [f"Bulan ke-{i}" for i in range(1, 13)]

plt.figure(figsize=(10, 5))
plt.plot(future_preds_80, marker='o', linestyle='-', label='Prediksi')
plt.xticks(ticks=range(12), labels=bulan_labels, rotation=45)
plt.title("Prediksi 12 Bulan ke Depan (Model LSTM - 80:20)")
plt.xlabel("Bulan")
plt.ylabel("Penjualan Bersih (dalam skala asli)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


# Buat dummy data untuk kebutuhan template kode ini (gantilah dengan variabel aslimu)
y_val_actual = np.array([100, 110, 120, 130, 125, 140])
y_val_pred = np.array([98, 112, 119, 128, 127, 138])


# In[20]:


# Residual error
residuals = y_val_actual - y_val_pred

# Scatter plot: aktual vs prediksi
plt.figure(figsize=(6, 6))
plt.scatter(y_val_actual, y_val_pred, alpha=0.7)
plt.plot([min(y_val_actual), max(y_val_actual)], [min(y_val_actual), max(y_val_actual)], 'r--')
plt.xlabel("Aktual")
plt.ylabel("Prediksi")
plt.title("Scatter Plot: Aktual vs Prediksi")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[21]:


# Residual plot
plt.figure(figsize=(10, 4))
plt.plot(residuals, marker='o', linestyle='-')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Error (Aktual - Prediksi)")
plt.xlabel("Index")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()


# In[22]:


# Distribution of residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=10, edgecolor='k', alpha=0.7)
plt.title("Distribusi Residual Error")
plt.xlabel("Residual")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()


# In[23]:


# Contoh data dummy (ganti dengan y_val_actual_80 dan y_val_pred_80 jika sudah tersedia di sesi kamu)
months = [f"Bulan {i+1}" for i in range(12)]
y_val_actual_80 = [210, 215, 220, 230, 240, 250, 255, 260, 270, 280, 285, 290]  # contoh
y_val_pred_80 =   [208, 218, 222, 228, 238, 248, 252, 259, 265, 278, 282, 288]  # contoh

# Plot garis tren aktual vs prediksi
plt.figure(figsize=(10, 5))
plt.plot(months, y_val_actual_80, marker='o', label='Aktual')
plt.plot(months, y_val_pred_80, marker='s', label='Prediksi')
plt.title("Tren Prediksi vs Aktual per Bulan (80:20)")
plt.xlabel("Bulan Validasi")
plt.ylabel("Penjualan Bersih")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# === SIMPAN MODEL DAN SCALER UNTUK STREAMLIT ===
model.save("model_lstm_best.h5")
import joblib
joblib.dump(scaler, "scaler.pkl")
print("Model dan Scaler berhasil disimpan.")
