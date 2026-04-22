import os
import base64
import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:5000"  # Ganti jika backend ada di server lain

# =========================
# Helper: safe loaders
# =========================
def show_image_safe(possible_paths, caption):
    """
    Tampilkan gambar pertama yang ditemukan dari daftar path.
    Menghindari error 'file not found' dan kompatibel dengan Streamlit terbaru.
    """
    for p in possible_paths:
        if os.path.exists(p):
            st.image(p, caption=caption, use_container_width=True)
            return True
    st.warning(
        f"Tidak menemukan file gambar untuk: {caption}. "
        f"Coba simpan salah satu nama: {', '.join(possible_paths)} | CWD: {os.getcwd()}"
    )
    return False

def show_csv_safe(possible_paths, success_msg=None, empty_msg=None):
    """
    Tampilkan dataframe dari CSV pertama yang ditemukan.
    """
    for p in possible_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if success_msg:
                    st.caption(success_msg)
                st.dataframe(df, use_container_width=True)
                return True
            except Exception as e:
                st.warning(f"Gagal membaca CSV '{p}': {e}")
                return False
    if empty_msg:
        st.info(empty_msg)
    return False

# =========================
# UI: Logo & Deskripsi
# =========================
def add_logo():
    logo_path = "aplikasilogo.png"
    try:
        with open(logo_path, "rb") as logo_file:
            encoded_logo = base64.b64encode(logo_file.read()).decode()
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{encoded_logo}" style="max-width: 200px; margin-bottom: 10px;"/>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.warning("Logo tidak ditemukan (aplikasilogo.png).")

def add_description():
    st.markdown(
        """
        <h2 style="text-align:center; color:#0c4f6a;">SISTEM PREDIKSI PENJUALAN PRODUK KECANTIKAN</h2>
        <p style="text-align:justify;">
        Aplikasi ini menggunakan model LSTM untuk memprediksi penjualan produk kecantikan
        berdasarkan data historis 2020–2023 dan menampilkan proyeksi tahun 2024, 2025, dan 2026
        (bergantung endpoint backend yang tersedia).
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UI: Visualisasi Hasil Model (Gambar Saja)
# =========================
def visualisasi_hasil_model():
    st.header("Visualisasi Hasil Model")
    tab1, tab2, tab3 = st.tabs(
        ["Model Penjualan", "Prediksi 12 Bulan", "Prediksi vs Aktual (80:20, 70:30, 60:40)"]
    )

    

    # --- Tab 2: Prediksi 12 Bulan ke Depan (dengan sub-tabs per split) ---
    with tab2:
        st.subheader("Prediksi 12 Bulan ke Depan (Per Split)")
        t1, t2, t3 = st.tabs(["80:20", "70:30", "60:40"])

        # 80:20
        with t1:
            show_image_safe(
                ["prediksi_12_bulan_80_20.png", "images/prediksi_12_bulan_80_20.png", "assets/prediksi_12_bulan_80_20.png"],
                caption="Prediksi 12 Bulan - Split 80:20",
            )
            with st.expander("Tabel Prediksi (Opsional)"):
                show_csv_safe(
                    ["forecast_12bulan_split_80_20.csv", "data/forecast_12bulan_split_80_20.csv"],
                    success_msg="Tabel forecast 12 bulan (80:20):",
                    empty_msg="Tidak ada file forecast CSV (80:20) yang ditemukan.",
                )

        # 70:30
        with t2:
            show_image_safe(
                ["prediksi_12_bulan_70_30.png", "images/prediksi_12_bulan_70_30.png", "assets/prediksi_12_bulan_70_30.png"],
                caption="Prediksi 12 Bulan - Split 70:30",
            )
            with st.expander("Tabel Prediksi (Opsional)"):
                show_csv_safe(
                    ["forecast_12bulan_split_70_30.csv", "data/forecast_12bulan_split_70_30.csv"],
                    success_msg="Tabel forecast 12 bulan (70:30):",
                    empty_msg="Tidak ada file forecast CSV (70:30) yang ditemukan.",
                )

        # 60:40
        with t3:
            show_image_safe(
                ["prediksi_12_bulan_60_40.png", "images/prediksi_12_bulan_60_40.png", "assets/prediksi_12_bulan_60_40.png"],
                caption="Prediksi 12 Bulan - Split 60:40",
            )
            with st.expander("Tabel Prediksi (Opsional)"):
                show_csv_safe(
                    ["forecast_12bulan_split_60_40.csv", "data/forecast_12bulan_split_60_40.csv"],
                    success_msg="Tabel forecast 12 bulan (60:40):",
                    empty_msg="Tidak ada file forecast CSV (60:40) yang ditemukan.",
                )

    # --- Tab 3: Prediksi vs Aktual (3 Skema Split) ---
    with tab3:
        st.subheader("Prediksi vs Aktual (Perbandingan 3 Skema Split)")
        s1, s2, s3 = st.tabs(["80:20", "70:30", "60:40"])

        # 80:20
        with s1:
            show_image_safe(
                ["prediksi_vs_aktual_80_20.png", "images/prediksi_vs_aktual_80_20.png", "assets/prediksi_vs_aktual_80_20.png"],
                caption="Prediksi vs Aktual - Split 80:20"
            )
            with st.expander("Tabel (Opsional)"):
                show_csv_safe(
                    ["prediksi_vs_aktual_split_80_20.csv", "data/prediksi_vs_aktual_split_80_20.csv"],
                    success_msg="Tabel Prediksi vs Aktual (80:20):",
                    empty_msg="Tidak ada CSV untuk 80:20 yang ditemukan."
                )

        # 70:30
        with s2:
            show_image_safe(
                ["prediksi_vs_aktual_70_30.png", "images/prediksi_vs_aktual_70_30.png", "assets/prediksi_vs_aktual_70_30.png"],
                caption="Prediksi vs Aktual - Split 70:30"
            )
            with st.expander("Tabel (Opsional)"):
                show_csv_safe(
                    ["prediksi_vs_aktual_split_70_30.csv", "data/prediksi_vs_aktual_split_70_30.csv"],
                    success_msg="Tabel Prediksi vs Aktual (70:30):",
                    empty_msg="Tidak ada CSV untuk 70:30 yang ditemukan."
                )

        # 60:40
        with s3:
            show_image_safe(
                ["prediksi_vs_aktual_60_40.png", "images/prediksi_vs_aktual_60_40.png", "assets/prediksi_vs_aktual_60_40.png"],
                caption="Prediksi vs Aktual - Split 60:40"
            )
            with st.expander("Tabel (Opsional)"):
                show_csv_safe(
                    ["prediksi_vs_aktual_split_60_40.csv", "data/prediksi_vs_aktual_split_60_40.csv"],
                    success_msg="Tabel Prediksi vs Aktual (60:40):",
                    empty_msg="Tidak ada CSV untuk 60:40 yang ditemukan."
                )

# =========================
# UI: Prediksi Penjualan (CSV + Backend)
# =========================
def prediksi_penjualan():
    st.subheader("📈 Prediksi Penjualan Produk Kecantikan")
    uploaded_file = st.file_uploader("📤 Upload File CSV", type=["csv"])

    # Pilih tahun output yang ingin ditampilkan (default: 2024–2026)
    selected_years = st.multiselect(
        "🎯 Pilih tahun prediksi yang ingin ditampilkan",
        options=[2024, 2025, 2026],
        default=[2024, 2025, 2026],
        help="Aplikasi akan mencoba memanggil endpoint /prediksi-<tahun> atau fallback ke /prediksi dengan parameter 'year'."
    )

    if uploaded_file is None:
        return

    # Baca CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return

    # Validasi kolom
    required_cols = {"Tanggal", "Penjualan_Bersih"}
    if not required_cols.issubset(set(df.columns)):
        st.error("❌ File harus memiliki kolom: 'Tanggal' dan 'Penjualan_Bersih'")
        return

    # Siapkan payload request (standarisasi tanggal -> string ISO)
    try:
        df_req = df.copy()
        df_req["Tanggal"] = pd.to_datetime(df_req["Tanggal"]).astype(str)
    except Exception as e:
        st.error(f"Gagal memproses kolom Tanggal: {e}")
        return

    # Kumpulkan hasil per tahun
    hasil_per_tahun = {}
    pred_colnames = {}  # map tahun -> nama kolom prediksi asli di respons

    # Jalankan prediksi per tahun
    for year in selected_years:
        with st.spinner(f"Meminta prediksi {year}..."):
            try:
                result, used_ep = _call_backend_for_year(year, df_req)
            except Exception as e:
                st.error(f"🚫 {year}: {e}")
                continue

            if "error" in result:
                st.error(f"🚫 {year}: {result['error']}")
                continue

            if "prediksi" not in result:
                st.error(f"🚫 {year}: Respons backend tidak memiliki key 'prediksi'.")
                continue

            df_pred = pd.DataFrame(result["prediksi"])
            if df_pred.empty or "Tanggal" not in df_pred.columns:
                st.error(f"🚫 {year}: Format hasil prediksi tidak sesuai. Minimal harus ada kolom 'Tanggal' dan nilai prediksi.")
                continue

            # Deteksi nama kolom nilai prediksi
            pred_col = next((c for c in ["Prediksi", "prediksi", "yhat", "forecast", "Prediction"] if c in df_pred.columns), None)
            if pred_col is None:
                st.error(f"🚫 {year}: Tidak menemukan kolom nilai prediksi. Harapkan salah satu: Prediksi/prediksi/yhat/forecast/Prediction")
                continue

            # Cast tanggal
            try:
                df_pred["Tanggal"] = pd.to_datetime(df_pred["Tanggal"])
            except Exception as e:
                st.error(f"🚫 {year}: Gagal parse tanggal hasil prediksi: {e}")
                continue

            # Simpan & rename kolom prediksi agar unik per tahun
            pred_col_renamed = f"Prediksi_{year}"
            df_pred = df_pred[["Tanggal", pred_col]].rename(columns={pred_col: pred_col_renamed})
            hasil_per_tahun[year] = {
                "df": df_pred,
                "metrics": {k: result.get(k) for k in ["mape", "mae", "mse"]},
                "endpoint": used_ep,
            }
            pred_colnames[year] = pred_col_renamed

    if not hasil_per_tahun:
        st.info("Tidak ada hasil prediksi yang berhasil ditampilkan.")
        return

    # Gabungkan semua tahun berdasarkan tanggal (outer join)
    merged = None
    for year, payload in sorted(hasil_per_tahun.items()):
        df_pred = payload["df"]
        merged = df_pred if merged is None else pd.merge(merged, df_pred, on="Tanggal", how="outer")

    # Urutkan berdasarkan tanggal
    merged = merged.sort_values("Tanggal").reset_index(drop=True)

    # Tampilkan grafik gabungan
    st.success("✅ Prediksi berhasil dihimpun!")
    st.markdown("**Grafik Gabungan Prediksi per Tahun**")
    try:
        st.line_chart(merged.set_index("Tanggal")[[pred_colnames[y] for y in sorted(hasil_per_tahun.keys())]])
    except Exception as e:
        st.warning(f"Grafik gabungan tidak dapat ditampilkan: {e}")

    # Tampilkan tabel gabungan
    st.dataframe(merged, use_container_width=True)

    # Unduhan gabungan
    st.download_button(
        label="⬇️ Unduh Semua Hasil (Gabungan)",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_gabungan_2024_2026.csv",
        mime="text/csv"
    )

    # Ringkasan dan unduh per tahun
    st.markdown("---")
    st.markdown("### Ringkasan Per Tahun")
    for year, payload in sorted(hasil_per_tahun.items()):
        df_pred = payload["df"]
        metrics = payload["metrics"]
        used_ep = payload["endpoint"]

        with st.expander(f"📦 {year} • Endpoint: {used_ep}"):
            # Metrik (jika tersedia)
            cols = st.columns(3)
            # MAPE
            if metrics.get("mape") is not None:
                try:
                    cols[0].metric(f"MAPE {year}", f"{float(metrics['mape']):.2f}%")
                except Exception:
                    cols[0].metric(f"MAPE {year}", str(metrics["mape"]))
            # MAE
            if metrics.get("mae") is not None:
                try:
                    cols[1].metric(f"MAE {year}", f"{float(metrics['mae']):,.2f}")
                except Exception:
                    cols[1].metric(f"MAE {year}", str(metrics["mae"]))
            # MSE
            if metrics.get("mse") is not None:
                try:
                    cols[2].metric(f"MSE {year}", f"{float(metrics['mse']):,.2f}")
                except Exception:
                    cols[2].metric(f"MSE {year}", str(metrics["mse"]))

            # Chart & tabel per tahun
            try:
                st.line_chart(df_pred.set_index("Tanggal")[pred_colnames[year]])
            except Exception:
                pass
            st.dataframe(df_pred, use_container_width=True)

            # Unduh per tahun
            st.download_button(
                label=f"⬇️ Unduh Hasil {year}",
                data=df_pred.to_csv(index=False).encode("utf-8"),
                file_name=f"hasil_prediksi_{year}.csv",
                mime="text/csv",
                key=f"dl_{year}"
            )

def _call_backend_for_year(year: int, df_req: pd.DataFrame):
    """
    Coba dua strategi pemanggilan backend untuk fleksibilitas:
    1) POST /prediksi-<year>  dengan payload = list of records
    2) POST /prediksi         dengan payload = {"year": year, "data": list of records}

    Return: (result_json: dict, used_endpoint: str)
    Raise Exception jika semua strategi gagal.
    """
    records = df_req.to_dict(orient="records")
    strategies = [
        (f"{API_URL}/prediksi-{year}", records),
        (f"{API_URL}/prediksi", {"year": year, "data": records}),
    ]

    last_err = None
    for url, payload in strategies:
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code >= 400:
                last_err = f"{resp.status_code} {resp.reason}"
                continue
            try:
                data = resp.json()
            except Exception:
                last_err = f"Bad JSON from {url}: {resp.text[:300]}"
                continue
            if isinstance(data, dict) and "prediksi" in data:
                return data, url
            last_err = f"Unexpected JSON shape from {url}: keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"Gagal memanggil backend untuk tahun {year}. Terakhir: {last_err}")



# =========================
# MAIN APP
# =========================
def main():
    add_logo()
    # Debug opsional: lihat folder kerja
    # st.caption(f"CWD: {os.getcwd()}")

    menu = st.sidebar.selectbox("📋 Menu", ["Beranda", "Visualisasi Hasil Model", "Prediksi Penjualan"])

    if menu == "Beranda":
        add_description()
    elif menu == "Visualisasi Hasil Model":
        visualisasi_hasil_model()
    elif menu == "Prediksi Penjualan":
        prediksi_penjualan()

if __name__ == "__main__":
    main()
