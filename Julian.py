import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from datetime import datetime, timedelta, date

st.set_page_config(page_title="📈 ARIMA Stock Forecasting", layout="centered")

st.title("📊 Stock Price Forecasting with ARIMA")

st.markdown("Masukkan ticker saham dan pilih tanggal. Kamu juga bisa sesuaikan parameter ARIMA dan periode forecasting.")

# SECTION: Input
ticker = st.text_input("📌 Masukkan Ticker Saham (Contoh: AAPL, GOTO.JK)", value="AAPL")
today = date.today()
start_date = st.date_input("📅 Tanggal Mulai", value=datetime(2022, 1, 1))
end_date = st.date_input("📅 Tanggal Selesai", value=today, max_value=today)

# SECTION: Parameter ARIMA
with st.expander("⚙ Parameter ARIMA (opsional)"):
    p = st.number_input("AR (p)", min_value=0, max_value=5, value=2)
    d = st.number_input("I (d)", min_value=0, max_value=2, value=1)
    q = st.number_input("MA (q)", min_value=0, max_value=5, value=2)

forecast_months = st.slider("📆 Berapa bulan ke depan ingin diprediksi?", 1, 24, 6)

# SECTION: Run Forecasting
if st.button("🔍 Forecast"):
    if ticker and start_date < end_date:
        with st.spinner("Mengambil data saham..."):
            try:
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    st.error(f"⚠ Data untuk {ticker} tidak ditemukan dalam rentang waktu tersebut.")
                else:
                    st.success(f"Data berhasil diambil. Menampilkan harga penutupan...")
                    st.line_chart(df["Close"])

                    # Fit ARIMA
                    with st.spinner("Melatih model ARIMA..."):
                        model = ARIMA(df['Close'], order=(p, d, q))
                        model_fit = model.fit()

                        forecast = model_fit.forecast(steps=forecast_months)

                        # Generate dates for forecast
                        last_date = df.index[-1]
                        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_months, freq='M')
                        forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

                        # Merge original and forecast
                        combined = pd.concat([df['Close'], forecast_df['Forecast']])

                        st.subheader("📉 Hasil Forecasting")
                        st.line_chart(combined)

                        st.write("📋 Tabel Forecast:")
                        st.dataframe(forecast_df)

                        st.download_button("⬇ Download Forecast", forecast_df.to_csv(), "forecast.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("❗ Pastikan tanggal valid dan ticker tidak kosong.")