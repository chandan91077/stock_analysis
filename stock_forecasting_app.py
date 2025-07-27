import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# =========================
# 🎯 Helper Functions
# =========================

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    if 'Ticker' in df.columns and df['Ticker'].nunique() == 1:
        df.drop(columns=['Ticker'], inplace=True)
    return df

def plot_overview(df):
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    st.line_chart(df[['Close', 'MA_20', 'MA_50']])
    with st.expander("📦 Volume Chart"):
        st.bar_chart(df['Volume'])

def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse

# =========================
# 🔮 Forecast Models
# =========================

def forecast_arima(df):
    train = df['Close'][:-30]
    test = df['Close'][-30:]
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return test, forecast

def forecast_sarima(df):
    train = df['Close'][:-30]
    test = df['Close'][-30:]
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return test, forecast

def forecast_prophet(df):
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    train = df_prophet[:-30]
    test = df_prophet[-30:]
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    pred = forecast.set_index('ds')['yhat'][-30:]
    return test.set_index('ds')['y'], pred

def forecast_lstm(df):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_sequences(data, step=60):
        X, y = [], []
        for i in range(step, len(data)):
            X.append(data[i-step:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_all, y_all = create_sequences(data_scaled)
    X_train, y_train = X_all[:-30], y_all[:-30]
    X_test, y_test = X_all[-30:], y_all[-30:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    idx = df.index[-30:]
    return pd.Series(y_test_rescaled.flatten(), index=idx), pd.Series(y_pred_rescaled.flatten(), index=idx)

# =========================
# 🌐 Streamlit Interface
# =========================

st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📊 Stock Forecasting using ARIMA, SARIMA, Prophet, and LSTM")
st.markdown("Upload your stock CSV and compare time series models for future stock price prediction.")

# === Sidebar ===
st.sidebar.header("📂 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Open, High, Low, Close, Volume)", type="csv")

st.sidebar.markdown("### 📌 Select Models to Run")
selected_models = []
if st.sidebar.checkbox("ARIMA", value=True):
    selected_models.append("ARIMA")
if st.sidebar.checkbox("SARIMA", value=True):
    selected_models.append("SARIMA")
if st.sidebar.checkbox("Prophet", value=True):
    selected_models.append("Prophet")
if st.sidebar.checkbox("LSTM", value=True):
    selected_models.append("LSTM")

run_forecast = st.sidebar.button("🚀 Run Forecast")

# === App Body ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    st.success("✅ Data loaded successfully!")

    with st.expander("🧾 Preview Data"):
        st.dataframe(df.tail())

    with st.expander("📊 Stock Trends & Volume"):
        plot_overview(df)

    if run_forecast:
        if not selected_models:
            st.warning("⚠️ Please select at least one model.")
        else:
            forecast_results = {}
            actual_values = {}
            completed_steps = 0
            total_steps = len(selected_models)
            progress = st.progress(0)

            if "ARIMA" in selected_models:
                with st.spinner("Running ARIMA..."):
                    actual, pred = forecast_arima(df)
                    forecast_results["ARIMA"] = pred
                    actual_values["ARIMA"] = actual
                completed_steps += 1
                progress.progress(completed_steps / total_steps)

            if "SARIMA" in selected_models:
                with st.spinner("Running SARIMA..."):
                    actual, pred = forecast_sarima(df)
                    forecast_results["SARIMA"] = pred
                    actual_values["SARIMA"] = actual
                completed_steps += 1
                progress.progress(completed_steps / total_steps)

            if "Prophet" in selected_models:
                with st.spinner("Running Prophet..."):
                    actual, pred = forecast_prophet(df)
                    forecast_results["Prophet"] = pred
                    actual_values["Prophet"] = actual
                completed_steps += 1
                progress.progress(completed_steps / total_steps)

            if "LSTM" in selected_models:
                with st.spinner("Running LSTM..."):
                    actual, pred = forecast_lstm(df)
                    forecast_results["LSTM"] = pred
                    actual_values["LSTM"] = actual
                completed_steps += 1
                progress.progress(completed_steps / total_steps)

            # === Forecast Plot ===
            st.subheader("📈 Forecast Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual.index, y=actual, name="Actual", line=dict(color="black")))
            for model in selected_models:
                fig.add_trace(go.Scatter(x=forecast_results[model].index, y=forecast_results[model], name=model))
            st.plotly_chart(fig, use_container_width=True)

            # === Evaluation ===
            st.subheader("📋 Evaluation Metrics")
            metrics = {
                model: compute_metrics(actual_values[model], forecast_results[model])
                for model in selected_models
            }
            metrics_df = pd.DataFrame(metrics, index=["MAE", "RMSE"]).T
            st.dataframe(metrics_df.style.format(precision=2))

            # Download option
            csv = metrics_df.to_csv().encode("utf-8")
            st.download_button("⬇️ Download Metrics as CSV", csv, "forecast_metrics.csv", "text/csv")

else:
    st.info("👈 Upload your CSV from the sidebar to begin.")
