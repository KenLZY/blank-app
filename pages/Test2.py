import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Header
st.title("Farm Product Analysis and Forecasting")

file_path = "Filled_Telur_Data.xlsx"

try:
    # Load data
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['HARI/TANGGAL'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date')
    df['JENIS'] = df['JENIS'].str.lower().str.strip()
    df['QTY (KG)'] = pd.to_numeric(df['QTY (KG)'], errors='coerce').fillna(0)

    # Filter by product type
    unique_names = df['JENIS'].unique()
    selected_product = st.selectbox("Select a product type:", unique_names)
    selected_df = df[df['JENIS'] == selected_product]

    if selected_df.empty:
        st.error("No data found for the selected product.")
        st.stop()

    # Handle missing or zero values
    if selected_df['QTY (KG)'].sum() == 0:
        st.error("The 'QTY (KG)' column contains only zeros or invalid values. Unable to forecast.")
        st.stop()

    # Plot historical data
    st.subheader(f"Historical Data for {selected_product}")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_df.index, selected_df['QTY (KG)'], marker='o', linestyle='-')
    plt.title(f"QTY (KG) Over Time for {selected_product}")
    plt.xlabel("Date")
    plt.ylabel("Quantity (KG)")
    plt.grid(True)
    st.pyplot(plt)

    # Ensure stationarity
    def enforce_stationarity(series):
        for i in range(3):  # Allow up to 3 rounds of differencing
            adf_test = adfuller(series)
            if adf_test[1] <= 0.05:  # Stationary if p-value <= 0.05
                return series
            series = series.diff().dropna()
        return series

    qty_series = enforce_stationarity(selected_df['QTY (KG)'])

    # Forecasting
    forecast_periods = st.slider("Select forecast periods (months):", 1, 24, 12)
    st.write("Automatically selecting ARIMA parameters...")

    try:
        # Fit Auto ARIMA
        auto_model = auto_arima(qty_series, seasonal=False, stepwise=True, suppress_warnings=True)
        st.write(f"Best ARIMA order: {auto_model.order}")

        # Fit ARIMA model
        model = ARIMA(qty_series, order=auto_model.order)
        model_fit = model.fit()

        # Generate Forecast
        forecast = model_fit.forecast(steps=forecast_periods)
        forecast_index = pd.date_range(selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:]
        forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

        # Display forecasted values
        st.write("Forecasted Values:")
        st.dataframe(forecast_df)

        # Plot forecast
        plt.figure(figsize=(10, 6))
        plt.plot(selected_df.index, selected_df['QTY (KG)'], label="Historical Data", marker='o')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color='orange', linestyle='--')
        plt.title(f"ARIMA Forecast for {selected_product}")
        plt.xlabel("Date")
        plt.ylabel("Quantity (KG)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"ARIMA model failed: {e}")

except Exception as e:
    st.error(f"An error occurred: {e}")
