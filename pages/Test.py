import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from hijri_converter import convert
import numpy as np

# Header
st.title("Farm Product Analysis and Forecasting with Holidays")

file_path = "Corrected_Telur_Data.xlsx"  # Ensure this file exists in the correct path

try:
    # Load Excel file
    df = pd.read_excel(file_path)

    # Data Cleaning
    df['Date'] = pd.to_datetime(df['HARI/TANGGAL'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date')  # Drop invalid dates
    df['JENIS'] = df['JENIS'].str.lower().str.strip()
    df['QTY (KG)'] = pd.to_numeric(df['QTY (KG)'], errors='coerce').fillna(0)

    # Ensure enough data points
    if len(df) < 30:
        st.error("Insufficient data for SARIMAX. Please provide at least 30 data points.")
        st.stop()

    # Automatically fabricate 2 years of backward data
    monthly_aggregates = df.groupby(['JENIS', pd.Grouper(freq='M')]).agg(
        avg_qty=('QTY (KG)', 'mean'),
        avg_price=('HARGA/KG', 'mean')
    ).reset_index()

    start_date = pd.to_datetime("2022-01-01")
    end_date = pd.to_datetime("2023-12-31")
    fabricated_data = []

    unique_products = df['JENIS'].unique()
    for product in unique_products:
        product_agg = monthly_aggregates[monthly_aggregates['JENIS'] == product]
        for date in pd.date_range(start=start_date, end=end_date, freq='M'):
            if not product_agg.empty:
                avg_qty = product_agg['avg_qty'].mean()
                avg_price = product_agg['avg_price'].mean()
                fabricated_data.append({
                    'NO': None,
                    'HARI/TANGGAL': date,
                    'Nama': f"Fabricated {product}",
                    'JENIS': product,
                    'QTY (KG)': max(0, np.random.normal(avg_qty, avg_qty * 0.1)),
                    'HARGA/KG': max(0, np.random.normal(avg_price, avg_price * 0.1)),
                    'TOTAL (RP)': None,
                    'Date': date
                })

    fabricated_df = pd.DataFrame(fabricated_data)
    fabricated_df['TOTAL (RP)'] = fabricated_df['QTY (KG)'] * fabricated_df['HARGA/KG']

    # Combine fabricated data with original dataset
    df = pd.concat([fabricated_df, df], ignore_index=True).sort_values(by='Date')

    # Add holiday information
    class IndonesianHolidays(AbstractHolidayCalendar):
        rules = [
            Holiday('New Year', month=1, day=1, observance=nearest_workday),
            Holiday('Independence Day', month=8, day=17, observance=nearest_workday),
            Holiday('Christmas', month=12, day=25, observance=nearest_workday)
        ]

    def add_islamic_holidays(start_year, end_year):
        islamic_holidays = []
        for year in range(max(1900, int(start_year)), min(2100, int(end_year)) + 1):
            try:
                eid_al_fitr = convert.Hijri(year, 10, 1).to_gregorian()  # 1st day of Shawwal
                eid_al_adha = convert.Hijri(year, 12, 10).to_gregorian()  # 10th day of Dhu al-Hijjah
                islamic_holidays.extend([eid_al_fitr, eid_al_adha])
            except (ValueError, OverflowError):
                # Skip invalid Hijri-to-Gregorian conversions
                continue
        return islamic_holidays

    calendar = IndonesianHolidays()
    start_year = int(df['Date'].dt.year.min())
    end_year = int(df['Date'].dt.year.max())
    gregorian_holidays = calendar.holidays(start=df['Date'].min(), end=df['Date'].max())
    islamic_holidays = add_islamic_holidays(start_year, end_year)

    # Combine holidays and remove duplicates
    all_holidays = pd.to_datetime(
        list(set(gregorian_holidays).union(islamic_holidays)), errors='coerce'
    ).dropna()
    df['Holiday'] = 0
    df.loc[df.index.isin(all_holidays), 'Holiday'] = 1

    # Debug fabricated data
    st.write("Fabricated Data with Holidays Preview:")
    st.dataframe(fabricated_df.head())

    # Filter by product type
    unique_names = df['JENIS'].unique()
    selected_product = st.selectbox("Select a product type:", unique_names)
    selected_df = df[df['JENIS'] == selected_product]

    if selected_df.empty:
        st.error("No data found for the selected product.")
        st.stop()

    # Handle missing values
    if selected_df['QTY (KG)'].sum() == 0:
        st.error("The 'QTY (KG)' column contains only zeros or invalid values. Unable to forecast.")
        st.stop()

    # Ensure the index is a DatetimeIndex
    if not isinstance(selected_df.index, pd.DatetimeIndex):
        selected_df = selected_df.set_index('Date')  # Set 'Date' column as the index

    # Resample to monthly frequency and aggregate
    st.write("Processing data...")
    selected_df = selected_df.resample('M').agg({'QTY (KG)': 'sum', 'Holiday': 'max'}).interpolate()

    # Debug selected data
    st.write("Historical Data Summary:")
    st.dataframe(selected_df.head())
    st.write("Summary Statistics for QTY (KG):")
    st.write(selected_df['QTY (KG)'].describe())

    # Plot historical data
    st.subheader(f"Historical Data for {selected_product.capitalize()}")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_df.index, selected_df['QTY (KG)'], marker='o', linestyle='-')
    plt.title(f"QTY (KG) Over Time for {selected_product.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Quantity (KG)")
    plt.grid(True)
    st.pyplot(plt)

    # SARIMAX Forecasting
    forecast_periods = st.slider("Select forecast periods (months):", min_value=1, max_value=60, value=12)
    st.write("Fitting SARIMAX model with holidays as exogenous variable...")

    try:
        exog = selected_df[['Holiday']]  # Use 'Holiday' as exogenous variable
        model = SARIMAX(selected_df['QTY (KG)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exog)
        model_fit = model.fit(disp=False)

        # Forecast
        future_holidays = []
        future_dates = pd.date_range(start=selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:]
        for date in future_dates:
            future_holidays.append(1 if date in all_holidays else 0)

        exog_forecast = pd.DataFrame({'Holiday': future_holidays}, index=future_dates)

        forecast = model_fit.get_forecast(steps=forecast_periods, exog=exog_forecast)
        forecast_df = forecast.predicted_mean
        forecast_df[forecast_df < 0] = 0  # Clip negative values

        forecast_df = pd.DataFrame({'Forecast': forecast_df.values}, index=future_dates)

        # Display Forecast
        st.subheader("Forecasted Values:")
        st.dataframe(forecast_df)

        # Plot Forecast
        plt.figure(figsize=(10, 6))
        plt.plot(selected_df.index, selected_df['QTY (KG)'], label="Historical Data", marker='o')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color='orange', linestyle='--')
        plt.title(f"SARIMAX Forecast for {selected_product.capitalize()} (with Holidays)")
        plt.xlabel("Date")
        plt.ylabel("Quantity (KG)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"SARIMAX model failed: {e}")

except FileNotFoundError:
    st.error(f"File '{file_path}' not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")
