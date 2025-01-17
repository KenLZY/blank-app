import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from hijri_converter import convert
import numpy as np
from supabase import create_client, Client

# Initialize Supabase connection
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# Fetch the data from the table
@st.cache_data(ttl=600)
def fetch_data_from_supabase():
    try:
        response = supabase.table("order").select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.error("No data retrieved from Supabase.")
            return None
    except Exception as e:
        st.error(f"Error querying Supabase: {e}")
        return None

# Streamlit App
st.title("Farm Product Analysis and Forecasting")

# Fetch data
df = fetch_data_from_supabase()

if df is not None:
    # Ensure the data is in the correct format
    if "collection_date" in df.columns:
        df['Date'] = pd.to_datetime(df['collection_date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')  # Drop invalid dates
    else:
        st.error("'collection_date' column is missing in the dataset.")
        st.stop()

    if "quantity" not in df.columns:
        st.error("'quantity' column is missing in the dataset.")
        st.stop()

    df['QTY (KG)'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

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
                eid_al_fitr = convert.Hijri(year, 10, 1).to_gregorian()
                eid_al_adha = convert.Hijri(year, 12, 10).to_gregorian()
                islamic_holidays.extend([eid_al_fitr, eid_al_adha])
            except (ValueError, OverflowError):
                continue
        return islamic_holidays

    calendar = IndonesianHolidays()
    start_year = int(df.index.year.min())
    end_year = int(df.index.year.max())
    gregorian_holidays = calendar.holidays(start=df.index.min(), end=df.index.max())
    islamic_holidays = add_islamic_holidays(start_year, end_year)

    all_holidays = pd.to_datetime(
        list(set(gregorian_holidays).union(islamic_holidays)), errors='coerce'
    ).dropna()
    df['Holiday'] = 0
    df.loc[df.index.isin(all_holidays), 'Holiday'] = 1

    # Preview the data
    st.subheader("Loaded Data Preview:")
    st.dataframe(df.head())

    # Export modified dataset as CSV
    full_csv = df.to_csv(index=True)
    st.download_button(
        label="Download Full Dataset as CSV",
        data=full_csv,
        file_name='full_modified_dataset.csv',
        mime='text/csv',
    )

    # Filter by product ID
    unique_products = df['product_id'].unique()
    selected_product = st.selectbox("Select a product ID:", unique_products)
    selected_df = df[df['product_id'] == selected_product]

    if selected_df.empty:
        st.error("No data found for the selected product.")
        st.stop()

    selected_df = selected_df.resample('M').agg({'QTY (KG)': 'sum', 'Holiday': 'max'}).interpolate()

    st.write("Historical Data Summary:")
    st.dataframe(selected_df.head())

    st.subheader(f"Historical Data for Product ID {selected_product}")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_df.index, selected_df['QTY (KG)'], marker='o', linestyle='-')
    plt.title(f"QTY (KG) Over Time for Product ID {selected_product}")
    plt.xlabel("Date")
    plt.ylabel("Quantity (KG)")
    plt.grid(True)
    st.pyplot(plt)

    forecast_periods = st.slider("Select forecast periods (months):", min_value=1, max_value=60, value=12)

    try:
        exog = selected_df[['Holiday']]
        model = SARIMAX(selected_df['QTY (KG)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exog)
        model_fit = model.fit(disp=False)

        future_holidays = [1 if date in all_holidays else 0 for date in pd.date_range(selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:]]
        exog_forecast = pd.DataFrame({'Holiday': future_holidays}, index=pd.date_range(selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:])
        forecast = model_fit.get_forecast(steps=forecast_periods, exog=exog_forecast)
        forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean.clip(lower=0)}, index=exog_forecast.index)

        st.subheader("Forecasted Values:")
        st.dataframe(forecast_df)

        plt.figure(figsize=(10, 6))
        plt.plot(selected_df.index, selected_df['QTY (KG)'], label="Historical Data", marker='o')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color='orange', linestyle='--')
        plt.title(f"SARIMAX Forecast for Product ID {selected_product} (with Holidays)")
        plt.xlabel("Date")
        plt.ylabel("Quantity (KG)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"SARIMAX model failed: {e}")

else:
    st.error("Failed to load data from the database.")
