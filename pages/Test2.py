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

# Fetch the data from the orders table
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

# Fetch product names from the product table
@st.cache_data(ttl=600)
def fetch_products_from_supabase():
    try:
        response = supabase.table("product").select("id, display_name").execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.error("No product data retrieved from Supabase.")
            return None
    except Exception as e:
        st.error(f"Error querying Supabase for products: {e}")
        return None

# Streamlit App
st.title("Farm Product Analysis and Forecasting")

# Fetch data from Supabase
df = fetch_data_from_supabase()
products_df = fetch_products_from_supabase()

if df is not None and products_df is not None:
    # Merge product names into the orders dataset
    if "product_id" in df.columns and "id" in products_df.columns and "display_name" in products_df.columns:
        products_df.rename(columns={"id": "product_id"}, inplace=True)
        df = df.merge(products_df[['product_id', 'display_name']], on='product_id', how='left')

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

    # Export modified dataset as CSV
    full_csv = df.to_csv(index=True)
    st.download_button(
        label="Download Full Dataset as CSV",
        data=full_csv,
        file_name='full_modified_dataset.csv',
        mime='text/csv',
    )

    # Create a dropdown with product names
    if "display_name" in df.columns:
        unique_products = df[['product_id', 'display_name']].drop_duplicates()
        unique_products['dropdown_label'] = unique_products.apply(
            lambda row: f"{row['display_name']} (ID: {row['product_id']})", axis=1
        )
        selected_product_label = st.selectbox("Select a product:", unique_products['dropdown_label'])
        selected_product_id = int(selected_product_label.split("(ID: ")[-1].strip(")"))
    else:
        # Fallback to product_id only
        unique_products = df['product_id'].unique()
        selected_product_id = st.selectbox("Select a product ID:", unique_products)

    selected_df = df[df['product_id'] == selected_product_id]

    if selected_df.empty:
        st.error("No data found for the selected product.")
        st.stop()

    selected_df = selected_df.resample('M').agg({'QTY (KG)': 'sum', 'Holiday': 'max'}).interpolate()

    forecast_periods = st.slider("Select forecast periods (months):", min_value=1, max_value=60, value=12)

    try:
        exog = selected_df[['Holiday']]
        model = SARIMAX(selected_df['QTY (KG)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exog)
        model_fit = model.fit(disp=False)

        future_holidays = [1 if date in all_holidays else 0 for date in pd.date_range(selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:]]
        exog_forecast = pd.DataFrame({'Holiday': future_holidays}, index=pd.date_range(selected_df.index[-1], periods=forecast_periods + 1, freq='M')[1:])
        forecast = model_fit.get_forecast(steps=forecast_periods, exog=exog_forecast)
        forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean.clip(lower=0)}, index=exog_forecast.index)

        # Add a bar chart to display the monthly forecast visually
        st.subheader("Visual Forecast: Monthly Egg Production")
        plt.figure(figsize=(12, 6))
        plt.bar(forecast_df.index.strftime('%Y-%m'), forecast_df['Forecast'], color='orange', alpha=0.7)
        plt.title(f"Monthly Egg Production Forecast for {selected_product_label}")
        plt.xlabel("Month")
        plt.ylabel("Quantity (KG)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)

        # Add user-friendly textual summary
        total_forecast = forecast_df['Forecast'].sum()
        max_forecast = forecast_df['Forecast'].max()
        min_forecast = forecast_df['Forecast'].min()
        st.write(f"### Forecast Summary:")
        st.write(f"- **Total Forecasted Quantity**: {total_forecast:.2f} KG")
        st.write(f"- **Peak Month**: {forecast_df['Forecast'].idxmax().strftime('%B %Y')} with **{max_forecast:.2f} KG**")
        st.write(f"- **Lowest Month**: {forecast_df['Forecast'].idxmin().strftime('%B %Y')} with **{min_forecast:.2f} KG**")

    except Exception as e:
        st.error(f"SARIMAX model failed: {e}")

else:
    st.error("Failed to load data from the database.")
