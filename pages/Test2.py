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
st.title("Farm Demand Analysis and Forecasting")

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

        # Add a combined bar chart to display both historical and forecasted values
        st.subheader("Visual Forecast: Historical and Predicted Egg Demand")

        # Combine historical and forecasted data for the bar chart
        historical_df = pd.DataFrame({'Quantity': selected_df['QTY (KG)']})
        historical_df['Type'] = 'Historical'
        forecast_df['Type'] = 'Forecast'
        combined_df = pd.concat([historical_df, forecast_df.rename(columns={'Forecast': 'Quantity'})])

        # Plot the combined bar chart
        plt.figure(figsize=(14, 7))
        colors = {'Historical': 'blue', 'Forecast': 'orange'}
        for data_type in combined_df['Type'].unique():
            data = combined_df[combined_df['Type'] == data_type]
            plt.bar(
                data.index.strftime('%Y-%m'),
                data['Quantity'],
                label=data_type,
                color=colors[data_type],
                alpha=0.7
            )

        plt.title(f"Historical and Forecasted Egg Demand for {selected_product_label}")
        plt.xlabel("Month")
        plt.ylabel("Quantity (KG)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)

        # Add a summary table for monthly forecast with pagination
        st.subheader("Monthly Egg Demand Table")

        # Number of rows per page
        rows_per_page = 12  # Show 12 months at a time
        total_rows = len(forecast_df)
        num_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page != 0 else 0)

        # Dropdown to select the page
        selected_page = st.selectbox("Select Page:", options=list(range(1, num_pages + 1)))

        # Slice the table based on the selected page
        start_idx = (selected_page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        paginated_table = forecast_df.iloc[start_idx:end_idx].reset_index()  # Ensure proper slicing
        paginated_table.rename(columns={'index': 'Month', 'Forecast': 'Forecasted Quantity (KG)'}, inplace=True)  # Proper renaming

        # Convert 'Month' column to a readable format
        paginated_table['Month'] = pd.to_datetime(paginated_table['Month']).dt.strftime('%B %Y')

        # Display the paginated table
        st.table(paginated_table)

        # Add download button for forecasted values
        forecast_csv = forecast_df.reset_index()
        forecast_csv.rename(columns={'index': 'Month', 'Forecast': 'Forecasted Quantity (KG)'}, inplace=True)
        forecast_csv['Month'] = pd.to_datetime(forecast_csv['Month']).dt.strftime('%B %Y')
        forecast_csv_data = forecast_csv.to_csv(index=False)
        st.download_button(
            label="Download Forecasted Values as CSV",
            data=forecast_csv_data,
            file_name='forecasted_values.csv',
            mime='text/csv',
        )



    except Exception as e:
        st.error(f"SARIMAX model failed: {e}")

else:
    st.error("Failed to load data from the database.")
