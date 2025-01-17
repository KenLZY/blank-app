import streamlit as st
from supabase import create_client, Client

# Initialize Supabase connection
url: str = "https://bwzsvppgjihgpkplyjly.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3enN2cHBnamloZ3BrcGx5amx5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzY5MjMwNzMsImV4cCI6MjA1MjQ5OTA3M30.wZ_-5AIcJYU-_WyCfFXQfEEOCqtEvitRpjCs3SkzUBQ"
supabase: Client = create_client(url, key)

# Perform a query
response = supabase.table("mytable").select("*").execute()

# Display results
if response.data:
    for row in response.data:
        st.write(f"{row['name']} has a {row['pet']}")
else:
    st.write("No data found.")
