import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import pickle
import io
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CSVtoDrill") \
    .master("local[*]") \
    .getOrCreate()

# Load CSV file
csv_file = "file:///C:/Users/rames/OneDrive/Documents/All coding projects/python codes/Weather prediction system/2003-2024.csv"

try:
    df = spark.read.option("header", "true").csv(csv_file)
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")

# Convert Spark DataFrame to Pandas DataFrame
df_pandas = df.toPandas()

# Streamlit app title
st.title("Weather Forecast App for 2025 and 2026")

# Display the dataset preview and columns
st.write("Dataset preview:")
st.dataframe(df_pandas.head())
st.write("Columns in the dataset:")
st.write(df_pandas.columns)

# Function to detect date and temperature columns
def detect_columns(df):
    date_column = None
    temp_columns = []
    for col in df.columns:
        # Check for date column
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
            date_column = col
        # Check for temperature columns
        elif "tavg" in col.lower() or "tmin" in col.lower() or "tmax" in col.lower():
            temp_columns.append(col)
    return date_column, temp_columns

# Detect the date and temperature columns
date_column, temp_columns = detect_columns(df_pandas)

# Check if the required columns are detected
if date_column and temp_columns:
    st.write(f"Detected Date column: {date_column}")
    st.write(f"Detected Temperature columns: {temp_columns}")

    # Convert the date column to datetime (if not already) and ensure no rows are missed
    df_pandas[date_column] = pd.to_datetime(df_pandas[date_column], errors='coerce')

    # Drop rows where the date is NaT
    df_pandas = df_pandas.dropna(subset=[date_column])

    # Specify a date range
    start_date = "2003-01-01"
    end_date = "2023-12-31"

    # Filter the data based on the date range
    df_filtered = df_pandas.loc[(df_pandas[date_column] >= start_date) & (df_pandas[date_column] <= end_date)]

    # Convert temperature columns to numeric and drop non-numeric entries
    for temp_col in temp_columns:
        df_filtered[temp_col] = pd.to_numeric(df_filtered[temp_col], errors='coerce')  # Convert to numeric

    # Drop rows with NaN in temperature columns but keep all years
    df_filtered = df_filtered.dropna(subset=temp_columns)

    # Display filtered data
    st.write("Filtered data:")
    st.dataframe(df_filtered)

    # Visualize the temperature data
    st.subheader("Temperature Data from 2003 to 2024")
    fig, ax = plt.subplots(figsize=(12, 6))
    for temp_col in temp_columns:
        ax.plot(df_filtered[date_column], df_filtered[temp_col], label=temp_col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Data from 2003 to 2024')
    ax.legend()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png')
    buf1.seek(0)
    st.image(buf1, caption='Temperature Data')
    # Rename columns for NeuralProphet compatibility
    df_pandas = df_pandas.rename(columns={date_column: "ds", temp_columns[0]: "y"})
    data = df_pandas[['ds', 'y']]

    # Train the NeuralProphet model
    m = NeuralProphet()
    st.write("Training the model... (may take some time)")
    m.fit(data, freq='D', epochs=500)

    # Forecast for 2025 and 2026 (next 730 days)
    future_periods = 730
    future = m.make_future_dataframe(data, periods=future_periods)
    forecast = m.predict(future)

    # Adjust the forecast (optional scaling for visualization purposes)
    forecast['yhat1'] = forecast['yhat1'] + (45 - forecast['yhat1'].max())

    # Display forecast data
    st.write("Columns in the forecast DataFrame:")
    st.write(forecast.columns)

    # Plot forecasted temperature data for 2025 and 2026
    st.subheader("Forecasted Temperature Data for 2025 and 2026")
    plt.figure(figsize=(10, 5))
    plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Forecasted Temperature Data for 2025 and 2026')
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    st.image(buf2, caption='Forecasted Temperature Data')

    # Plot forecast components
    st.subheader("Forecast Components")

    # NeuralProphet.plot_components() returns a list of figures
    component_figs = m.plot_components(forecast)

    # Display each component figure
    for fig in component_figs:
        st.pyplot(fig)

    # Save the model if the user clicks the button
    if st.button("Save Model"):
        with open('saved_model.pkl', "wb") as f:
            pickle.dump(m, f)
        st.success("Model saved successfully!")

    # Load a saved model if the user clicks the button
    if st.button("Load Model"):
        try:
            with open('saved_model.pkl', "rb") as f:
                m = pickle.load(f)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error("No model found to load. Please save a model first.")

else:
    # Display error if required columns are not detected
    st.error("Could not detect the required date or temperature columns. Please check your CSV file.")
