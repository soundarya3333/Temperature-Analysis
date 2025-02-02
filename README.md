# Temperature Analysis:
# Weather Forecast App

## Overview
The Weather Forecast App is a Streamlit application designed to predict weather temperatures for the years 2025 and 2026 using historical weather data. The app utilizes the NeuralProphet library for time series forecasting and integrates with Apache Spark for efficient data handling. This project aims to provide users with an interactive interface to visualize historical temperature data and forecast future temperatures.

## Features
- **Data Loading**: The app loads historical weather data from a CSV file using Apache Spark, ensuring efficient processing of large datasets.
- **Data Visualization**: Users can view a preview of the dataset, including the columns and filtered data based on a specified date range.
- **Temperature Forecasting**: The app employs the NeuralProphet model to forecast temperature data for the next two years (2025 and 2026).
- **Interactive Components**: Users can save and load trained models, allowing for persistent forecasting capabilities.
- **Forecast Visualization**: The app visualizes both historical temperature data and forecasted temperatures, along with forecast components.

## Requirements
To run this application, you need the following libraries:
- Streamlit
- Pandas
- NeuralProphet
- Matplotlib
- PySpark

You can install the required libraries using pip:
```bash
pip install streamlit pandas neuralprophet matplotlib pyspark
```

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/weather-forecast-app.git
   cd weather-forecast-app
   ```

2. **Prepare the Data**:
   - Ensure you have a CSV file containing historical weather data. The file should include date and temperature columns (e.g., average, minimum, and maximum temperatures).

3. **Modify the CSV Path**:
   - Update the `csv_file` variable in `main2.py` to point to your local CSV file path.

4. **Run the Application**:
   ```bash
   streamlit run main2.py
   ```

5. **Interact with the App**:
   - Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
   - Explore the dataset, visualize historical data, and generate forecasts for the upcoming years.

## Code Explanation
- **Spark Session Initialization**: The app initializes a Spark session to handle large CSV files efficiently.
- **Data Loading**: The CSV file is loaded into a Spark DataFrame, which is then converted to a Pandas DataFrame for easier manipulation.
- **Column Detection**: The app detects the date and temperature columns from the dataset to ensure proper data processing.
- **Data Filtering**: The data is filtered based on a specified date range, and non-numeric entries in temperature columns are handled.
- **Model Training**: The NeuralProphet model is trained on the historical data, and forecasts are generated for the next 730 days.
- **Visualization**: The app visualizes both the historical and forecasted temperature data, along with the components of the forecast.

## Saving and Loading Models
- Users can save the trained model to a file and load it later for further predictions. This feature allows for continuity in forecasting without retraining the model each time.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/) for providing an easy way to create web applications.
- [NeuralProphet](https://neuralprophet.com/) for advanced time series forecasting capabilities.
- [Apache Spark](https://spark.apache.org/) for efficient data processing.

---

Feel free to customize the repository link, license, and any other sections as needed!
