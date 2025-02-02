from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CSVtoDrill") \
    .master("local[*]") \
    .getOrCreate()

csv_file = "file:///C:/Users/rames/OneDrive/Documents/All coding projects/python codes/Weather prediction system/chennai weather data from year/2012-2023.csv"

try:
    df = spark.read.option("header", "true").csv(csv_file)
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")

output_parquet = "file:///C:/Users/rames/OneDrive/Documents/All coding projects/python codes/Weather prediction system/output_parquet"

try:
    df.write.parquet(output_parquet)
    print("DataFrame written to Parquet successfully.")
except Exception as e:
    print(f"Error writing DataFrame to Parquet: {e}")

spark.stop()