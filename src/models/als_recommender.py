import os
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# Configuration (MongoDB & Spark)
# ---------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/bio_recommender_db.patients_interactions"

# Depending on your PySpark version, the connector version might vary.
# We use the modern MongoDB Spark Connector (v10+)
MONGO_SPARK_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"

def create_spark_session():
    """
    Initialize Apache Spark Session and inject the MongoDB Connector.
    """
    print("Initializing Apache Spark Session (This might take a minute to download packages)...")

    # Create Spark Session with MongoDB packages configured
    spark = SparkSession.builder \
        .appName("DrugTargetRecommenderALS") \
        .config("spark.jars.packages", MONGO_SPARK_CONNECTOR) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Set log level to ERROR to keep the console clean from Spark's verbose INFO logs
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark Session initialized successfully!")

    return spark


def test_mongodb_connection(spark):
    """
    Test reading data directly from MongoDB into a Spark DataFrame.
    """
    print("Attempting to read data from MongoDB...")

    try:
        # Load data from MongoDB
        df = spark.read \
            .format("mongodb") \
            .load()

        print(f"Successfully loaded {df.count()} records from MongoDB.")
        print("🔍 Showing the first 5 records:")

        # Show top 5 rows
        df.show(5)

        return df
    except Exception as e:
        print("Error connecting Spark to MongoDB. Details:")
        print(e)
        exit(1)


def main():
    """Main execution flow for Spark Recommender Initialization."""
    spark = create_spark_session()

    # Test the connection and Data Loading
    df = test_mongodb_connection(spark)

    # Stop Spark gracefully
    spark.stop()
    print("Spark Session gracefully stopped.")


if __name__ == "__main__":
    main()