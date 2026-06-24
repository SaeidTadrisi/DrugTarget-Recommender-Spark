from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType

# ---------------------------------------------------------
# Configuration (MongoDB & Spark)
# ---------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/bio_recommender_db.patients_interactions"
MONGO_SPARK_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"


def create_spark_session():
    """Initialize Apache Spark Session."""
    print("Initializing Apache Spark Session...")
    spark = SparkSession.builder \
        .appName("DrugTargetRecommenderALS") \
        .config("spark.jars.packages", MONGO_SPARK_CONNECTOR) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark):
    """Load data from MongoDB."""
    print("Loading data from MongoDB...")
    df = spark.read.format("mongodb").load()

    # Cast ReactionScore to Float (Required by ALS algorithm)
    df = df.withColumn("ReactionScore", df["ReactionScore"].cast(FloatType()))
    return df


def train_recommender_model(df):
    """
    Build and train the ALS Recommender Pipeline.
    1. Convert String IDs to Numeric Indices (StringIndexer)
    2. Train the Alternating Least Squares (ALS) model
    """
    print("Preparing Data Pipeline for Machine Learning...")

    # Initialize Indexers for String to Integer conversion
    patient_indexer = StringIndexer(inputCol="PatientID", outputCol="patient_idx", handleInvalid="keep")
    drug_indexer = StringIndexer(inputCol="DrugID", outputCol="drug_idx", handleInvalid="keep")

    # Initialize ALS Model
    als = ALS(
        maxIter=10,  # Number of iterations
        regParam=0.1,  # Regularization parameter to prevent overfitting
        userCol="patient_idx",  # The numeric patient column
        itemCol="drug_idx",  # The numeric drug column
        ratingCol="ReactionScore",  # The reaction score
        coldStartStrategy="drop"  # Drop NaN values in predictions
    )

    # Create the Machine Learning Pipeline
    pipeline = Pipeline(stages=[patient_indexer, drug_indexer, als])

    print("Training the ALS Model (This requires high CPU usage)...")
    # Fit the pipeline to the data
    model = pipeline.fit(df)
    print("Model training completed successfully!")

    return model


def generate_recommendations(model):
    """Generate top 3 drug recommendations for all patients."""
    print("Generating novel drug-target predictions...")

    # Extract the trained ALS model from the Pipeline (it's the last stage)
    als_model = model.stages[-1]

    # Generate top 3 drug recommendations for each user
    user_recs = als_model.recommendForAllUsers(3)

    print("Top 3 Recommended Drugs for Patients:")
    user_recs.show(10, truncate=False)


def main():
    """Main execution flow."""
    spark = create_spark_session()

    try:
        # 1. Load Data
        df = load_data(spark)

        # 2. Train Model
        trained_pipeline_model = train_recommender_model(df)

        # 3. Generate Predictions
        generate_recommendations(trained_pipeline_model)

    except Exception as e:
        print("An error occurred during the ML pipeline execution:")
        print(e)
    finally:
        spark.stop()
        print("Spark Session gracefully stopped.")


if __name__ == "__main__":
    main()