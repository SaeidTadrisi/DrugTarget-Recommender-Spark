from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Configuration
MONGO_URI = "mongodb://localhost:27017/bio_recommender_db.integrated_interactions"
MONGO_SPARK_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"


def create_spark_session():
    print("Initializing PySpark ML Environment...")
    spark = SparkSession.builder \
        .appName("Advanced_DTI_ALS") \
        .config("spark.jars.packages", MONGO_SPARK_CONNECTOR) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("./spark_checkpoints")
    return spark


def load_data(spark):
    print("Loading harmonized DTI data from MongoDB...")
    df = spark.read.format("mongodb").load()
    return df


def train_and_evaluate_als(df):
    print("Preparing DTI Pipeline (Drug & Target Indexing)...")

    # 1. Index Strings to Numeric (ALS requirement)
    drug_indexer = StringIndexer(inputCol="DrugID", outputCol="drug_idx", handleInvalid="keep")
    target_indexer = StringIndexer(inputCol="TargetID", outputCol="target_idx", handleInvalid="keep")

    # Apply indexing
    pipeline = Pipeline(stages=[drug_indexer, target_indexer])
    indexed_df = pipeline.fit(df).transform(df)

    # 2. Train-Test Split (80% Train, 20% Test)
    print("Splitting data into Training (80%) and Testing (20%) sets...")
    (training, test) = indexed_df.randomSplit([0.8, 0.2], seed=42)

    # 3. Build ALS Model (IMPLICIT FEEDBACK architecture)
    print("Training Implicit ALS Model (Addressing Sparsity)...")
    als = ALS(
        maxIter=15,
        regParam=0.05,
        alpha=40.0,  # Confidence parameter for implicit feedback
        implicitPrefs=True,  # CRITICAL: Treats scores as confidence of interaction, not exact ratings
        userCol="drug_idx",
        itemCol="target_idx",
        ratingCol="NormalizedScore",
        coldStartStrategy="drop"
    )

    model = als.fit(training)

    # 4. Generate Predictions on Test Set
    predictions = model.transform(test)
    print("Checkpointing predictions to break lineage graph and prevent StackOverflow...")
    predictions = predictions.checkpoint()

    # 5. Advanced Evaluation (Master's Degree Level)
    print("Evaluating Model Performance...")

    # Drop null predictions (Crucial for Cold Start items)
    predictions = predictions.dropna(subset=["prediction"])

    # Metric 1: RMSE
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="NormalizedScore", predictionCol="prediction")
    rmse = evaluator_rmse.evaluate(predictions)

    # Metric 2: AUPR (Using modern DataFrame API - Safe from StackOverflow due to Checkpoint)
    print("Calculating AUPR (Using Native JVM DataFrame Engine)...")

    binarized_predictions = predictions.withColumn(
        "TrueLabel", F.when(F.col("NormalizedScore") >= 3.0, 1.0).otherwise(0.0)
    ).withColumn("prediction", F.col("prediction").cast("double"))

    evaluator_aupr = BinaryClassificationEvaluator(
        rawPredictionCol="prediction", labelCol="TrueLabel", metricName="areaUnderPR"
    )
    aupr = evaluator_aupr.evaluate(binarized_predictions)

    print("========================================")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Area Under PR Curve (AUPR):    {aupr:.4f} (Closer to 1.0 is better)")
    print("========================================")

    return model, pipeline


def main():
    spark = create_spark_session()

    try:
        df = load_data(spark)
        model, indexer_pipeline = train_and_evaluate_als(df)

        # Display top 3 novel drug recommendations for 5 target proteins
        print("Generating Top 3 Novel Drug Candidates for Sample Targets...")
        target_recs = model.recommendForAllItems(3)  # items = targets
        target_recs.show(5, truncate=False)
        print("Saving predicted recommendations to MongoDB...")
        target_recs.write \
            .format("mongodb") \
            .option("spark.mongodb.write.connection.uri",
                    "mongodb://localhost:27017/bio_recommender_db.predicted_interactions") \
            .mode("overwrite") \
            .save()
        print("Predictions saved successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()