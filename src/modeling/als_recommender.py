import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import explode, col, collect_list, struct
from datetime import datetime, timezone


os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
# ---------------------------------------------------------
# MongoDB Configuration
# ---------------------------------------------------------
MONGO_DB          = "bio_recommender_db"
MONGO_BASE_URI    = "mongodb://localhost:27017"
MONGO_READ_URI    = f"{MONGO_BASE_URI}/{MONGO_DB}.integrated_interactions"
MONGO_PRED_URI    = f"{MONGO_BASE_URI}/{MONGO_DB}.predicted_interactions"
MONGO_METRICS_URI = f"{MONGO_BASE_URI}/{MONGO_DB}.model_metrics"
MONGO_CONNECTOR   = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("DTI_ALS_Production")
        .master("local[*]")
        .config("spark.jars.packages", MONGO_CONNECTOR)
        .config("spark.mongodb.read.connection.uri", MONGO_READ_URI)
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("./spark_checkpoints")
    return spark


def load_data(spark):
    print("[INFO] Loading integrated DTI data from MongoDB...")
    return spark.read.format("mongodb").load()


def train_and_evaluate_als(df):
    """
    Prepare data, index string IDs, cache for iterative ALS,
    train explicit ALS, evaluate RMSE + AUPR.
    """
    # Keep only the columns we need; drop any nulls
    clean_df = df.select(
        "DrugID", "TargetID", "NormalizedScore"
    ).dropna()

    # --- String Indexing ---
    drug_indexer   = StringIndexer(inputCol="DrugID",   outputCol="drug_idx",   handleInvalid="keep")
    target_indexer = StringIndexer(inputCol="TargetID", outputCol="target_idx", handleInvalid="keep")
    pipeline       = Pipeline(stages=[drug_indexer, target_indexer])
    pipeline_model = pipeline.fit(clean_df)
    indexed_df     = pipeline_model.transform(clean_df)

    # -----------------------------------------------------------
    # CACHING: ALS is an iterative algorithm — it repeatedly reads
    # the training data in each iteration. Caching the indexed
    # DataFrame in RAM avoids recomputing from disk on every pass,
    # significantly improving Spark performance.
    # -----------------------------------------------------------
    indexed_df = indexed_df.cache()
    indexed_df.count()   # materialise the cache now
    print(f"[INFO] Cached indexed_df with {indexed_df.count()} rows")

    # Train / test split
    training, test = indexed_df.randomSplit([0.8, 0.2], seed=42)
    training = training.cache()
    test     = test.cache()
    training.count()   # force materialisation
    test.count()

    print("[INFO] Training Spark ALS model (explicit, NormalizedScore as rating)...")

    # -----------------------------------------------------------
    # ALS — explicit mode
    # We use implicitPrefs=False because NormalizedScore is derived
    # from biochemical affinity measurements and behaves as an
    # explicit interaction score (higher = stronger binding),
    # not a binary preference signal.
    # -----------------------------------------------------------
    als = ALS(
        maxIter=15,
        regParam=0.05,
        rank=20,
        implicitPrefs=False,
        userCol="drug_idx",
        itemCol="target_idx",
        ratingCol="NormalizedScore",
        coldStartStrategy="drop",
    )

    model       = als.fit(training)
    predictions = model.transform(test)

    # Break long lineage graph before evaluation
    predictions = predictions.checkpoint()
    predictions = predictions.dropna(subset=["prediction"])

    # --- RMSE (primary metric, matches course material) ---
    evaluator_rmse = RegressionEvaluator(
        metricName="rmse",
        labelCol="NormalizedScore",
        predictionCol="prediction",
    )
    rmse = evaluator_rmse.evaluate(predictions)

    # --- AUPR (secondary metric; positive = NormalizedScore >= 3) ---
    binarized = (
        predictions
        .withColumn("TrueLabel", F.when(F.col("NormalizedScore") >= 3.0, 1.0).otherwise(0.0))
        .withColumn("prediction", F.col("prediction").cast("double"))
    )
    evaluator_aupr = BinaryClassificationEvaluator(
        rawPredictionCol="prediction",
        labelCol="TrueLabel",
        metricName="areaUnderPR",
    )
    aupr = evaluator_aupr.evaluate(binarized)

    print("=" * 46)
    print(f"  RMSE :  {rmse:.4f}")
    print(f"  AUPR :  {aupr:.4f}")
    print("=" * 46)

    # Build a one-row metrics DataFrame to persist
    metrics_df = predictions.sparkSession.createDataFrame(
        [(
            "ALS-explicit",
            float(rmse),
            float(aupr),
            20,
            0.05,
            15,
            False,
            datetime.now(timezone.utc).isoformat()
        )],
        ["algorithm", "rmse", "aupr", "rank",
         "regParam", "maxIter", "implicitPrefs", "timestamp"],
    )

    return model, pipeline_model, metrics_df


def generate_recommendations(spark, model, pipeline_model):
    """
    Produce top-5 drug candidates for every target
    and decode indices back to real IDs.
    """
    print("[INFO] Generating top-5 drug recommendations per target...")
    target_recs = model.recommendForAllItems(5)

    drug_labels   = pipeline_model.stages[0].labels
    target_labels = pipeline_model.stages[1].labels

    # Flatten nested recommendation array
    flat_recs = (
        target_recs
        .select("target_idx", explode("recommendations").alias("rec"))
        .select(
            "target_idx",
            col("rec.drug_idx").alias("drug_idx"),
            col("rec.rating").alias("score"),
        )
    )

    # Decode integer indices back to original string IDs
    target_converter = IndexToString(inputCol="target_idx", outputCol="TargetID", labels=target_labels)
    drug_converter   = IndexToString(inputCol="drug_idx",   outputCol="DrugID",   labels=drug_labels)

    decoded = target_converter.transform(flat_recs)
    decoded = drug_converter.transform(decoded)

    final_predictions = (
        decoded
        .groupBy("TargetID")
        .agg(collect_list(struct("DrugID", "score")).alias("recommendations"))
    )

    return final_predictions


def main():
    spark = create_spark_session()
    try:
        df = load_data(spark)

        model, pipeline_model, metrics_df = train_and_evaluate_als(df)

        # Persist metrics to MongoDB
        print("[INFO] Saving model metrics to MongoDB...")
        metrics_df.write \
            .format("mongodb") \
            .option("spark.mongodb.write.connection.uri", MONGO_METRICS_URI) \
            .mode("overwrite") \
            .save()

        # Persist recommendations to MongoDB
        print("[INFO] Saving recommendations to MongoDB...")
        final_predictions = generate_recommendations(spark, model, pipeline_model)
        final_predictions.write \
            .format("mongodb") \
            .option("spark.mongodb.write.connection.uri", MONGO_PRED_URI) \
            .mode("overwrite") \
            .save()

        print("[SUCCESS] All done. Run app.py to explore results.")

    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()