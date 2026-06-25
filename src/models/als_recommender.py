from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import explode, col, collect_list, struct

# Configuration
MONGO_URI = "mongodb://localhost:27017/bio_recommender_db.integrated_interactions"
MONGO_SPARK_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"


def create_spark_session():
    print("Initializing PySpark ML Environment...")
    spark = SparkSession.builder \
        .appName("DTI_ALS_Production") \
        .master("local[*]") \
        .config("spark.jars.packages", MONGO_SPARK_CONNECTOR) \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("./spark_checkpoints")
    return spark


def load_data(spark):
    print("Loading harmonized DTI data from MongoDB...")
    return spark.read.format("mongodb").load()


def train_and_evaluate_als(df):
    print("Preparing DTI Pipeline (Drug & Target Indexing)...")
    drug_indexer = StringIndexer(inputCol="DrugID", outputCol="drug_idx", handleInvalid="keep")
    target_indexer = StringIndexer(inputCol="TargetID", outputCol="target_idx", handleInvalid="keep")

    pipeline = Pipeline(stages=[drug_indexer, target_indexer])

    # BUG FIX: We MUST save the fitted PipelineModel to extract labels later
    pipeline_model = pipeline.fit(df)
    indexed_df = pipeline_model.transform(df)

    print("Splitting data into Training (80%) and Testing (20%) sets...")
    (training, test) = indexed_df.randomSplit([0.8, 0.2], seed=42)

    print("Training Implicit ALS Model (Optimized Parameters)...")
    als = ALS(
        maxIter=15,
        regParam=0.05,
        alpha=40.0,
        implicitPrefs=True,
        userCol="drug_idx",
        itemCol="target_idx",
        ratingCol="NormalizedScore",
        coldStartStrategy="drop"
    )

    model = als.fit(training)
    predictions = model.transform(test)

    print("Checkpointing predictions to break lineage graph and prevent StackOverflow...")
    predictions = predictions.checkpoint()

    print("Evaluating Model Performance...")
    predictions = predictions.dropna(subset=["prediction"])

    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="NormalizedScore", predictionCol="prediction")
    rmse = evaluator_rmse.evaluate(predictions)

    print("Calculating AUPR (Crucial for Sparse Biological Data)...")
    binarized_predictions = predictions.withColumn(
        "TrueLabel", F.when(F.col("NormalizedScore") >= 3.0, 1.0).otherwise(0.0)
    ).withColumn("prediction", F.col("prediction").cast("double"))

    evaluator_aupr = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="TrueLabel",
                                                   metricName="areaUnderPR")
    aupr = evaluator_aupr.evaluate(binarized_predictions)

    print("========================================")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Area Under PR Curve (AUPR):    {aupr:.4f}")
    print("========================================")

    # BUG FIX: Return the pipeline_model, not the raw pipeline
    return model, pipeline_model


def main():
    spark = create_spark_session()
    try:
        df = load_data(spark)
        model, pipeline_model = train_and_evaluate_als(df)

        print("Generating Top 3 Novel Drug Candidates for Targets...")
        target_recs = model.recommendForAllItems(3)

        print("Translating ML indices back to real Chemical & Biological sequences...")

        # Now this will work perfectly!
        drug_labels = pipeline_model.stages[0].labels
        target_labels = pipeline_model.stages[1].labels

        flat_recs = target_recs.select("target_idx", explode("recommendations").alias("rec")) \
            .select("target_idx", col("rec.drug_idx").alias("drug_idx"), col("rec.rating").alias("score"))

        target_converter = IndexToString(inputCol="target_idx", outputCol="TargetID", labels=target_labels)
        flat_recs = target_converter.transform(flat_recs)

        drug_converter = IndexToString(inputCol="drug_idx", outputCol="DrugID", labels=drug_labels)
        flat_recs = drug_converter.transform(flat_recs)

        final_predictions = flat_recs.groupBy("TargetID").agg(
            collect_list(struct("DrugID", "score")).alias("recommendations")
        )

        print("Saving REAL BIOCHEMICAL recommendations to MongoDB...")
        final_predictions.write \
            .format("mongodb") \
            .option("spark.mongodb.write.connection.uri",
                    "mongodb://localhost:27017/bio_recommender_db.predicted_interactions") \
            .mode("overwrite") \
            .save()
        print("Predictions saved successfully for the Dashboard!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()