import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------
# Dynamic Paths
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/bio_recommender_db.integrated_interactions"
MONGO_SPARK_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"



def create_spark_session():
    print("Initializing PySpark for Data Fusion...")
    spark = SparkSession.builder \
        .appName("BigData_Harmonization_ETL") \
        .config("spark.jars.packages", MONGO_SPARK_CONNECTOR) \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def process_and_normalize_dataset(spark, file_name, file_format, drug_col, target_col, score_col, dataset_name):
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return None

    print(f"Loading and transforming {file_name}...")
    sep = "\t" if file_format == "tsv" else ","
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=sep)

    # Standardize columns (using string cast for SMILES and Sequences)
    df = df.select(
        F.col(drug_col).cast("string").alias("DrugID"),
        F.col(target_col).cast("string").alias("TargetID"),
        F.col(score_col).cast("float").alias("RawScore")
    )
    df = df.withColumn("Source", F.lit(dataset_name))

    # Drop nulls and zero/negative values (log10 of <=0 is mathematically undefined)
    df = df.filter((F.col("RawScore").isNotNull()) & (F.col("RawScore") > 0))

    # ---------------------------------------------------------
    # BIOINFORMATICS CORRECTION: The p-scale transformation
    # ---------------------------------------------------------
    # For TSV files (nM concentration), lower is better. We apply: 9 - log10(X)
    # For Davis/KIBA, higher is already better, so we keep RawScore.
    if dataset_name in ["IC50", "Kd", "Ki", "EC50"]:
        df = df.withColumn("AffinityScore", 9.0 - F.log10(F.col("RawScore")))
    else:
        df = df.withColumn("AffinityScore", F.col("RawScore"))

    # --- MIN-MAX NORMALIZATION (Scaling p-scale to 1-5 for ALS) ---
    window_spec = Window.partitionBy("Source")
    df = df.withColumn("MaxScore", F.max("AffinityScore").over(window_spec)) \
        .withColumn("MinScore", F.min("AffinityScore").over(window_spec))

    df = df.withColumn(
        "NormalizedScore",
        F.when(F.col("MaxScore") == F.col("MinScore"), F.lit(3.0))
        .otherwise(((F.col("AffinityScore") - F.col("MinScore")) / (F.col("MaxScore") - F.col("MinScore")) * 4) + 1)
    )

    return df.select("DrugID", "TargetID", "NormalizedScore", "Source")


def main():
    spark = create_spark_session()
    # Define the schemas based on exact column names in the files
    datasets_config = [
        {"file": "davis_all.csv", "format": "csv", "d_col": "compound_iso_smiles", "t_col": "target_sequence",
         "s_col": "affinity", "name": "Davis"},
        {"file": "kiba_all.csv", "format": "csv", "d_col": "compound_iso_smiles", "t_col": "target_sequence",
         "s_col": "affinity", "name": "KIBA"},
        {"file": "IC50_bind.tsv", "format": "tsv", "d_col": "drug_id", "t_col": "target_id", "s_col": "affinity",
         "name": "IC50"},
        {"file": "Kd_bind.tsv", "format": "tsv", "d_col": "drug_id", "t_col": "target_id", "s_col": "affinity",
         "name": "Kd"},
        {"file": "Ki_bind.tsv", "format": "tsv", "d_col": "drug_id", "t_col": "target_id", "s_col": "affinity",
         "name": "Ki"},
        {"file": "EC50_bind.tsv", "format": "tsv", "d_col": "drug_id", "t_col": "target_id", "s_col": "affinity",
         "name": "EC50"}
    ]

    dataframes = []

    # Process each dataset
    for config in datasets_config:
        df = process_and_normalize_dataset(
            spark, config["file"], config["format"],
            config["d_col"], config["t_col"], config["s_col"], config["name"]
        )
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        print("Error: No datasets were found in the raw folder!")
        return

    # Fusion: Union all dataframes together
    print("Fusing all datasets into a single massive graph...")
    master_df = dataframes[0]
    for df in dataframes[1:]:
        master_df = master_df.union(df)

    total_records = master_df.count()
    print(f"Total integrated interactions: {total_records}")

    # Show a sample of the harmonized data
    master_df.show(10, truncate=False)

    # Save to MongoDB
    print("Saving harmonized Big Data to MongoDB...")
    master_df.write \
        .format("mongodb") \
        .mode("overwrite") \
        .save()

    print("Success! ETL Data Fusion Pipeline completed.")
    spark.stop()


if __name__ == "__main__":
    main()