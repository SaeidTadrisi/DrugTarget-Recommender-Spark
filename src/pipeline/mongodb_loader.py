import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ---------------------------------------------------------
# MongoDB Configuration
# ---------------------------------------------------------
MONGO_DB          = "bio_recommender_db"
MONGO_BASE_URI    = "mongodb://localhost:27017"
MONGO_WRITE_URI   = f"{MONGO_BASE_URI}/{MONGO_DB}.integrated_interactions"
MONGO_STRING_URI  = f"{MONGO_BASE_URI}/{MONGO_DB}.string_ppi_edges"
MONGO_CONNECTOR   = "org.mongodb.spark:mongo-spark-connector_2.13:11.1.0"

# ---------------------------------------------------------
# STRING Configuration
# ---------------------------------------------------------
STRING_FILE             = "9606.protein.links.v12.0.txt"
STRING_CONF_THRESHOLD   = 0.70   # keep only high-confidence PPI edges


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("BigData_DTI_ETL")
        .config("spark.jars.packages", MONGO_CONNECTOR)
        .config("spark.mongodb.write.connection.uri", MONGO_WRITE_URI)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def process_affinity_dataset(
    spark, file_name, file_format,
    drug_col, target_col, score_col, dataset_name
):
    """Load one affinity source, apply p-scale if needed, min-max normalise."""
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"[WARN] File not found, skipping: {file_name}")
        return None

    sep = "\t" if file_format == "tsv" else ","
    df  = spark.read.csv(file_path, header=True, inferSchema=True, sep=sep)

    df = df.select(
        F.col(drug_col).cast("string").alias("DrugID"),
        F.col(target_col).cast("string").alias("TargetID"),
        F.col(score_col).cast("double").alias("RawScore"),
    )
    df = df.withColumn("Source", F.lit(dataset_name))
    df = df.filter(
        F.col("DrugID").isNotNull() &
        F.col("TargetID").isNotNull() &
        F.col("RawScore").isNotNull() &
        (F.col("RawScore") > 0)
    )

    # -------------------------------------------------------
    # Bioinformatics: p-scale transformation for concentration
    # measures (IC50, Kd, Ki, EC50): lower nM → stronger binding
    # p-value = 9 - log10(nM)  →  higher score = stronger binding
    # Davis/KIBA affinity scores are already "higher is better"
    # -------------------------------------------------------
    if dataset_name in ["IC50", "Kd", "Ki", "EC50"]:
        df = df.withColumn("AffinityScore", F.lit(9.0) - F.log10(F.col("RawScore")))
        df = df.withColumn("AffinityType",  F.lit(f"p{dataset_name}"))
    else:
        df = df.withColumn("AffinityScore", F.col("RawScore"))
        df = df.withColumn("AffinityType",  F.lit(dataset_name))

    # -------------------------------------------------------
    # Min-Max normalisation per source → [1, 5] range
    # -------------------------------------------------------
    w  = Window.partitionBy("Source")
    df = (
        df
        .withColumn("MinScore", F.min("AffinityScore").over(w))
        .withColumn("MaxScore", F.max("AffinityScore").over(w))
        .withColumn(
            "NormalizedScore",
            F.when(F.col("MaxScore") == F.col("MinScore"), F.lit(3.0))
             .otherwise(
                 ((F.col("AffinityScore") - F.col("MinScore")) /
                  (F.col("MaxScore") - F.col("MinScore")) * 4.0) + 1.0
             ),
        )
    )

    return df.select(
        "DrugID", "TargetID", "RawScore",
        "AffinityScore", "NormalizedScore", "AffinityType", "Source"
    )


def integrate_affinity_data(spark):
    """Load all 6 sources, union them, then deduplicate by (DrugID, TargetID)."""
    datasets_config = [
        {
            "file": "davis_all.csv", "format": "csv",
            "d_col": "compound_iso_smiles", "t_col": "target_sequence",
            "s_col": "affinity", "name": "Davis"
        },
        {
            "file": "kiba_all.csv", "format": "csv",
            "d_col": "compound_iso_smiles", "t_col": "target_sequence",
            "s_col": "affinity", "name": "KIBA"
        },
        {
            "file": "IC50_bind.tsv", "format": "tsv",
            "d_col": "drug_id", "t_col": "target_id",
            "s_col": "affinity", "name": "IC50"
        },
        {
            "file": "Kd_bind.tsv", "format": "tsv",
            "d_col": "drug_id", "t_col": "target_id",
            "s_col": "affinity", "name": "Kd"
        },
        {
            "file": "Ki_bind.tsv", "format": "tsv",
            "d_col": "drug_id", "t_col": "target_id",
            "s_col": "affinity", "name": "Ki"
        },
        {
            "file": "EC50_bind.tsv", "format": "tsv",
            "d_col": "drug_id", "t_col": "target_id",
            "s_col": "affinity", "name": "EC50"
        },
    ]

    dataframes = []
    for cfg in datasets_config:
        df = process_affinity_dataset(
            spark,
            cfg["file"], cfg["format"],
            cfg["d_col"], cfg["t_col"], cfg["s_col"], cfg["name"],
        )
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        raise RuntimeError("No affinity datasets found in data/raw. Aborting.")

    # Union all sources
    master_df = dataframes[0]
    for df in dataframes[1:]:
        master_df = master_df.unionByName(df)

    total_raw = master_df.count()
    print(f"[INFO] Total rows before deduplication: {total_raw}")

    # -------------------------------------------------------
    # Deduplication: if same (DrugID, TargetID) appears in
    # multiple sources, average the score and keep metadata
    # -------------------------------------------------------
    integrated = (
        master_df
        .groupBy("DrugID", "TargetID")
        .agg(
            F.avg("NormalizedScore").alias("NormalizedScore"),
            F.avg("AffinityScore").alias("AffinityScore"),
            F.first("AffinityType").alias("AffinityType"),
            F.collect_set("Source").alias("Sources"),
            F.countDistinct("Source").alias("SourceCount"),
            F.count("*").alias("EvidenceRows"),
        )
    )

    total_deduped = integrated.count()
    print(f"[INFO] Unique (DrugID, TargetID) pairs after deduplication: {total_deduped}")
    return integrated


def load_string_ppi(spark):
    """
    Load the STRING human PPI network.
    File: 9606.protein.links.v12.0.txt
    combined_score is in range 0-1000 → we divide by 1000 to get [0, 1].
    We keep only edges with confidence >= STRING_CONF_THRESHOLD (0.70).
    """
    string_path = os.path.join(RAW_DATA_DIR, STRING_FILE)
    if not os.path.exists(string_path):
        print(f"[WARN] STRING file not found: {STRING_FILE}. Skipping PPI step.")
        return None

    ppi = spark.read.csv(string_path, header=True, inferSchema=True, sep=" ")
    print(f"[INFO] STRING raw columns: {ppi.columns}")

    if not {"protein1", "protein2", "combined_score"}.issubset(set(ppi.columns)):
        raise ValueError(f"Unexpected STRING file schema. Got columns: {ppi.columns}")

    ppi = (
        ppi.select(
            F.col("protein1").cast("string").alias("ProteinA"),
            F.col("protein2").cast("string").alias("ProteinB"),
            (F.col("combined_score").cast("double") / F.lit(1000.0)).alias("ConfidenceScore"),
        )
        .filter(F.col("ConfidenceScore") >= F.lit(STRING_CONF_THRESHOLD))
        .dropDuplicates(["ProteinA", "ProteinB"])
    )

    edge_count = ppi.count()
    print(f"[INFO] STRING edges kept (confidence >= {STRING_CONF_THRESHOLD}): {edge_count}")
    return ppi


def main():
    spark = create_spark_session()
    try:
        # Step 1: Affinity data fusion
        print("\n=== STEP 1: Affinity Data Integration ===")
        integrated = integrate_affinity_data(spark)
        integrated.show(10, truncate=True)

        integrated.write \
            .format("mongodb") \
            .mode("overwrite") \
            .save()
        print("[SUCCESS] Saved integrated_interactions to MongoDB.")

        # Step 2: STRING PPI network
        print("\n=== STEP 2: STRING PPI Network ===")
        ppi = load_string_ppi(spark)
        if ppi is not None:
            ppi.show(10, truncate=True)
            ppi.write \
                .format("mongodb") \
                .option("spark.mongodb.write.connection.uri", MONGO_STRING_URI) \
                .mode("overwrite") \
                .save()
            print(f"[SUCCESS] Saved string_ppi_edges to MongoDB.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()