# 🧬 Drug-Target Interaction Recommender System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.0+-orange.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-green.svg)
![Machine Learning](https://img.shields.io/badge/MLlib-ALS-red.svg)

##  Project Overview
This project is developed as the final assignment for the **Big Data Management** course. It aims to predict potential and novel interactions between drugs and protein targets using Big Data technologies. 

By framing this bioinformatics challenge as a **Collaborative Filtering** problem, we utilize the **Alternating Least Squares (ALS)** algorithm to recommend new drug-target associations. To handle the large-scale nature of biological data and simulate a real-world Data Engineering pipeline, the project integrates **MongoDB** for data storage and **Apache Spark** for distributed machine learning processing.

##  Architecture & Pipeline
1. **Data Sourcing:** Biological data is extracted from well-known databases such as **DisGeNET** (Disease-Gene associations) and **DrugBank** (Drug-Target associations).
2. **Data Generation:** A mock dataset of virtual patients with specific genetic profiles is generated to simulate real-world scenarios.
3. **ETL Pipeline:** Data is pre-processed, cleaned, and loaded into a **MongoDB** NoSQL database.
4. **Machine Learning:** **Apache Spark (PySpark)** connects directly to MongoDB, loads the large datasets into DataFrames, and trains an ALS recommendation model.

##  Technologies Used
* **Language:** Python
* **Big Data Processing:** Apache Spark, PySpark, Spark MLlib
* **Database:** MongoDB
* **Data Manipulation:** Pandas, NumPy

##  How to Run the Project

### Prerequisites
* Java 8 or 11 (Required for Apache Spark)
* Python 3.8+
* MongoDB Community Server running locally (port `27017`)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/SaeidTadrisi/DrugTarget-Recommender-Spark.git](https://github.com/SaeidTadrisi/DrugTarget-Recommender-Spark.git)
   cd YourRepositoryName
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the required dependencies:

Bash
pip install -r requirements.txt
Execution
(Instructions on which Python scripts to run will be added here as the project develops, e.g., python data_generator.py followed by python spark_recommender.py)

Dataset Note
Due to the large size of the datasets, the raw .csv files are not included in this repository. Please download the datasets directly from DisGeNET and DrugBank, and place them in the /data directory before running the pipeline.
