# 🧬 Drug–Target Interaction Explorer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-orange.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3b82f6.svg)

## Project Overview

This project is a **Drug–Target Interaction (DTI) recommendation system** developed for the **Big Data Management** course. It combines **Apache Spark**, **MongoDB**, and a **Streamlit analytics dashboard** to predict possible drug–target interactions and present them through an interactive bioinformatics explorer.[file:2]

The main idea is to model the DTI problem as a **recommendation task**. Instead of manually testing all possible drug–target pairs, the system uses **Spark ALS (Alternating Least Squares)** to learn latent patterns from integrated biological interaction datasets and recommend promising drug candidates for selected targets.[file:2]

In the current version, the project also includes:
- a **Recommendations** tab for top predicted compounds,
- an **Interaction Network** tab with target–drug and STRING PPI context,
- a **Chemistry** tab for compound structure and physicochemical properties,
- a **Disease Context** tab powered by DisGeNET,
- a **Dataset Analytics** tab showing pipeline and source-level visual summaries.[file:2]

## Objectives

The goals of this project are:

- Build an end-to-end **Big Data pipeline** for drug–target interaction analysis.
- Integrate multiple biological datasets into one unified interaction table.
- Train a recommendation model using **PySpark MLlib ALS**.
- Store processed data and predictions in **MongoDB** for efficient retrieval.
- Build an interactive **Streamlit** interface for exploration and presentation.[file:2]

## System Workflow

### 1. Data integration
Multiple DTI-related datasets are collected and merged into a common structure. In the current dashboard, the integrated pipeline is described as combining **6 DTI datasets** with p-scale normalization across **IC50 / Kd / Ki / EC50** measures.[file:2]

### 2. Data preprocessing
Raw affinity values are transformed into a comparable **p-scale affinity score**, then normalized for use in the recommendation model. The dashboard also reflects this transformation visually in the dataset analytics section.[file:2]

### 3. Storage in MongoDB
Processed targets, interaction records, predictions, and network-related information are stored in **MongoDB**, which acts as the serving database for the dashboard and model outputs.[file:2]

### 4. Model training with Spark ALS
The project uses **Apache Spark MLlib ALS** to learn latent factors for drugs and targets and generate ranked recommendations. The current dashboard reports **RMSE = 0.2162** and **AUPR = 0.9985** for the model shown in the analytics view.[file:2]

### 5. Interactive exploration
A **Streamlit-based web application** lets the user choose a target and inspect predictions, local interaction networks, compound chemistry, disease associations, and dataset-level graphs in one place.[file:2]

## Current Dashboard Features

The current application interface includes the following major sections:[file:2]

### Recommendations
Shows the top predicted drugs for a selected target, together with prediction scores and export support.[file:2]

### Interaction Network
Visualizes:
- the selected target,
- predicted drugs from the ALS recommender,
- STRING protein–protein interaction neighbors when available.[file:2]

### Chemistry
Displays the lead compound structure and molecular properties when a valid structure can be resolved from the candidate identifiers.[file:2]

### Disease Context
Uses **DisGeNET** to enrich the selected target with human gene–disease association context, if the selected target can be mapped appropriately.[file:2]

### Dataset Analytics
Provides a compact overview of the data pipeline, including:
- integrated DTI pairs by source,
- p-scale transformation concept,
- Spark ALS pipeline steps,
- score distribution for the selected target.[file:2]

## Example Metrics

From the currently uploaded analytics view, the system reports:[file:2]

- **Unique targets:** 500 [file:2]
- **Integrated DTI pairs:** about 1.707 million [file:2]
- **STRING edges:** 473,860 [file:2]
- **Model RMSE:** 0.2162 [file:2]
- **Model AUPR:** 0.9985 [file:2]

These values may change depending on the exact datasets loaded and the version of the processed database used.[file:2]

## Technologies Used

- **Python**
- **Apache Spark / PySpark**
- **Spark MLlib (ALS)**
- **MongoDB**
- **Streamlit**
- **Plotly**
- **Pandas**
- **NumPy**
- **PyVis**
- **RDKit** for chemistry calculations and rendering
- **Requests / python-dotenv** for API integration and environment configuration

## Project Structure

A typical structure for this project is:

```text
DrugTarget-Recommender-Spark/
│
├── data/                         # Raw and processed datasets
├── src/
│   ├── database/                 # MongoDB access layer
│   ├── services/                 # UniProt, STRING, PubChem, DisGeNET, chemistry services
│   ├── frontend/                 # Streamlit app
│   ├── preprocessing/            # Cleaning, transformation, integration scripts
│   └── modeling/                 # Spark ALS training and recommendation logic
│
├── .env                          # API keys and local configuration
├── requirements.txt
└── README.md
```

Adjust this section if your exact folders are slightly different.

## Installation

### Prerequisites

Before running the project, make sure you have:

- **Python 3.10+**
- **Java 8 or 11** for Apache Spark
- **MongoDB Community Server** running locally on port `27017`
- Optional but recommended: a virtual environment

### Clone the repository

```bash
git clone https://github.com/SaeidTadrisi/DrugTarget-Recommender-Spark.git
cd DrugTarget-Recommender-Spark
```

### Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
DISGENET_API_KEY=your_token_here
MONGO_URI=mongodb://localhost:27017/
```

The dashboard sidebar also reminds the user how to set up the DisGeNET token.[file:2]

## How to Run

Because project structures differ slightly between versions, adapt the commands to your filenames if needed.

### Step 1 — Start MongoDB
Make sure your MongoDB server is running locally.

### Step 2 — Run preprocessing / loading scripts
Run your data loading and preprocessing scripts to populate MongoDB with:
- integrated DTI records,
- prediction-ready mappings,
- STRING PPI data if available,
- model outputs.

Typical examples:

```bash
python src/preprocessing/mongodb_loader.py
python src/modeling/als_recommender.py
```

### Step 3 — Start the Streamlit app

```bash
streamlit run src/frontend/app.py
```

Then open the local URL shown by Streamlit in your browser.

## Notes on Chemistry and External APIs

### Chemistry tab
Some candidate identifiers may not directly resolve to valid structures. In that case, the Chemistry tab may show a message indicating that no valid molecular structure was retrieved.

### STRING fallback
If no local PPI neighbors exist in MongoDB for a selected target, the app may use a live STRING fallback service depending on your implementation.

### DisGeNET
Disease enrichment depends on:
- a valid API key,
- a human target or a valid human gene mapping,
- API availability.

## Dataset Note

The raw source datasets are not always included in the repository because of size or licensing constraints. If needed, place the required files in the `data/` directory before running the pipeline.

## Academic Context

This project was developed as part of the **Big Data Management** course and demonstrates how Big Data tools can be applied to a real bioinformatics recommendation problem involving large-scale biological interaction data.

## Future Improvements

Possible next steps include:

- stronger compound identifier resolution,
- richer chemistry coverage,
- improved network layouts and graph libraries,
- more benchmarking against alternative recommenders,
- more export and reporting options,
- more advanced analytics across all tabs.

## Author

**Saeid Tadrisi**

GitHub: [SaeidTadrisi](https://github.com/SaeidTadrisi)