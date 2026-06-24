from pathlib import Path
import pandas as pd
import numpy as np
import random

# --- Robust Path Configuration (Absolute & OS-Independent) ---
# __file__ refers to this script: .../src/data_generation/clinical_cohort_simulator.py
# .resolve().parent.parent.parent goes strictly 3 levels up to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
GENERATED_DATA_DIR = PROJECT_ROOT / "data" / "generated"

DAVIS_FILE = RAW_DATA_DIR / "davis_all.csv"
OUTPUT_FILE = GENERATED_DATA_DIR / "patient_drug_interactions.csv"


def create_directories():
    """Ensure that the output directories exist before saving data."""
    if not GENERATED_DATA_DIR.exists():
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {GENERATED_DATA_DIR}")


def read_real_data():
    """
    Load real drug-target interaction data from the downloaded Kaggle dataset.
    Returns a pandas DataFrame.
    """
    print(f"Loading real data from:\n   {DAVIS_FILE}")

    if not DAVIS_FILE.exists():
        print(f"\n CRITICAL ERROR: The file literally does not exist at the resolved absolute path:\n   {DAVIS_FILE}")
        print(
            "Windows Hint: Check your raw folder to make sure Windows didn't hide the extension (e.g., naming it 'davis_all.csv.csv')")
        exit(1)

    try:
        df_davis = pd.read_csv(DAVIS_FILE)
        print(f"Successfully loaded {len(df_davis)} real interactions.")

        # Display column names to verify the structure
        columns = df_davis.columns.tolist()
        print(f"Columns detected: {columns}")

        return df_davis
    except Exception as e:
        print(f"Failed to parse the CSV file. Error details: {e}")
        exit(1)


def generate_in_silico_cohort(df_davis, num_patients=1000):
    """
    Generate simulated (in-silico) patient data based on real drug-target interactions.
    Simulates personalized medicine scenarios for the ALS recommender system.
    """
    print(f"🧪 Generating In-Silico Cohort for {num_patients} patients based on real targets...")

    col_drug = df_davis.columns[0]
    col_target = df_davis.columns[1]

    unique_drugs = df_davis[col_drug].unique().tolist()
    unique_targets = df_davis[col_target].unique().tolist()

    interactions = []

    for i in range(1, num_patients + 1):
        patient_id = f"PT{str(i).zfill(5)}"
        patient_targets = random.sample(unique_targets, random.randint(2, 5))
        effective_drugs = df_davis[df_davis[col_target].isin(patient_targets)][col_drug].tolist()

        if effective_drugs:
            prescribed_good_drugs = random.sample(effective_drugs, min(len(effective_drugs), random.randint(3, 8)))
            for drug in prescribed_good_drugs:
                interactions.append({
                    "PatientID": patient_id,
                    "DrugID": drug,
                    "ReactionScore": random.randint(4, 5)
                })

        random_drugs = random.sample(unique_drugs, random.randint(2, 5))
        for drug in random_drugs:
            if drug not in effective_drugs:
                interactions.append({
                    "PatientID": patient_id,
                    "DrugID": drug,
                    "ReactionScore": random.randint(1, 3)
                })

    df_cohort = pd.DataFrame(interactions)
    return df_cohort


def main():
    """Main execution flow for data generation."""
    create_directories()

    df_davis = read_real_data()
    df_cohort = generate_in_silico_cohort(df_davis, num_patients=2000)

    df_cohort.to_csv(OUTPUT_FILE, index=False)

    print(f"🎉 Success! Generated {len(df_cohort)} scientific patient-drug records.")
    print(f"📄 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()