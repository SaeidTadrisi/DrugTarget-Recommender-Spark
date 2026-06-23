import pandas as pd
import numpy as np
import random
import os

# Path configurations (Clean Architecture)
RAW_DATA_DIR = "data/raw"
GENERATED_DATA_DIR = "data/generated"
DAVIS_FILE = os.path.join(RAW_DATA_DIR, "davis_all.csv")
OUTPUT_FILE = os.path.join(GENERATED_DATA_DIR, "patient_drug_interactions.csv")


def create_directories():
    """Ensure that the output directories exist before saving data."""
    if not os.path.exists(GENERATED_DATA_DIR):
        os.makedirs(GENERATED_DATA_DIR)
        print(f"📁 Created directory: {GENERATED_DATA_DIR}")


def read_real_data():
    """
    Load real drug-target interaction data from the downloaded Kaggle dataset.
    Returns a pandas DataFrame.
    """
    print(f"📥 Loading real data from {DAVIS_FILE}...")
    try:
        df_davis = pd.read_csv(DAVIS_FILE)
        print(f"✅ Successfully loaded {len(df_davis)} real interactions.")

        # Display column names to verify the structure
        columns = df_davis.columns.tolist()
        print(f"📊 Columns detected: {columns}")

        return df_davis
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DAVIS_FILE}. Please check the path.")
        exit(1)


def generate_in_silico_cohort(df_davis, num_patients=1000):
    """
    Generate simulated (in-silico) patient data based on real drug-target interactions.
    Simulates personalized medicine scenarios for the ALS recommender system.
    """
    print(f"🧪 Generating In-Silico Cohort for {num_patients} patients based on real targets...")

    # Dynamically extract column names for drug and target
    # Assuming column 0 is Drug (e.g., SMILES/Compound) and column 1 is Target (Protein)
    col_drug = df_davis.columns[0]
    col_target = df_davis.columns[1]

    unique_drugs = df_davis[col_drug].unique().tolist()
    unique_targets = df_davis[col_target].unique().tolist()

    interactions = []

    # Generate a genetic/protein profile for each virtual patient
    for i in range(1, num_patients + 1):
        patient_id = f"PT{str(i).zfill(5)}"

        # Assume each patient has 2 to 5 overexpressed target proteins (disease condition)
        patient_targets = random.sample(unique_targets, random.randint(2, 5))

        # Find drugs that actually interact with these specific targets in the real dataset
        effective_drugs = df_davis[df_davis[col_target].isin(patient_targets)][col_drug].tolist()

        # If effective drugs exist, assign a high reaction score (4 or 5)
        if effective_drugs:
            prescribed_good_drugs = random.sample(effective_drugs, min(len(effective_drugs), random.randint(3, 8)))
            for drug in prescribed_good_drugs:
                interactions.append({
                    "PatientID": patient_id,
                    "DrugID": drug,
                    "ReactionScore": random.randint(4, 5)  # High binding affinity = Positive reaction
                })

        # Prescribe random drugs (not interacting with patient's targets) to simulate neutral/negative reactions
        # This is crucial for training the ALS algorithm to differentiate between good and bad recommendations
        random_drugs = random.sample(unique_drugs, random.randint(2, 5))
        for drug in random_drugs:
            if drug not in effective_drugs:
                interactions.append({
                    "PatientID": patient_id,
                    "DrugID": drug,
                    "ReactionScore": random.randint(1, 3)  # Low/No binding affinity = Negative/Neutral reaction
                })

    df_cohort = pd.DataFrame(interactions)
    return df_cohort


def main():
    """Main execution flow for data generation."""
    create_directories()

    # 1. Read the real dataset (Davis)
    df_davis = read_real_data()

    # 2. Generate the clinical cohort simulation
    df_cohort = generate_in_silico_cohort(df_davis, num_patients=2000)

    # 3. Save the generated cohort for the PySpark pipeline
    df_cohort.to_csv(OUTPUT_FILE, index=False)

    print(f"🎉 Success! Generated {len(df_cohort)} scientific patient-drug records.")
    print(f"📄 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()