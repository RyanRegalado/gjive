import pandas as pd
import numpy as np
import requests
from pathlib import Path
import re

DIR = Path().cwd() / "autism"

def download_abide_cc220():

    

    # Load phenotype file
    path =  DIR / "Phenotypic_V1_0b_preprocessed1.csv"
    pheno = pd.read_csv(path)

    # Get all participants with available files
    subjects = pheno[
        pheno["FILE_ID"] != "no_filename"
    ]["FILE_ID"].unique()

    # Create output directory
    output = DIR / "abide_cc200_all"
    output.mkdir(exist_ok=True)

    total = len(subjects)

    print(f"Found {total} subjects.")

    for i, file_id in enumerate(subjects, start=1):

        url = (
            "https://s3.amazonaws.com/"
            "fcp-indi/data/Projects/ABIDE_Initiative/"
            f"Outputs/cpac/filt_global/rois_cc200/"
            f"{file_id}_rois_cc200.1D"
        )

        outfile = output / f"{file_id}_rois_cc200.1D"

        # Skip already downloaded files
        if outfile.exists():
            print(f"[{i}/{total}] Skipping {file_id} (already exists)")
            continue

        try:
            print(f"[{i}/{total}] Downloading {file_id}")

            r = requests.get(url, timeout=60)

            if r.status_code == 200:
                outfile.write_bytes(r.content)
            else:
                print(
                    f"Failed {file_id}: HTTP {r.status_code}"
                )

        except Exception as e:
            print(f"Error downloading {file_id}: {e}")

    print("Download complete.")

def assign_subject_ids():

    # 1. Setup paths
    cc220_dir = DIR / "abide_cc200_all"
    output_file = DIR / "csvs/all_subjects_connectivity_long.csv"
    pheno_file = DIR / "csvs/Phenotypic_V1_0b_preprocessed1.csv"
    labels_file = DIR / "csvs/CC200_ROI_labels.csv"

    # Load metadata
    pheno_df = pd.read_csv(pheno_file)
    labels_df = pd.read_csv(labels_file)
    region_labels = labels_df['AAL'].tolist()

    # Create a mapping: { 50003: 50003 }
    # We map the SUB_ID to itself so we can easily look it up by integer
    id_map = dict(zip(pheno_df['SUB_ID'], pheno_df['SUB_ID']))

    all_data = []

    # 2. Loop through files
    for file_path in sorted(cc220_dir.glob("*.1D")):
        raw_name = file_path.stem
    
        # Use regex to find the digits (SUB_ID) in the filename
        # This looks for the sequence of digits that appears before '_rois'
        match = re.search(r'(\d+)_rois', raw_name)
    
        if match:
            extracted_id = int(match.group(1)) # Convert "0051201" to integer 51201
        
            # Check if this ID exists in our Phenotypic CSV
            if extracted_id in id_map:
                subject_id = extracted_id
            
                # Process the file
                df = pd.read_csv(file_path, sep=None, engine='python', header=None, comment='#')
                corr_matrix = df.corr()
            
                corr_matrix.columns = region_labels
                corr_matrix.index = region_labels
            
                # Convert to "Long Format"
                long_df = corr_matrix.stack().reset_index()
                long_df.columns = ['Region_A', 'Region_B', 'Correlation']
                long_df['Subject_ID'] = subject_id
            
                all_data.append(long_df)
                print(f"Successfully processed: {raw_name} -> ID: {subject_id}")
            else:
                print(f"Skipping: {raw_name} | ID {extracted_id} not in phenotypic file.")
        else:
            print(f"Skipping: {raw_name} | Could not extract ID.")

    # 3. Combine and save
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Success! Saved to {output_file}")
    else:
        print("Error: No data was collected. Check your file paths or matching logic.")

if __name__ == "__main__":
    assign_subject_ids()