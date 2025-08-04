import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure

from delafossite_cwgnn.concept_extraction.include import CONCEPT_FUNCTIONS

def read_structure(path):
    return Structure.from_file(path)

def extract_concepts(structure, concept_keys):
    features = {}
    for key in concept_keys:
        if key in CONCEPT_FUNCTIONS:
            features.update(CONCEPT_FUNCTIONS[key](structure))
    return features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <directory> [concept1 concept2 ...]")
        sys.exit(1)

    base_dir = sys.argv[1]
    selected_concepts = sys.argv[2:] if len(sys.argv) > 2 else list(CONCEPT_FUNCTIONS.keys())

    csv_path = os.path.join(base_dir, "HighThroughput_TM_with_J.csv")
    df = pd.read_csv(csv_path)
    structures = df.iloc[:, 0]

    results = []
    for name in tqdm(structures, desc="Extracting Concepts"):
        vasp_path = os.path.join(base_dir, f"{name}.vasp")
        if not os.path.exists(vasp_path):
            continue
        structure = read_structure(vasp_path)
        concepts = extract_concepts(structure, selected_concepts)
        concepts["id"] = name
        results.append(concepts)

    with open(os.path.join(base_dir, "concepts.json"), "w") as f:
        json.dump(results, f, indent=2)
