import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from pymatgen.io.vasp.inputs import BadPoscarWarning
import warnings
from concurrent.futures import ProcessPoolExecutor

from delafossite_cwgnn.concept_extraction.include import CONCEPT_FUNCTIONS
from delafossite_cwgnn.utils.io import parse_Delafossite_POSCAR_entry
from delafossite_cwgnn.graph_construction.include import elemental_CSV_to_nested_dict

import gc

warnings.filterwarnings("ignore", category=FutureWarning, module=".*dgl.backend.pytorch.sparse.*")
warnings.filterwarnings("ignore", category=BadPoscarWarning)

# Global variables for worker processes
_global_element_dict = None
_global_selected_concepts = None
_global_base_dir = None
_global_target_map = None

def init_worker(element_dict, selected_concepts, base_dir, target_map):
    global _global_element_dict, _global_selected_concepts, _global_base_dir, _global_target_map
    _global_element_dict = element_dict
    _global_selected_concepts = selected_concepts
    _global_base_dir = base_dir
    _global_target_map = target_map

def process_entry(name):
    import traceback
    try:
        vasp_path = os.path.join(_global_base_dir, f"{name}.vasp")
        if not os.path.exists(vasp_path):
            return None

        parsed = parse_Delafossite_POSCAR_entry(vasp_path)
        structure = parsed["structure"]
        A, B, C = parsed["A"], parsed["B"], parsed["C"]
        str_type = parsed["structure_type"]

        features = {}
        for key in _global_selected_concepts:
            if key in CONCEPT_FUNCTIONS:
                features.update(CONCEPT_FUNCTIONS[key](structure, A, B, C, str_type, _global_element_dict))

        features["id"] = name
        features["target"] = _global_target_map.get(name, None)

        del structure
        gc.collect()
        return features

    except Exception:
        print(f"[ERROR] processing {name}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Run concept extraction on VASP structures.")
    parser.add_argument("directory", help="Directory containing the .vasp files and HighThroughput_TM_with_J.csv")
    parser.add_argument("concepts", nargs="*", help="List of concept names to extract")
    parser.add_argument("--target_col", type=str, default=None, help="Column name from CSV to use as target")
    args = parser.parse_args()

    base_dir = args.directory
    selected_concepts = args.concepts if args.concepts else list(CONCEPT_FUNCTIONS.keys())

    print("Active concept functions:", selected_concepts)

    # Validate selected concepts
    invalid = [c for c in selected_concepts if c not in CONCEPT_FUNCTIONS]
    if invalid:
        print(f"Warning: Unrecognized concepts: {invalid}")
    selected_concepts = [c for c in selected_concepts if c in CONCEPT_FUNCTIONS]
    if not selected_concepts:
        print("No valid concepts selected. Exiting.")
        sys.exit(1)

    element_dict = elemental_CSV_to_nested_dict()

    # Read structure names and target column
    csv_path = os.path.join(base_dir, "HighThroughput_TM_with_J.csv")
    df = pd.read_csv(csv_path)
    structure_names = df.iloc[:, 0].tolist()

    # Map id to target value
    target_map = {}
    if args.target_col:
        if args.target_col not in df.columns:
            print(f"Target column '{args.target_col}' not found in CSV.")
            sys.exit(1)
        target_map = dict(zip(df.iloc[:, 0], df[args.target_col]))

    # Config
    max_workers = max(1, os.cpu_count() - 2)
    batch_size = 500

    # Sanitize concept names for filename
    concept_str = "_".join(args.concepts).replace(" ", "").replace(",", "")[:40]
    target_str = args.target_col.replace(" ", "")

    output_filename = f"target_{target_str}_concepts_{concept_str}.csv"
    output_csv = os.path.join(base_dir, output_filename)
    first_write = True
    open(output_csv, 'w').close()

    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=init_worker,
                             initargs=(element_dict, selected_concepts, base_dir, target_map)) as executor:
        for i in range(0, len(structure_names), batch_size):
            batch = structure_names[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} with {len(batch)} structures...")

            results = list(tqdm(executor.map(process_entry, batch), total=len(batch)))
            results = [r for r in results if r]

            if results:
                df_batch = pd.DataFrame(results)

                # Reorder columns: id, target, rest...
                id_col = df_batch.pop("id")
                target_col = df_batch.pop("target") if "target" in df_batch else pd.Series([None] * len(df_batch))
                df_batch.insert(0, "id", id_col)
                df_batch.insert(1, target_str, target_col)

                df_batch.to_csv(output_csv, mode='a', header=first_write, index=False)
                first_write = False

            del results
            gc.collect()

    print(f"All done. Saved to {output_csv}")

if __name__ == "__main__":
    main()
