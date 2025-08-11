import delafossite_cwgnn.graph_construction.include as include
import sys
import os
import pandas as pd
import warnings
import concurrent.futures
from dgl import save_graphs
from tqdm import tqdm
from pymatgen.io.vasp.inputs import BadPoscarWarning

warnings.filterwarnings("ignore", category=BadPoscarWarning)

def process_entry(entry_data):
    entry, directory_path, graph_save_dir, elemental_dict = entry_data
    vasp_path = os.path.join(directory_path, entry + ".vasp")
    save_path = os.path.join(graph_save_dir, entry + ".bin")

    try:
        if not os.path.isfile(vasp_path):
            return f"Skipped (missing): {entry}"

        supercell = include.POSCAR_to_supercell(vasp_path, supercell_matrix=[1, 1, 1])  # primitive cell
        g = include.structure_to_dgl_graph(supercell, elemental_dict)
        save_graphs(save_path, [g])
        return f"Success: {entry}"

    except Exception as e:
        return f"Failed: {entry} ({str(e)})"


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) > 1:
        directory_path = argv[1]
        csv_filename = "target_structure_type_concepts_triplet_angles.csv"
        csv_path = os.path.join(directory_path, csv_filename)
        graph_save_dir = os.path.join(directory_path, "dgl_graphs")

        if not os.path.isfile(csv_path):
            print("CSV file not found.")
            sys.exit(1)

        print(f"Found CSV file: {csv_path}")
        delafossite_data = pd.read_csv(csv_path)
        delafossite_data = delafossite_data.drop_duplicates(delafossite_data.columns[0], keep='first')
        print("CSV file contains", len(delafossite_data), "entries.")

        os.makedirs(graph_save_dir, exist_ok=True)
        elemental_dict = include.elemental_CSV_to_nested_dict()  # defaults to your data folder

        entry_args = [
            (entry, directory_path, graph_save_dir, elemental_dict)
            for entry in delafossite_data.iloc[:, 0]
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            results = list(tqdm(
                executor.map(process_entry, entry_args),
                total=len(entry_args),
                desc="Processing structures"
            ))

        for res in results:
            print(res)

    else:
        print("Usage: python main.py <directory_path>")

if __name__ == "__main__":
    main()
