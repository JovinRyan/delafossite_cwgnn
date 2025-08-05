from pymatgen.core.periodic_table import Element
from mp_api.client import MPRester
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

def main():
    load_dotenv()
    API_KEY = os.getenv("MP_API_KEY")

    # Get all elements from pymatgen
    all_elements = [el.symbol for el in Element]
    print(f"Total no. of elements considered: {len(all_elements)}")

    # Setup output path
    output_dir = Path("..") / "delafossite_cwgnn" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "element_reference_energies.csv"

    data = []

    with MPRester(API_KEY) as mpr:
        for el in sorted(all_elements):
            try:
                docs = mpr.materials.summary.search(
                    elements=[el],
                    num_elements=1,
                    fields=["material_id", "formula_pretty", "energy_per_atom", "formation_energy_per_atom", "energy_above_hull"]
                )
                if not docs:
                    print(f"No entry found for {el}")
                    continue

                most_stable = sorted(docs, key=lambda d: d.energy_above_hull)[0]

                data.append({
                    "Element": el,
                    "Formula": most_stable.formula_pretty,
                    "Material_ID": most_stable.material_id,
                    "Energy_per_atom_eV": most_stable.energy_per_atom,
                    "Formation_energy_per_atom_eV": most_stable.formation_energy_per_atom,
                    "Energy_above_hull_eV": most_stable.energy_above_hull
                })

            except Exception as e:
                print(f"Error retrieving data for {el}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved elemental reference energies to {output_path}")


if __name__ == "__main__":
    main()
