import pandas as pd
import sys
import os

from delafossite_cwgnn.utils.io import element_reference_energies_to_dict

def get_delafossite_formation_energy(delafossite_entry: str, delafossite_total_E: float,
                                     element_reference_energy_dict: dict, stoich=[3, 3, 6]) -> float:
    A, B, C, _ = delafossite_entry.split("_")
    ref_energy = (
        stoich[0] * element_reference_energy_dict[A] +
        stoich[1] * element_reference_energy_dict[B] +
        stoich[2] * element_reference_energy_dict[C]
    )
    return delafossite_total_E - ref_energy

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_delafossite_formation_energy.py <input_csv_path>")
        sys.exit(1)

    input_csv = sys.argv[1]

    # Generate output file name
    input_name = os.path.basename(input_csv)
    name_part, ext = os.path.splitext(input_name)
    output_csv = os.path.join(os.path.dirname(input_csv), name_part + "_with_formationE" + ext)

    # Load data
    df = pd.read_csv(input_csv, usecols=["Src", "TotalE(eV)"])
    element_ref = element_reference_energies_to_dict()

    # Compute formation energies
    formation_energies = []
    for idx, row in df.iterrows():
        try:
            entry = row["Src"]
            total_E = row["TotalE(eV)"]
            formation_E = get_delafossite_formation_energy(entry, total_E, element_ref)
            formation_energies.append(formation_E)
        except Exception as e:
            print(f"Skipping row {idx} ({entry}): {e}")
            formation_energies.append(None)

    # Add and save
    df["FormationE(eV)"] = formation_energies
    df.to_csv(output_csv, index=False)
    print(f"Saved formation energies to: {output_csv}")

if __name__ == "__main__":
    main()
