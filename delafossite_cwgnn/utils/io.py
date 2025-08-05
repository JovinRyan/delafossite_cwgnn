import os
from pymatgen.io.vasp import Poscar
import pandas as pd

def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), '..', 'data', filename)

def parse_Delafossite_POSCAR_entry(POSCAR_file : str) -> dict:
    structure = Poscar.from_file(POSCAR_file).structure # File name of the form: A_B_C_T.vasp

    A, B, C, structure_type = POSCAR_file.split(".")[0].split("/")[-1].split("_")

    return {"structure": structure, "A": A, "B": B, "C": C, "structure_type": structure_type}

def element_reference_energies_to_dict(filename: str = "element_reference_energies.csv") -> dict:
    """
    Load elemental reference energies from a CSV file and return a dictionary
    mapping element symbols to their energy per atom.

    Parameters:
    ----------
    filename : str, optional
        Name of the CSV file containing the elemental reference energies.
        This file must contain at least two columns: "Element" and "Energy_per_atom_eV".
        The file is located using `get_data_path(filename)`.

    Returns:
    -------
    dict
        A dictionary where the keys are element symbols (e.g., "Fe", "O")
        and the values are the corresponding energy per atom in electronvolts (eV).

    Raises:
    ------
    FileNotFoundError
        If the specified CSV file cannot be found by `get_data_path`.
    KeyError
        If the required columns ("Element" and "Energy_per_atom_eV") are not in the file.
    """
    df = pd.read_csv(get_data_path(filename))
    energy_dict = dict(zip(df["Element"], df["Energy_per_atom_eV"]))
    return energy_dict
