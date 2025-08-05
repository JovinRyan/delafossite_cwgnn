import os
from pymatgen.io.vasp import Poscar
import pandas as pd

def get_data_path(filename):
    """
    Returns the absolute path to a data file located in the project's data directory.

    Args:
        filename (str): The name of the file (e.g., "myfile.csv").

    Returns:
        str: Absolute path to the file located in the ../data/ directory relative to this script.
    """
    return os.path.join(os.path.dirname(__file__), '..', 'data', filename)

def parse_Delafossite_POSCAR_entry(POSCAR_file : str) -> dict:
    """
    Parses a delafossite POSCAR file and extracts structure information along with elemental labels.

    The file is expected to follow the naming convention: A_B_C_T.vasp, where:
        - A, B, and C are elemental symbols
        - T is the structure type (e.g., 'T', 'D', 'C')

    Args:
        POSCAR_file (str): Full path to the POSCAR file.

    Returns:
        dict: A dictionary containing:
            - "structure" (pymatgen.Structure): Parsed structure from the POSCAR
            - "A" (str): Symbol of the A-site element
            - "B" (str): Symbol of the B-site element
            - "C" (str): Symbol of the C-site element
            - "structure_type" (str): Structure type identifier (e.g., 'ABC')
    """
    structure = Poscar.from_file(POSCAR_file).structure # File name of the form: A_B_C_T.vasp

    A, B, C, structure_type = POSCAR_file.split(".")[0].split("/")[-1].split("_")

    return {"structure": structure, "A": A, "B": B, "C": C, "structure_type": structure_type}

def element_reference_energies_to_dict(filename: str = "element_reference_energies.csv") -> dict:
    """
    Load elemental reference energies from a CSV file and return a dictionary
    mapping element symbols to their energy per atom.

    Args:
        filename (str): Name of the CSV file containing the elemental reference energies.
            This file must contain at least two columns: "Element" and "Energy_per_atom_eV".
            The file is located using `get_data_path(filename)`.

    Returns:
        dict: A dictionary where keys are element symbols (e.g., "Fe", "O") and values are
            the corresponding energy per atom in electronvolts (eV).

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found by `get_data_path`.
        KeyError: If the required columns ("Element" and "Energy_per_atom_eV") are not in the file.
    """
    df = pd.read_csv(get_data_path(filename))
    energy_dict = dict(zip(df["Element"], df["Energy_per_atom_eV"]))
    return energy_dict
