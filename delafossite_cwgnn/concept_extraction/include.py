from pymatgen.core import Structure
import numpy as np
import warnings
from collections import Counter
from mendeleev import element
from functools import lru_cache

def extract_AB_rel_features(structure: Structure, A: str, B: str, C: str, structure_type: str, element_dict: dict)->dict:
    """
    Extracts relative elemental features between A and B elements based on physical properties.

    Args:
        structure (Structure): Pymatgen structure object (not used directly here).
        A (str): Symbol of element A (e.g., 'Cu').
        B (str): Symbol of element B (e.g., 'Fe').
        C (str): Symbol of element C (not used here).
        structure_type (str): Type of structure (not used here).
        element_dict (dict): Dictionary mapping element symbols to their properties.

    Returns:
        dict: Dictionary with keys:
            - 'AB_AtomicRadius'
            - 'AB_Electronegativity'
            - 'AB_ElectronAffinity'
            - 'AB_IonizationEnergy'
    """

    try:
        AB_AtomicRadius = element_dict[A]["AtomicRadius"] / element_dict[B]["AtomicRadius"]
    except (KeyError, ZeroDivisionError) as e:
        warnings.warn(f"Error computing AB_AtomicRadius for {A}/{B}: {e}")
        AB_AtomicRadius = 0.0

    try:
        AB_Electronegativity = element_dict[A]["Electronegativity"] / element_dict[B]["Electronegativity"]
    except (KeyError, ZeroDivisionError) as e:
        warnings.warn(f"Error computing AB_Electronegativity for {A}/{B}: {e}")
        AB_Electronegativity = 0.0

    # try:
    #     AB_ElectronAffinity = element_dict[A]["ElectronAffinity"] / element_dict[B]["ElectronAffinity"] # Too many missing/zero datapoints
    # except (KeyError, ZeroDivisionError) as e:
    #     warnings.warn(f"Error computing AB_ElectronAffinity for {A}/{B}: {e}")
    #     AB_ElectronAffinity = 0.0

    try:
        AB_IonizationEnergy = element_dict[A]["IonizationEnergy"] / element_dict[B]["IonizationEnergy"]
    except (KeyError, ZeroDivisionError) as e:
        warnings.warn(f"Error computing AB_IonizationEnergy for {A}/{B}: {e}")
        AB_IonizationEnergy = 0.0

    return {
        "AB_AtomicRadius": AB_AtomicRadius,
        "AB_Electronegativity": AB_Electronegativity,
        # "AB_ElectronAffinity": AB_ElectronAffinity,
        "AB_IonizationEnergy": AB_IonizationEnergy
    }

def extract_AB_magnetic_classification(structure: Structure, A: str, B: str, C: str, structure_type: str, element_dict: dict)->dict:
    """
    Returns bool encoded magnetic property indicators for elements A and B.

    Magnetic properties are classified into:
        - Ferromagnetic (FM)
        - Antiferromagnetic (AFM)
        - Paramagnetic (PM)
        - Diamagnetic (DM)

    Args:
        structure (Structure): The crystal structure object (not used in this function but kept for consistency).
        A (str): The chemical symbol for element A.
        B (str): The chemical symbol for element B.
        C (str): The chemical symbol for element C (not used here).
        structure_type (str): The type of structure (not used here).
        element_dict (dict): A dictionary of element properties (not used here).

    Returns:
        dict: Bool encoded magnetic property indicators for A and B.
    """

    # Element classifications by magnetic property
    ferromagnetic = {"Fe", "Co", "Ni", "Gd"}
    antiferromagnetic = {"Cr"}
    paramagnetic = {
        "Li", "O", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Mn",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Th", "Pa", "U", "Pu", "Am"
    }
    diamagnetic = {
        "H", "He", "Be", "B", "C", "N", "Ne", "Si", "P", "S", "Cl", "Ar",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe", "Au", "Hg", "Tl", "Pb", "Bi"
    }

    # Elements with unclear or unknown magnetic classification
    unknown_magnetic = {
        "F", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Np", "Cm", "Bk", "Cf", "Es",
        "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Pm"
    }

    # Bool encode magnetic properties for A and B
    return {
        "A_is_FM": int(A in ferromagnetic),
        "A_is_AFM": int(A in antiferromagnetic),
        "A_is_PM": int(A in paramagnetic),
        "A_is_DM": int(A in diamagnetic),
        "B_is_FM": int(B in ferromagnetic),
        "B_is_AFM": int(B in antiferromagnetic),
        "B_is_PM": int(B in paramagnetic),
        "B_is_DM": int(B in diamagnetic)
    }

metal_categories = {
    'Alkali metals',
    'Alkaline earth metals',
    'Transition metals',
    'Poor metals',
    'Metalloids'
}

@lru_cache(maxsize=None)
def get_series(symbol: str) -> str:
    return element(symbol).series

def extract_AB_metal_classification(structure: Structure, A: str, B: str, C: str, structure_type: str, element_dict: dict) -> dict:
    def classify(symbol: str, prefix: str) -> dict:
        series = get_series(symbol)
        return {
            f"{prefix}_is_alkali": int(series == "Alkali metals"),
            f"{prefix}_is_alkaline_earth": int(series == "Alkaline earth metals"),
            f"{prefix}_is_transition": int(series == "Transition metals"),
            f"{prefix}_is_poor": int(series == "Poor metals"),
            f"{prefix}_is_metalloid": int(series == "Metalloids")
        }

    features = {}
    features.update(classify(A, "A"))
    features.update(classify(B, "B"))
    return features

# Master concept function dictionary
CONCEPT_FUNCTIONS = {
    "AB_rel": extract_AB_rel_features,
    "AB_magnetism": extract_AB_magnetic_classification,
    "AB_metal_category": extract_AB_metal_classification
}

