import numpy as np
import pandas as pd
import re
import os
import dgl
import torch
import sys
from collections import defaultdict
from pymatgen.io.vasp import Poscar
from scipy.signal import find_peaks
from pymatgen.core import Structure
from matminer.featurizers.structure import RadialDistributionFunction
from sklearn.preprocessing import LabelEncoder
from mendeleev import element

from delafossite_cwgnn.utils.io import get_data_path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

os.environ['DGLBACKEND'] = 'pytorch'

def POSCAR_to_supercell(input_POSCAR_file: str, supercell_matrix = [1, 1, 1]):
  structure = Poscar.from_file(input_POSCAR_file).structure

  return structure.make_supercell(supercell_matrix)


def compute_per_atom_rdfs(structure: Structure, cutoff=10.0, bin_size=0.05):
    """
    Computes per-atom RDFs for a given pymatgen structure.

    Args:
        structure (Structure): Pymatgen structure object (can be supercell).
        cutoff (float): Maximum distance to consider (Å).
        bin_size (float): Size of each RDF bin (Å).

    Returns:
        r_values (np.ndarray): Bin centers for RDF.
        rdfs (np.ndarray): Shape [n_atoms, n_bins] — per-atom RDFs.
    """
    n_atoms = len(structure)
    n_bins = int(cutoff / bin_size)
    r_values = np.linspace(0, cutoff, n_bins)
    rdfs = np.zeros((n_atoms, n_bins))

    # Precompute all neighbors with periodic images
    all_neighbors = structure.get_all_neighbors(r=cutoff, include_index=True)

    for i in range(n_atoms):
        neighbors = all_neighbors[i]
        distances = [dist for site, dist, j, image in neighbors]
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, cutoff))
        rdfs[i] = hist

    # Normalize RDFs (optional)
    shell_volumes = 4/3 * np.pi * (
        np.power(np.linspace(bin_size, cutoff, n_bins), 3) -
        np.power(np.linspace(0, cutoff - bin_size, n_bins), 3)
    )
    avg_density = len(structure) / structure.volume
    normalization = avg_density * shell_volumes
    rdfs = rdfs / normalization[np.newaxis, :]

    return r_values, rdfs

def get_element_cutoff_dict_from_rdf(
    structure: Structure,
    cutoff=10.0,
    bin_size=0.05,
    padding=0.1  # Å padding to second peak
):
    """
    Estimates element-specific cutoff distances using the second peak of the
    Radial Distribution Function (RDF) for atoms in a structure.

    This function computes per-atom RDFs for a given structure and identifies the
    second RDF peak (or first, if only one is found) for each element type.
    It adds a small padding to this peak to generate a cutoff distance useful
    for defining local environments (e.g., in graph construction).

    Parameters:
    -----------
    structure : pymatgen.core.structure.Structure
        A Pymatgen Structure object (typically a supercell) for which the RDF will be computed.
    cutoff : float, optional
        Maximum radius (in Å) up to which the RDF is calculated. Default is 10.0 Å.
    bin_size : float, optional
        Bin size (in Å) used for RDF histogram discretization. Default is 0.05 Å.
    padding : float, optional
        Additional distance (in Å) added to the detected second RDF peak. Default is 0.1 Å.

    Returns:
    --------
    cutoff_dict : dict[str, float]
        A dictionary mapping each unique element symbol in the structure to a
        corresponding cutoff distance (in Å), estimated from the RDF.
        For elements with no detectable peaks, the default `cutoff` value is used.
    """

    # Compute per-atom RDFs
    rdf_calc = RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
    r_vals, per_atom_rdfs = compute_per_atom_rdfs(structure, cutoff=cutoff, bin_size=bin_size)

    # Group RDFs by element
    elements = [site.specie.symbol for site in structure]
    rdf_groups = defaultdict(list)
    for i, elem in enumerate(elements):
        rdf_groups[elem].append(per_atom_rdfs[i])

    # Compute mean RDFs
    mean_rdfs = {elem: np.mean(rdfs, axis=0) for elem, rdfs in rdf_groups.items()}

    # Find 2nd RDF peak + padding
    cutoff_dict = {}
    for elem, rdf_curve in mean_rdfs.items():
        peaks, _ = find_peaks(rdf_curve)
        if len(peaks) >= 2:
            second_peak = r_vals[peaks[1]]
        elif len(peaks) == 1:
            second_peak = r_vals[peaks[0]]
        else:
            second_peak = cutoff  # fallback

        cutoff_dict[elem] = second_peak + padding

    return cutoff_dict

def encode_ElectronConfiguration(electronic_configuration: str):
    """
    Parses an electronic configuration string to calculate the number of valence electrons
    and their distribution among s, p, d, and f orbitals.

    This function removes bracketed terms (e.g., [Ar]) and optional content (e.g., in parentheses),
    then tokenizes the remaining orbital configurations (e.g., "4s2", "3d10").
    It builds a dictionary mapping each principal quantum number (n) to the number of electrons
    in each orbital type. It uses the highest occupied shells to estimate the valence electron count.

    Args:
        electronic_configuration (str): Electronic configuration string, such as
            "1s2 2s2 2p6 3s2 3p6 4s1" or "[Ar] 4s1".

    Returns:
        tuple: A 5-element tuple containing:
            - total valence electrons (int)
            - number of electrons in s orbitals of outermost shell (int)
            - number of electrons in p orbitals of outermost shell (int)
            - number of electrons in d orbitals of penultimate shell (int)
            - number of electrons in f orbitals of antepenultimate shell (int)
    """
    clean_config = re.sub(r'\[.*?\]', '', electronic_configuration)
    clean_config = re.sub(r'\(.*?\)', '', clean_config)
    tokens = clean_config.strip().split()

    shells = defaultdict(lambda: {'s': 0, 'p': 0, 'd': 0, 'f': 0})
    all_ns = set()

    for token in tokens:
        match = re.match(r'(\d+)([spdf])(\d+)', token)
        if match:
            n, orb, count = match.groups()
            n = int(n)
            count = int(count)
            shells[n][orb] += count
            all_ns.add(n)

    if not all_ns:
        return (0, 0, 0, 0, 0)

    max_n = max(all_ns)

    # Collect relevant shells
    s = shells[max_n]['s']
    p = shells[max_n]['p']
    d = shells[max_n - 1]['d']
    f = shells[max_n - 2]['f']

    valence_electrons = s + p + d + f

    return (valence_electrons, s, p, d, f)

def count_unpaired_electrons(s, p, d, f):
    def unpaired(n_electrons, orbital_count):
        max_electrons = 2 * orbital_count
        if n_electrons > max_electrons:
            raise ValueError(f"Invalid electron count {n_electrons} for {orbital_count} orbitals")
        if n_electrons <= orbital_count:
            return n_electrons  # each electron occupies a separate orbital (Hund's rule)
        return max_electrons - n_electrons  # electrons start pairing up

    s_unpaired = unpaired(s, 1)
    p_unpaired = unpaired(p, 3)
    d_unpaired = unpaired(d, 5)
    f_unpaired = unpaired(f, 7)

    return s_unpaired + p_unpaired + d_unpaired + f_unpaired

def get_count_unpaired_electrons(element_dict : dict, element):
    s = element_dict[element]["element_s_e"]
    p = element_dict[element]["element_p_e"]
    d = element_dict[element]["element_d_e"]
    f = element_dict[element]["element_f_e"]

    return count_unpaired_electrons(s, p, d, f)

def elemental_CSV_to_nested_dict(csv_file_name: str = "PubChemElements_all.csv"):
    """
    Loads and processes elemental properties from a PubChem periodic table CSV file.

    The function reads element-level data and parses the electron configuration string into
    numerical features for use in ML models (e.g., valence electron count, s/p/d/f orbital counts).
    It returns a dictionary where each element symbol maps to a dictionary of its properties.

    Source: National Center for Biotechnology Information. Periodic Table of Elements.
    https://pubchem.ncbi.nlm.nih.gov/periodic-table/. Accessed May 30, 2025.

    Args:
        csv_file_name (str): Filename of the elemental CSV containing properties for each element.

    Returns:
        dict: A nested dictionary where each key is an element symbol (e.g., 'Fe'), and the value is
              another dictionary containing its processed properties:
              {
                  "AtomicNumber": int,
                  "AtomicMass": float,
                  "element_valence_e": int,
                  "element_s_e": int,
                  "element_p_e": int,
                  "element_d_e": int,
                  "element_f_e": int,
                  "Electronegativity": float,
                  "AtomicRadius": float (in Å),
                  "IonizationEnergy": float,
                  "ElectronAffinity": float,
                  "OxidationStates": str
              }
    """

    # Get full path to the CSV file (may use internal path logic via get_data_path)
    csv_path = get_data_path(csv_file_name)

    # Columns we want to retain or compute
    relevant_columns = [
        "AtomicNumber", "Symbol", "AtomicMass",
        "element_valence_e", "element_s_e", "element_p_e", "element_d_e", "element_f_e",
        "Electronegativity", "AtomicRadius", "IonizationEnergy", "ElectronAffinity", "OxidationStates"
    ]

    # Load raw CSV
    elemental_CSV = pd.read_csv(csv_path)

    # Lists to store processed electron configuration features
    element_valence_e_list = []
    element_s_e_list = []
    element_p_e_list = []
    element_d_e_list = []
    element_f_e_list = []

    # Parse the electron configuration string for each element
    for i in range(len(elemental_CSV)):
        val_e, s_e, p_e, d_e, f_e = encode_ElectronConfiguration(
            elemental_CSV["ElectronConfiguration"][i]
        )
        element_valence_e_list.append(val_e)
        element_s_e_list.append(s_e)
        element_p_e_list.append(p_e)
        element_d_e_list.append(d_e)
        element_f_e_list.append(f_e)

    # Add parsed features as new columns to the dataframe
    elemental_CSV = elemental_CSV.assign(
        element_valence_e=element_valence_e_list,
        element_s_e=element_s_e_list,
        element_p_e=element_p_e_list,
        element_d_e=element_d_e_list,
        element_f_e=element_f_e_list
    )

    # Keep only relevant columns and fill missing values with 0
    elemental_CSV = elemental_CSV[relevant_columns].fillna(0)

    # Convert atomic radius from picometers (pm) to angstroms (Å)
    elemental_CSV["AtomicRadius"] = elemental_CSV["AtomicRadius"] * 0.01

    # Convert to nested dictionary keyed by element symbol (e.g., 'Fe')
    return elemental_CSV.set_index("Symbol").to_dict(orient="index")


def structure_to_dgl_graph(structure: Structure,
                           elemental_dict: dict,
                           cutoff=None,
                           cutoff_padding=0.1,
                           rdf_cutoff=10.0,
                           rdf_bin_size=0.05):
    """
    Converts a Pymatgen structure into a DGLGraph with node and edge features.

    Parameters
    ----------
    structure : pymatgen Structure
        Structure to convert.
    elemental_dict : dict
        Nested dict of elemental features indexed by element symbol.
    cutoff : dict or None
        Optional cutoff dictionary per element.
    cutoff_padding : float
        Padding added to RDF-derived cutoffs.
    rdf_cutoff : float
        RDF distance cutoff.
    rdf_bin_size : float
        RDF bin resolution.

    Returns
    -------
    g : dgl.DGLGraph
        Graph with node and edge features.
    """

    elements = [site.specie.symbol for site in structure]
    n_atoms = len(elements)

    # 1. Get cutoffs
    if cutoff is None:
        cutoff = get_element_cutoff_dict_from_rdf(structure,
                                                  cutoff=rdf_cutoff,
                                                  bin_size=rdf_bin_size,
                                                  padding=cutoff_padding)

    # 2. Build edge list
    edge_src, edge_dst, edge_length, edge_image = [], [], [], []
    all_neighbors = structure.get_all_neighbors(r=max(cutoff.values()), include_index=True)

    for i in range(n_atoms):
        elem_i = elements[i]
        elem_cutoff = cutoff.get(elem_i, max(cutoff.values()))
        for site, dist, j, image in all_neighbors[i]:
            if dist <= elem_cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                edge_length.append(dist)
                edge_image.append(image)

    # 3. Build node features from dict
    feature_keys = [
        "AtomicNumber", "AtomicMass", "element_valence_e", "element_s_e", "element_p_e",
        "element_d_e", "element_f_e", "Electronegativity", "AtomicRadius",
        "IonizationEnergy", "ElectronAffinity"
    ]

    atomic_features = []
    for el in elements:
        if el not in elemental_dict:
            raise ValueError(f"Element {el} not found in elemental_dict.")
        props = elemental_dict[el]
        base_feats = []
        for k in feature_keys:
            val = props.get(k, 0)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            base_feats.append(val)

        unpaired_e = count_unpaired_electrons(
            int(props.get("element_s_e")),
            int(props.get("element_p_e")),
            int(props.get("element_d_e")),
            int(props.get("element_f_e"))
        )
        atomic_features.append(base_feats + [unpaired_e])
    node_features = torch.tensor(atomic_features, dtype=torch.float32)


    # 4. Build base graph
    src = torch.tensor(edge_src, dtype=torch.int64)
    dst = torch.tensor(edge_dst, dtype=torch.int64)
    lengths = torch.tensor(edge_length, dtype=torch.float32).unsqueeze(1)
    images = torch.tensor(edge_image, dtype=torch.int8)

    # Manually make it bidirectional
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    lengths_all = torch.cat([lengths, lengths])
    images_all = torch.cat([images, images])

    # Add self-loops manually
    self_loop_src = torch.arange(n_atoms, dtype=torch.int64)
    self_loop_dst = torch.arange(n_atoms, dtype=torch.int64)
    self_loop_feats = torch.zeros((n_atoms, 1), dtype=torch.float32)
    self_loop_periodicity = torch.zeros((n_atoms, 3), dtype=torch.int8)

    # Concatenate all edges including self-loops
    src_all = torch.cat([src_all, self_loop_src])
    dst_all = torch.cat([dst_all, self_loop_dst])
    lengths_all = torch.cat([lengths_all, self_loop_feats])
    images_all = torch.cat([images_all, self_loop_periodicity])

    # Build graph
    g = dgl.graph((src_all, dst_all), num_nodes=n_atoms)
    g.ndata['feat'] = node_features
    g.edata['length'] = lengths_all
    g.edata['periodicity'] = images_all

    return g

def get_element_to_role_map(structure: Structure):
    """
    Infers role assignment (A, B, C) for ABC2-type structures.

    Returns:
        element_to_role (dict): Mapping from element symbol to role label ('A', 'B', or 'C').
    """
    formula = structure.composition.get_reduced_formula_and_factor()[0]
    # Parse elements in order of appearance in formula
    elements = re.findall(r'([A-Z][a-z]*)', formula)

    # Assign based on order: A, B, C (or C2)
    roles = ['A', 'B', 'C']
    if len(elements) > 3:
        raise ValueError(f"Too many elements in formula: {formula} (expected ABC₂-type)")

    return {el: roles[i] for i, el in enumerate(elements)}


def get_period_and_group(symbol):
    try:
        el = element(symbol)
        return el.period, el.group_id
    except Exception:
        return None, None  # for error handling

from pymatgen.core.periodic_table import Element

NUM_ORBITALS = {"s": 1, "p": 3, "d": 5, "f": 7}

def count_orbital_fill(electrons: int, num_orbitals: int):
    orbitals = [0] * num_orbitals

    # First pass: add 1 electron per orbital (Hund's rule)
    for i in range(min(electrons, num_orbitals)):
        orbitals[i] += 1

    remaining = electrons - num_orbitals

    # Second pass: pair up electrons
    for i in range(num_orbitals):
        if remaining <= 0:
            break
        if orbitals[i] == 1:
            orbitals[i] += 1
            remaining -= 1

    full = sum(1 for e in orbitals if e == 2)
    partial = sum(1 for e in orbitals if e == 1)
    unfilled = sum(1 for e in orbitals if e == 0)

    return unfilled, partial, full


def get_element_d_f_orbital_fill(symbol: str, allow_f_block=True) -> dict:
    features = {
        "is_d_block": 0,
        "is_f_block": 0,
        "d_unfilled": 0,
        "d_partial": 0,
        "d_full": 0,
        "f_unfilled": 0,
        "f_partial": 0,
        "f_full": 0,
    }
    try:
        el = Element(symbol)
        if el.block == "d":
            features["is_d_block"] = 1
            config = el.full_electronic_structure
            for _, subshell, electrons in config:
                if subshell == "d":
                    unfilled, partial, full = count_orbital_fill(electrons, 5)
                    features["d_unfilled"] = unfilled
                    features["d_partial"] = partial
                    features["d_full"] = full
        elif el.block == "f" and allow_f_block:
            features["is_f_block"] = 1
            config = el.full_electronic_structure
            for _, subshell, electrons in config:
                if subshell == "f":
                    unfilled, partial, full = count_orbital_fill(electrons, 7)
                    features["f_unfilled"] = unfilled
                    features["f_partial"] = partial
                    features["f_full"] = full
        # For B, if allow_f_block=False, this skips f-block features
    except Exception:
        pass
    return features

def get_aggregated_orbital_fill(entry: str) -> dict:
    A, B, _ = entry.split("_")[:3]

    A_feats = get_element_d_f_orbital_fill(A, allow_f_block=True)
    B_feats = get_element_d_f_orbital_fill(B, allow_f_block=True)

    # Ensure B's f-block contributions are zeroed (if not already)
    for key in list(B_feats.keys()):
        if key.startswith("f_"):
            B_feats[key] = 0

    # Aggregate counts
    unfilled = A_feats["d_unfilled"] + A_feats["f_unfilled"] + B_feats["d_unfilled"] + B_feats["f_unfilled"]
    partial = A_feats["d_partial"] + A_feats["f_partial"] + B_feats["d_partial"] + B_feats["f_partial"]
    full = A_feats["d_full"] + A_feats["f_full"] + B_feats["d_full"] + B_feats["f_full"]

    return {
        "total_unfilled": unfilled,
        "total_partial": partial,
        "total_full": full
    }


def get_element_valence_orbital_fill_from_dict(symbol: str, element_data_dict: dict) -> dict:
    """
    Computes the number of filled, partially filled, and unfilled orbitals
    in the valence shell for each orbital type (s, p, d, f) using external data.
    """
    concepts = {
        "s_filled": 0, "s_partial": 0, "s_unfilled": 0,
        "p_filled": 0, "p_partial": 0, "p_unfilled": 0,
        "d_filled": 0, "d_partial": 0, "d_unfilled": 0
    }

    max_orbitals = {"s": 1, "p": 3, "d": 5, "f": 7}

    try:
        row = element_data_dict[symbol]

        for subshell in ["s", "p", "d"]:
            electrons = row.get(f"element_{subshell}_e", 0) or 0
            num_orbitals = max_orbitals[subshell]
            unfilled, partial, full = count_orbital_fill(electrons, num_orbitals)
            concepts[f"{subshell}_unfilled"] = unfilled
            concepts[f"{subshell}_partial"] = partial
            concepts[f"{subshell}_filled"] = full

    except KeyError:
        print(f"Element {symbol} not found in dictionary.")

    return concepts

def get_element_magnetism_type(symbol: str) -> dict:
    """
    Returns magnetic classification for a pure element based on empirical magnetic ordering.
    Categories: Ferromagnetic, Antiferromagnetic, Nonmagnetic.
    """
    symbol = symbol.capitalize()

    # Bulk elemental ferromagnets (at or near room temperature)
    ferromagnetic = {"Fe", "Co", "Ni", "Gd", "Tb", "Dy"}

    # Known antiferromagnetic elements in pure form
    # Note: Some may require low temperatures to exhibit this behavior
    antiferromagnetic = {"Cr", "Mn", "O", "Er", "Tm"}

    paramagnetic = {"Li", "O", "Na", "Mg", "Al",
                    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
                    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Sn",
                    "Cs", "Ba", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Fr", "La"

    }

    # Diamagnetic or weakly paramagnetic (considered nonmagnetic in bulk)
    nonmagnetic = {"H", "He", "Be", "B", "C", "N", "F", "Ne",
                   "Si", "P", "S", "Cl", "Ar",
                   "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                   "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                   "Au", "Hg", "Tl", "Pb", "Bi",
                   "Rg"}

    # Initialize result
    result = {
        "is_ferromagnetic": 0,
        "is_antiferromagnetic": 0,
        "is_paramagnetic": 0,
        "is_nonmagnetic": 0,
    }

    if symbol in ferromagnetic:
        result["is_ferromagnetic"] = 1
    elif symbol in antiferromagnetic:
        result["is_antiferromagnetic"] = 1
    elif symbol in paramagnetic:
        result["is_paramagnetic"] = 1
    elif symbol in nonmagnetic:
        result["is_nonmagnetic"] = 1
    else:
        # Fallback: treat unknowns as nonmagnetic for safety
        result["is_nonmagnetic"] = 1

    return result

def get_element_magnetism_type_soft(symbol: str) -> dict:
    """
    Returns soft magnetic classification for a pure element based on empirical magnetic ordering.
    If an element fits multiple categories, the weights are normalized to sum to 1.
    """
    symbol = symbol.capitalize()

    ferromagnetic = {
    "Fe", "Co", "Ni", "Gd"
    }

    antiferromagnetic = {
        "Cr"
    }

    paramagnetic = {
        "Li", "O", "Na", "Mg", "Al",
        "K", "Ca", "Sc", "Ti", "V", "Mn",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Th", "Pa", "U", "Pu", "Am"
    }

    diamagnetic = {
        "H", "He", "Be", "B", "C", "N", "Ne",
        "Si", "P", "S", "Cl", "Ar",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
        "Au", "Hg", "Tl", "Pb", "Bi"
    }

    unknown_magnetic = {
        "F", "Po", "At", "Rn", "Fr", "Ra", "Ac",
        "Np", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Pm"
    }

    result = {
        "is_ferromagnetic": 0.0,
        "is_antiferromagnetic": 0.0,
        "is_paramagnetic": 0.0,
        "is_diamagnetic": 0.0,
    }

    categories = []
    if symbol in ferromagnetic:
        categories.append("is_ferromagnetic")
    if symbol in antiferromagnetic:
        categories.append("is_antiferromagnetic")
    if symbol in paramagnetic:
        categories.append("is_paramagnetic")
    if symbol in diamagnetic:
        categories.append("is_diamagnetic")

    if not categories:
        # Fallback for unknown elements
        categories = ["is_diamagnetic"]

    weight = 1.0 / len(categories)
    for cat in categories:
        result[cat] = weight

    return result


def get_concept_labels(entry: str, elem_dict: dict) -> dict:
    A, B, _ = entry.split("_")[:3]

    # A_features = get_element_valence_orbital_fill_from_dict(A, elem_dict)
    # B_features = get_element_valence_orbital_fill_from_dict(B, elem_dict)  # B cannot be f-block

    A_features = get_element_magnetism_type_soft(A)
    B_features = get_element_magnetism_type_soft(B)

    return {f"A_{k}": v for k, v in A_features.items()} | {f"B_{k}": v for k, v in B_features.items()}


# def get_concept_labels(entry: str) -> dict:
#     A, B, C, Structure = entry.split("_")
