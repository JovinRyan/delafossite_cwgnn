import os
import torch
import pandas as pd
import dgl
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

class BaseGraphConceptDataset(Dataset):
    """
    Base dataset for graph + concept vector tasks.

    This class handles:
        - Reading a CSV file containing IDs, targets, and concept features.
        - Loading corresponding DGLGraph objects from .bin files.
        - Extracting concept feature vectors.

    CSV Format
    ----------
    Column 0 : str
        Structure ID (used to find .bin file).
    Column 1 : varies
        Target value (numeric for regression, integer/string for classification).
    Column 2..n : varies
        Additional metadata or unused columns.
    Columns [concept_start_idx:] : float
        Concept features.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    graph_dir : str
        Directory containing .bin graph files.
    concept_start_idx : int, default=2
        Index (0-based) of the first concept column.
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=2, scaler=None, is_training=False):
        self.data = pd.read_csv(csv_path)
        self.graph_dir = graph_dir

        # First column = structure ID
        self.structure_ids = self.data.iloc[:, 0].tolist()

        # All concept columns start from concept_start_idx
        self.concept_columns = self.data.columns[concept_start_idx:]
        self.concepts = self.data[self.concept_columns].values.astype(float)

        # Save concept column names as a list of strings
        self.concept_column_names = list(self.concept_columns)

        self.concepts = self.data[self.concept_columns].values.astype(float)

        # Handle scaling
        if scaler is None and is_training:
            self.scaler = StandardScaler()
            self.scaler.fit(self.concepts)
            self.concepts = self.scaler.transform(self.concepts)
        elif scaler is not None:
            self.scaler = scaler
            self.concepts = self.scaler.transform(self.concepts)
        else:
            self.scaler = None  # no scaling

    def __len__(self):
        return len(self.structure_ids)

    def _load_graph(self, struct_id):
        """
        Load a DGLGraph from the graph directory using the structure ID.
        """
        graph_path = os.path.join(self.graph_dir, f"{struct_id}.bin")
        g, _ = dgl.load_graphs(graph_path)
        return g[0]

    def _get_concepts(self, idx):
        """
        Return the concept feature vector for a given index as a torch.FloatTensor.
        """
        return torch.tensor(self.concepts[idx], dtype=torch.float32)


class GraphConceptRegressionDataset(BaseGraphConceptDataset):
    """
    Dataset for regression tasks with graph + concept vector inputs.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing structure IDs, targets, and concepts.
    graph_dir : str
        Directory containing .bin graph files.
    concept_start_idx : int, default=2
        Index of first concept column.
    target_col : str or None, default=None
        Column name for target values. If None, uses second column in CSV.
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=2, target_col=None, scaler=None, is_training=False):
        super().__init__(csv_path, graph_dir, concept_start_idx, scaler=scaler, is_training=is_training)
        if target_col is None:
            target_col = self.data.columns[1]
        self.targets = self.data[target_col].astype(float).tolist()
        self.target_column_name = target_col

    def __getitem__(self, idx):
        """
        Returns
        -------
        graph : dgl.DGLGraph
        target : torch.FloatTensor
            Shape (1,) regression target.
        concept_vector : torch.FloatTensor
            Shape (num_concepts,) concept features.
        """
        graph = self._load_graph(self.structure_ids[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        concept_vector = self._get_concepts(idx)
        return graph, target, concept_vector


class GraphConceptClassificationDataset(BaseGraphConceptDataset):
    """
    Dataset for classification tasks with graph + concept vector inputs.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing structure IDs, targets, and concepts.
    graph_dir : str
        Directory containing .bin graph files.
    concept_start_idx : int, default=2
        Index of first concept column.
    target_col : str or None, default=None
        Column name for target labels. If None, uses second column in CSV.
    """

    def __init__(self, csv_path, graph_dir, concept_start_idx=2, target_col=None, scaler=None, is_training=False):
        super().__init__(csv_path, graph_dir, concept_start_idx, scaler=scaler, is_training=is_training)
        if target_col is None:
            target_col = self.data.columns[1]

        if pd.api.types.is_numeric_dtype(self.data[target_col]):
            self.targets = self.data[target_col].astype(int).tolist()
        else:
            self.targets = pd.Categorical(self.data[target_col]).codes.tolist()

        self.target_column_name = target_col

    def __getitem__(self, idx):
        """
        Returns
        -------
        graph : dgl.DGLGraph
        target : torch.LongTensor
            Shape (1,) classification label.
        concept_vector : torch.FloatTensor
            Shape (num_concepts,) concept features.
        """
        graph = self._load_graph(self.structure_ids[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        concept_vector = self._get_concepts(idx)
        return graph, target, concept_vector

    @property
    def num_classes(self):
        """
        Number of unique classification labels.
        """
        return len(set(self.targets))


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Parameters
    ----------
    batch : list of tuples
        Each tuple is (graph, label, concept_vector).

    Returns
    -------
    batched_graph : dgl.DGLGraph
        All graphs in batch combined into one batched graph.
    labels : torch.Tensor
        Targets stacked into a tensor (dtype depends on dataset type).
    concepts : torch.FloatTensor
        Shape (batch_size, num_concepts) concept features.
    """
    graphs, labels, concepts = zip(*batch)  # unzip tuples
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)  # keeps dtype from dataset
    concepts = torch.stack(concepts)
    return batched_graph, labels, concepts

def create_graph_dataloaders(dataset_cls, csv_path, graph_dir,
                             batch_size=32, concept_start_idx=2,
                             target_col=None, scaler=None,
                             train_frac=0.7, val_frac=0.15, test_frac=None,
                             collate_fn=None, shuffle=True, random_seed=42):
    """
    Create train, val, test DataLoaders for your GraphConceptDataset objects.
    """
    np.random.seed(random_seed)

    # Step 1: Load full dataset WITHOUT scaling
    full_dataset = dataset_cls(csv_path, graph_dir,
                               concept_start_idx=concept_start_idx,
                               target_col=target_col, scaler=None,
                               is_training=False)

    n = len(full_dataset)
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac
    assert train_frac + val_frac + test_frac <= 1.0 + 1e-6, "Fractions must sum <= 1"

    # Step 2: Shuffle indices
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Step 3: Fit scaler on training concepts if not provided
    if scaler is None:
        train_concepts = full_dataset.concepts[train_idx]
        scaler = StandardScaler()
        scaler.fit(train_concepts)

    # Step 4: Recreate dataset with scaler applied
    scaled_dataset = dataset_cls(csv_path, graph_dir,
                                 concept_start_idx=concept_start_idx,
                                 target_col=target_col, scaler=scaler,
                                 is_training=False)

    # Step 5: Create subsets
    train_ds = Subset(scaled_dataset, train_idx)
    val_ds = Subset(scaled_dataset, val_idx)
    test_ds = Subset(scaled_dataset, test_idx)

    # Step 6: Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, scaled_dataset, scaler
