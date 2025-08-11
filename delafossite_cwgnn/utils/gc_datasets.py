import os
import torch
import pandas as pd
import dgl
from torch.utils.data import Dataset

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
    concept_start_idx : int, default=3
        Index (0-based) of the first concept column.
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=3):
        self.data = pd.read_csv(csv_path)
        self.graph_dir = graph_dir

        # First column = structure ID
        self.structure_ids = self.data.iloc[:, 0].tolist()

        # All concept columns start from concept_start_idx
        self.concept_columns = self.data.columns[concept_start_idx:]
        self.concepts = self.data[self.concept_columns].values.astype(float)

        # Save concept column names as a list of strings
        self.concept_column_names = list(self.concept_columns)

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
    concept_start_idx : int, default=3
        Index of first concept column.
    target_col : str or None, default=None
        Column name for target values. If None, uses second column in CSV.
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=3, target_col=None):
        super().__init__(csv_path, graph_dir, concept_start_idx)
        if target_col is None:
            target_col = self.data.columns[1]
        self.targets = self.data[target_col].astype(float).tolist()

        # Save target column name
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
    concept_start_idx : int, default=3
        Index of first concept column.
    target_col : str or None, default=None
        Column name for target labels. If None, uses second column in CSV.
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=3, target_col=None):
        super().__init__(csv_path, graph_dir, concept_start_idx)
        if target_col is None:
            target_col = self.data.columns[1]
        self.targets = self.data[target_col].astype(int).tolist()

        # Save target column name
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
