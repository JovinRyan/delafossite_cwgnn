import os
import torch
import pandas as pd
import dgl
from torch.utils.data import Dataset

class GraphConceptDataset(Dataset):
    def __init__(self, csv_path, graph_dir, concept_start_idx=3, target_col="mag_class_encoded"):
        self.data = pd.read_csv(csv_path)
        self.graph_dir = graph_dir

        # Extract structure IDs and target labels
        self.structure_ids = self.data.iloc[:, 0].tolist()
        self.targets = self.data[target_col].astype(int).tolist()

        # Automatically extract concept columns by index
        self.concept_columns = self.data.columns[concept_start_idx:]
        self.concepts = self.data[self.concept_columns].values.astype(float)

    def __len__(self):
        return len(self.structure_ids)

    def __getitem__(self, idx):
        struct_id = self.structure_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{struct_id}.bin")

        # Load graph
        g, _ = dgl.load_graphs(graph_path)
        graph = g[0]

        # Get target and concepts
        label = self.targets[idx]
        concept_vector = torch.tensor(self.concepts[idx], dtype=torch.float)

        return graph, label, concept_vector

    @property
    def num_classes(self):
        return len(set(self.targets))



def collate_fn(batch):
    graphs, labels, concepts = zip(*batch)  # unzip
    labels = torch.tensor(labels, dtype=torch.long)
    batched_graph = dgl.batch(graphs)
    concepts = torch.stack(concepts)
    return batched_graph, labels, concepts
