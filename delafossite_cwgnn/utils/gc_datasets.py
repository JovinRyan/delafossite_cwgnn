import os
import torch
import pandas as pd
import dgl
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
import copy

# ---------------------------
# Utility: make RNG + helpers
# ---------------------------
def _make_rng(seed):
    rng = np.random.RandomState(seed if seed is not None else 42)
    return rng

def _apply_gaussian_noise(arr, std, apply_prob, rng):
    """
    Add elementwise Gaussian noise with optional per-feature masking.
    Works on numpy arrays; returns a new array.
    """
    if std <= 0 or apply_prob <= 0:
        return arr.copy()
    mask = rng.rand(*arr.shape) < apply_prob
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape)
    noisy = arr.copy()
    noisy[mask] = noisy[mask] + noise[mask]
    return noisy

def _corrupt_labels(labels, num_classes, prob, strategy, rng):
    """
    Flip labels with probability prob.
    strategy:
      - 'uniform': new label is sampled uniformly from other classes
      - 'adjacent': new label is (y+1) % num_classes (useful for ordinal-ish classes)
    """
    if prob <= 0:
        return labels.copy()
    labels = np.asarray(labels, dtype=int)
    noisy = labels.copy()
    flip_mask = rng.rand(labels.shape[0]) < prob
    idxs = np.where(flip_mask)[0]
    if strategy == "adjacent":
        noisy[idxs] = (labels[idxs] + 1) % num_classes
    else:  # uniform
        for i in idxs:
            choices = list(range(num_classes))
            choices.remove(labels[i])
            noisy[i] = rng.choice(choices)
    return noisy.tolist()

# ---------------------------
# Base dataset (with noise)
# ---------------------------
class BaseGraphConceptDataset(Dataset):
    """
    Base dataset for graph + concept vector tasks, with optional noise injection.

    CSV Format
    ----------
    Column 0 : str   -> Structure ID
    Column 1 : varies -> Target (regression/classification)
    Columns [concept_start_idx:] : float -> Concept features (will be scaled)

    Parameters
    ----------
    csv_path : str
    graph_dir : str
    concept_start_idx : int, default=2
    scaler : sklearn StandardScaler or None
    is_training : bool
        If scaler is None and is_training=True, fit scaler on this dataset's concepts.
    noise_config : dict or None
        {
          "concept_gaussian_std": float (default 0.0),
          "concept_apply_prob": float in [0,1] (default 1.0),
          "seed": int or None (default 42)
        }
        Noise is applied in the **scaled** feature space.
    use_noisy_concepts : bool
        If True (default), __getitem__ returns noisy concepts (if configured).
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=2, scaler=None,
                 is_training=False, noise_config=None, use_noisy_concepts=True):
        self.data = pd.read_csv(csv_path)
        self.graph_dir = graph_dir

        # IDs
        self.structure_ids = self.data.iloc[:, 0].tolist()

        # Concepts
        self.concept_columns = self.data.columns[concept_start_idx:]
        self.concept_column_names = list(self.concept_columns)
        raw_concepts = self.data[self.concept_columns].values.astype(float)

        # Scaling
        if scaler is None and is_training:
            self.scaler = StandardScaler().fit(raw_concepts)
            self.concepts_scaled = self.scaler.transform(raw_concepts)
        elif scaler is not None:
            self.scaler = scaler
            self.concepts_scaled = self.scaler.transform(raw_concepts)
        else:
            self.scaler = None
            self.concepts_scaled = raw_concepts  # unscaled

        # Save clean concepts (numpy)
        self._concepts_clean = self.concepts_scaled.astype(np.float32)

        # Noise settings
        nc = noise_config or {}
        self._concept_std = float(nc.get("concept_gaussian_std", 0.0))
        self._concept_p = float(nc.get("concept_apply_prob", 1.0))
        self._seed = int(nc.get("seed", 42)) if nc.get("seed", 42) is not None else 42

        # Precompute noisy concepts (deterministic across epochs)
        rng = _make_rng(self._seed)
        self._concepts_noisy = _apply_gaussian_noise(self._concepts_clean, self._concept_std, self._concept_p, rng).astype(np.float32)

        # Switch
        self.use_noisy_concepts = use_noisy_concepts

    def __len__(self):
        return len(self.structure_ids)

    def _load_graph(self, struct_id):
        graph_path = os.path.join(self.graph_dir, f"{struct_id}.bin")
        g, _ = dgl.load_graphs(graph_path)
        return g[0]

    def _get_concepts(self, idx):
        if self.use_noisy_concepts:
            return torch.tensor(self._concepts_noisy[idx], dtype=torch.float32)
        else:
            return torch.tensor(self._concepts_clean[idx], dtype=torch.float32)

    # Convenience toggles
    def enable_concept_noise(self): self.use_noisy_concepts = True
    def disable_concept_noise(self): self.use_noisy_concepts = False


class GraphConceptRegressionDataset(BaseGraphConceptDataset):
    """
    Regression dataset with optional target noise.

    Extra noise_config keys:
      - "regression_target_gaussian_std": float (default 0.0)
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=2, target_col=None,
                 scaler=None, is_training=False, noise_config=None,
                 use_noisy_concepts=True, use_noisy_targets=True):
        super().__init__(csv_path, graph_dir, concept_start_idx, scaler, is_training,
                         noise_config=noise_config, use_noisy_concepts=use_noisy_concepts)

        if target_col is None:
            target_col = self.data.columns[1]
        self.target_column_name = target_col

        # Clean targets
        y_clean = self.data[target_col].astype(float).values
        self._targets_clean = y_clean.astype(np.float32)

        # Target noise
        rt_std = float((noise_config or {}).get("regression_target_gaussian_std", 0.0))
        rng = _make_rng((noise_config or {}).get("seed", 42))
        if rt_std > 0:
            self._targets_noisy = (self._targets_clean + rng.normal(0.0, rt_std, size=self._targets_clean.shape)).astype(np.float32)
        else:
            self._targets_noisy = self._targets_clean.copy()

        self.use_noisy_targets = use_noisy_targets

    def __getitem__(self, idx):
        graph = self._load_graph(self.structure_ids[idx])
        target = self._targets_noisy[idx] if self.use_noisy_targets else self._targets_clean[idx]
        concept_vector = self._get_concepts(idx)
        return graph, torch.tensor(target, dtype=torch.float32), concept_vector

    # Toggles
    def enable_target_noise(self): self.use_noisy_targets = True
    def disable_target_noise(self): self.use_noisy_targets = False


class GraphConceptClassificationDataset(BaseGraphConceptDataset):
    """
    Classification dataset with optional label corruption.

    Extra noise_config keys:
      - "label_corruption_prob": float in [0,1] (default 0.0)
      - "label_corruption_strategy": 'uniform' or 'adjacent' (default 'uniform')
    """
    def __init__(self, csv_path, graph_dir, concept_start_idx=2, target_col=None,
                 scaler=None, is_training=False, noise_config=None,
                 use_noisy_concepts=True, use_noisy_labels=True):
        super().__init__(csv_path, graph_dir, concept_start_idx, scaler, is_training,
                         noise_config=noise_config, use_noisy_concepts=use_noisy_concepts)

        if target_col is None:
            target_col = self.data.columns[1]
        self.target_column_name = target_col

        # Clean integer labels
        if pd.api.types.is_numeric_dtype(self.data[target_col]):
            y_clean = self.data[target_col].astype(int).tolist()
        else:
            y_clean = pd.Categorical(self.data[target_col]).codes.tolist()
        self._labels_clean = y_clean
        self._num_classes = len(set(y_clean))

        # Corruption
        nc = noise_config or {}
        p = float(nc.get("label_corruption_prob", 0.0))
        strat = str(nc.get("label_corruption_strategy", "uniform"))
        rng = _make_rng(nc.get("seed", 42))
        self._labels_noisy = _corrupt_labels(self._labels_clean, self._num_classes, p, strat, rng)

        self.use_noisy_labels = use_noisy_labels

    def __getitem__(self, idx):
        graph = self._load_graph(self.structure_ids[idx])
        label = self._labels_noisy[idx] if self.use_noisy_labels else self._labels_clean[idx]
        concept_vector = self._get_concepts(idx)
        return graph, torch.tensor(label, dtype=torch.long), concept_vector

    @property
    def num_classes(self):
        return self._num_classes

    # Toggles
    def enable_label_corruption(self): self.use_noisy_labels = True
    def disable_label_corruption(self): self.use_noisy_labels = False


def collate_fn(batch):
    graphs, labels, concepts = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    concepts = torch.stack(concepts)
    return batched_graph, labels, concepts


import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler

def create_graph_dataloaders(
    dataset_class,
    csv_path,
    graph_dir,
    batch_size,
    collate_fn,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=None,
    random_seed=42,
    train_concept_noise_std=0.0,         # <-- concept noise std (default off)
    train_label_corruption_prob=0.0      # <-- label corruption prob (default off)
):
    # Step 1: Load full dataset (no scaler yet)
    full_dataset = dataset_class(csv_path=csv_path, graph_dir=graph_dir)

    # Step 2: Split indices
    dataset_size = len(full_dataset)
    train_size = int(train_frac * dataset_size)
    val_size = int(val_frac * dataset_size)
    test_size = dataset_size - train_size - val_size if test_frac is None else int(test_frac * dataset_size)

    generator = torch.Generator().manual_seed(random_seed)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size, test_size], generator=generator
    )

    # Step 3: Fit scaler on train concepts
    train_concepts = np.stack([full_dataset[i][2] for i in train_indices])
    scaler = StandardScaler().fit(train_concepts)

    # Step 4: Create datasets with same scaler
    train_dataset = dataset_class(csv_path=csv_path, graph_dir=graph_dir, scaler=scaler)
    val_dataset = dataset_class(csv_path=csv_path, graph_dir=graph_dir, scaler=scaler)
    test_dataset = dataset_class(csv_path=csv_path, graph_dir=graph_dir, scaler=scaler)

    # Step 5: Subset to indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Step 6: Apply noise/corruption to train set only
    if train_concept_noise_std > 0.0:
        noisy_train_dataset = copy.deepcopy(train_dataset)
        for i in range(len(noisy_train_dataset)):
            graph, label, concepts = noisy_train_dataset[i]
            noise = np.random.normal(loc=0.0, scale=train_concept_noise_std, size=concepts.shape)
            noisy_train_dataset[i] = (graph, label, concepts + noise)
        train_dataset = noisy_train_dataset

    if train_label_corruption_prob > 0.0:
        corrupted_train_dataset = copy.deepcopy(train_dataset)
        num_classes = len(set([label for _, label, _ in full_dataset]))
        for i in range(len(corrupted_train_dataset)):
            graph, label, concepts = corrupted_train_dataset[i]
            if np.random.rand() < train_label_corruption_prob:
                new_label = np.random.choice([l for l in range(num_classes) if l != label])
                corrupted_train_dataset[i] = (graph, new_label, concepts)
        train_dataset = corrupted_train_dataset

    # Step 7: Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, full_dataset, scaler
