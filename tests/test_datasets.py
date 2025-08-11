# tests/test_datasets.py
import os
import tempfile
import pandas as pd
import torch
import dgl
from torch.utils.data import DataLoader
from delafossite_cwgnn.utils.gc_datasets import (
    GraphConceptRegressionDataset,
    GraphConceptClassificationDataset,
    collate_fn
)

def create_dummy_graph(path):
    g = dgl.graph(([0, 1], [1, 0]))
    g.ndata['feat'] = torch.ones(2, 3)
    dgl.save_graphs(path, [g])

def create_dummy_csv(csv_path, graph_dir):
    rows = []
    for struct_id, target_reg, target_class in [
        ("Ag_Co_O_D", 1.23, 0),
        ("Ag_Co_O_T", 2.34, 1),
        ("Ag_Co_S_D", 3.45, 0),
        ("Ag_Co_S_T", 4.56, 1),
    ]:
        create_dummy_graph(os.path.join(graph_dir, f"{struct_id}.bin"))
        concepts = [0, 1, 0, 1, 0]
        rows.append([struct_id, target_reg, target_class] + concepts)
    columns = ["id", "sum_J", "mag_class_encoded"] + [f"concept_{i}" for i in range(5)]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)

def test_datasets():
    lines = []
    lines.append("="*40)
    lines.append("Dataset and DataLoader Test")
    lines.append("="*40)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            graph_dir = os.path.join(tmpdir, "graphs")
            os.makedirs(graph_dir, exist_ok=True)

            create_dummy_csv(csv_path, graph_dir)
            lines.append(f"Created dummy CSV and graphs in {tmpdir}")

            # Regression Dataset
            reg_ds = GraphConceptRegressionDataset(
                csv_path=csv_path,
                graph_dir=graph_dir,
                concept_start_idx=3,
                target_col="sum_J"
            )
            lines.append(f"Regression dataset length: {len(reg_ds)}")

            # Check target column name attribute
            assert hasattr(reg_ds, 'target_column_name'), "Regression dataset missing target_column_name attribute"
            assert reg_ds.target_column_name == "sum_J", f"Expected target_column_name 'sum_J', got {reg_ds.target_column_name}"
            lines.append(f"Regression target column name: {reg_ds.target_column_name}")

            # Check concept column names
            expected_concepts = [f"concept_{i}" for i in range(5)]
            assert hasattr(reg_ds, 'concept_column_names'), "Regression dataset missing concept_column_names attribute"
            assert reg_ds.concept_column_names == expected_concepts, f"Expected concept_column_names {expected_concepts}, got {reg_ds.concept_column_names}"
            lines.append(f"Regression concept column names: {reg_ds.concept_column_names}")

            g, y, c = reg_ds[0]
            lines.append(f"Regression sample graph nodes: {g.num_nodes()}, target: {y.item()}, concepts shape: {c.shape}")

            # Classification Dataset
            cls_ds = GraphConceptClassificationDataset(
                csv_path=csv_path,
                graph_dir=graph_dir,
                concept_start_idx=3,
                target_col="mag_class_encoded"
            )
            lines.append(f"Classification dataset length: {len(cls_ds)}")

            # Check target column name attribute
            assert hasattr(cls_ds, 'target_column_name'), "Classification dataset missing target_column_name attribute"
            assert cls_ds.target_column_name == "mag_class_encoded", f"Expected target_column_name 'mag_class_encoded', got {cls_ds.target_column_name}"
            lines.append(f"Classification target column name: {cls_ds.target_column_name}")

            # Check concept column names
            assert hasattr(cls_ds, 'concept_column_names'), "Classification dataset missing concept_column_names attribute"
            assert cls_ds.concept_column_names == expected_concepts, f"Expected concept_column_names {expected_concepts}, got {cls_ds.concept_column_names}"
            lines.append(f"Classification concept column names: {cls_ds.concept_column_names}")

            lines.append(f"Classification num classes: {cls_ds.num_classes}")
            g, y, c = cls_ds[0]
            lines.append(f"Classification sample label: {y.item()}, concepts shape: {c.shape}")

            # DataLoader test
            loader = DataLoader(cls_ds, batch_size=2, collate_fn=collate_fn)
            batch = next(iter(loader))
            batched_graph, labels, concepts = batch
            lines.append(f"Batched graph nodes: {batched_graph.num_nodes()}, edges: {batched_graph.num_edges()}")
            lines.append(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            lines.append(f"Concepts shape: {concepts.shape}")

        lines.append("\nDataset and DataLoader test completed successfully.")

    except Exception as e:
        lines.append("\nAn error occurred during dataset test:")
        lines.append(str(e))

    return "\n".join(lines)


if __name__ == "__main__":
    output = test_datasets()

    print(output)
    with open("test_datasets_log.txt", "w") as f:
        f.write(output + "\n")
