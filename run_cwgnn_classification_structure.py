import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from delafossite_cwgnn.utils.gc_datasets import GraphConceptClassificationDataset, collate_fn  # Adjust imports
from delafossite_cwgnn.models.cw_layer import IterNormRotation as cw_layer
import dgl.nn as dglnn
import dgl
from sklearn.metrics import accuracy_score, r2_score

def mean_pooling(node_feats, batch):
    """
    node_feats: (num_nodes, feature_dim)
    batch: (num_nodes,) tensor indicating graph assignment of each node
    returns: (batch_size, feature_dim) graph embeddings
    """
    batch_size = batch.max().item() + 1
    graph_emb = torch.zeros(batch_size, node_feats.size(1), device=node_feats.device)
    count = torch.zeros(batch_size, device=node_feats.device)

    graph_emb = graph_emb.index_add_(0, batch, node_feats)
    count = count.index_add_(0, batch, torch.ones_like(batch, dtype=node_feats.dtype))

    graph_emb = graph_emb / count.unsqueeze(1)
    return graph_emb

def evaluate(model, dataloader, device, concepts_are_continuous=False):
    model.eval()
    all_labels = []
    all_preds = []
    all_concept_labels = []
    all_concept_preds = []

    with torch.no_grad():
        for g, labels, concept_labels in dataloader:
            g = g.to(device)
            feats = g.ndata['feat'].to(device)
            labels = labels.to(device)
            concept_labels = concept_labels.to(device)

            emb, logits, concept_logits = model(g, feats)
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_concept_labels.append(concept_labels.cpu())
            all_concept_preds.append(concept_logits.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_concept_labels = torch.cat(all_concept_labels)
    all_concept_preds = torch.cat(all_concept_preds)

    cls_acc = accuracy_score(all_labels, all_preds)

    if concepts_are_continuous:
        concept_metric = r2_score(all_concept_labels.numpy(), all_concept_preds.numpy())
    else:
        concept_pred_binary = (torch.sigmoid(all_concept_preds) >= 0.5).int()
        concept_metric = (concept_pred_binary == all_concept_labels.int()).float().mean().item()

    return cls_acc, concept_metric

class EdgeConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats_dim=1):
        super().__init__()
        # MLP for edge message function, input: [h_i || h_j || e_ij]
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_feats + edge_feats_dim, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, g, h, e):
        """
        g: DGLGraph
        h: node features (N, in_feats)
        e: edge features (E, edge_feats_dim)

        Returns:
          new node features (N, out_feats)
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e

            # Compute messages for each edge
            # src = h_j, dst = h_i for message from j->i
            g.apply_edges(lambda edges: {
                'm': self.mlp(torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))
            })
            # Aggregate messages by sum
            g.update_all(message_func=dgl.function.copy_e('m', 'm'),
                         reduce_func=dgl.function.sum('m', 'h_new'))

            return g.ndata['h_new']


class EdgeConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_concepts=8, cw_flag=True, dropout=0.2, edge_feats_dim=1):
        super().__init__()
        self.cw_flag = cw_flag
        self.dropout = dropout
        self.num_concepts = num_concepts

        self.edgeconv1 = EdgeConvLayer(input_dim, hidden_dim, edge_feats_dim)
        self.norm1 = cw_layer(hidden_dim) if cw_flag else nn.BatchNorm1d(hidden_dim)

        self.edgeconv2 = EdgeConvLayer(hidden_dim, hidden_dim, edge_feats_dim)
        self.norm2 = cw_layer(hidden_dim) if cw_flag else nn.BatchNorm1d(hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.concept_head = nn.Linear(hidden_dim, num_concepts)

    def forward(self, g, x):
      edge_feats = g.edata['length']  # (E, 1)

      x = self.edgeconv1(g, x, edge_feats)
      if self.cw_flag:
          x = self.norm1(x, g, None)
      else:
          x = self.norm1(x)
      x = F.relu(x)
      x = self.dropout_layer(x)

      x = self.edgeconv2(g, x, edge_feats)
      if self.cw_flag:
          x = self.norm2(x, g, None)
      else:
          x = self.norm2(x)
      x = F.relu(x)

      batch_num_nodes = g.batch_num_nodes()
      batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(batch_num_nodes)]).to(x.device)
      graph_emb = mean_pooling(x, batch)

      logits = self.classifier(graph_emb)
      concept_logits = self.concept_head(graph_emb)

      return x, logits, concept_logits

    def change_mode(self, mode):
        if self.cw_flag:
            self.norm1.mode = mode
            self.norm2.mode = mode

    def update_rotation_matrix(self):
        if self.cw_flag:
            self.norm1.update_rotation_matrix()
            self.norm2.update_rotation_matrix()

def train():
    # Config
    csv_path = "/mnt/d/ORNL Internship Summer 2025/Delafossite ML/bigvasp/dgl_graphs/target_structure_type_concepts_triplet_angles.csv"
    graph_dir = "/mnt/d/ORNL Internship Summer 2025/Delafossite ML/bigvasp/dgl_graphs"
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 32
    lr = 1e-3
    num_epochs = 50
    patience = 5  # stop if no val improvement for 5 epochs
    lr_decay_patience = 3  # reduce LR if no improvement for 3 epochs
    lr_decay_factor = 0.5
    concept_start_idx = 3
    target_col = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GraphConceptClassificationDataset(csv_path, graph_dir, concept_start_idx, target_col)

    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = EdgeConvNet(
        input_dim=dataset[0][0].ndata['feat'].shape[1],
        hidden_dim=128,
        output_dim=dataset.num_classes,
        num_concepts=len(dataset.concept_column_names),
        cw_flag=True,
        dropout=0.2,
        edge_feats_dim=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_decay_factor, patience=lr_decay_patience, verbose=True
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_cls_loss, total_concept_loss, total_correct, total_samples = 0, 0, 0, 0

        for g, labels, concept_labels in train_loader:
            g = g.to(device)
            feats = g.ndata['feat'].to(device)
            labels = labels.to(device)
            concept_labels = concept_labels.to(device)

            optimizer.zero_grad()
            emb, logits, concept_logits = model(g, feats)

            loss_cls = F.cross_entropy(logits, labels)
            loss_concept = F.mse_loss(concept_logits, concept_labels.float())
            loss = loss_cls + loss_concept

            loss.backward()
            optimizer.step()

            total_cls_loss += loss_cls.item() * labels.size(0)
            total_concept_loss += loss_concept.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_cls_loss = total_cls_loss / total_samples
        train_concept_loss = total_concept_loss / total_samples
        train_acc = total_correct / total_samples

        val_cls_acc, val_concept_r2 = evaluate(model, val_loader, device, concepts_are_continuous=True)
        test_cls_acc, test_concept_r2 = evaluate(model, test_loader, device, concepts_are_continuous=True)

        scheduler.step(val_cls_acc)  # adjust LR if plateau

        print(f"Epoch {epoch:02d} | Train CLS Loss: {train_cls_loss:.4f} CONCEPT Loss: {train_concept_loss:.4f} Acc: {train_acc:.4f}")
        print(f"          Val CLS Acc: {val_cls_acc:.4f} CONCEPT Acc: {val_concept_r2:.4f}")
        print(f"          Test CLS Acc: {test_cls_acc:.4f} CONCEPT Acc: {test_concept_r2:.4f}")
        print(f"          LR: {optimizer.param_groups[0]['lr']:.6f}")

        metrics.append({
            'epoch': epoch,
            'train_cls_loss': train_cls_loss,
            'train_concept_loss': train_concept_loss,
            'train_acc': train_acc,
            'val_cls_acc': val_cls_acc,
            'val_concept_r2': val_concept_r2,
            'test_cls_acc': test_cls_acc,
            'test_concept_r2': test_concept_r2,
        })

        if val_cls_acc > best_val_acc:
            best_val_acc = val_cls_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_cwgnn_classifier.pth"))
            print(f"Saved best model at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break

    csv_path_metrics = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path_metrics, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_cls_loss', 'train_concept_loss', 'train_acc',
                      'val_cls_acc', 'val_concept_r2', 'test_cls_acc', 'test_concept_r2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print("Training finished.")
    print(f"Best val classification accuracy: {best_val_acc:.4f}")
    print(f"Training metrics saved to {csv_path_metrics}")

if __name__ == "__main__":
    train()
