import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from delafossite_cwgnn.models.iter_norm import IterNormRotation as cw_layer

# ---------------- Edge-aware Graph Convolution Layer ----------------
class EGConv(nn.Module):
    """
    Edge-GCN layer that uses node features and edge features to compute messages.
    """
    def __init__(self, in_node_feats, in_edge_feats, out_feats):
        super().__init__()
        # MLP for edge-aware message
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_node_feats + in_edge_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            # Compute edge messages
            g.apply_edges(lambda edges: {
                'm': self.mlp(torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))
            })
            # Aggregate messages to nodes
            g.update_all(dgl.function.copy_e('m', 'm'), dgl.function.sum('m', 'h_new'))
            return g.ndata['h_new']


# ---------------- Graph Pooling ----------------
def mean_pooling(node_feats, batch):
    batch_size = batch.max().item() + 1
    graph_emb = torch.zeros(batch_size, node_feats.size(1), device=node_feats.device)
    count = torch.zeros(batch_size, device=node_feats.device)
    graph_emb = graph_emb.index_add_(0, batch, node_feats)
    count = count.index_add_(0, batch, torch.ones_like(batch, dtype=node_feats.dtype))
    graph_emb = graph_emb / count.unsqueeze(1)
    return graph_emb


# ---------------- Edge-GCN for Classification with CW ----------------
class EGNN_Classification(nn.Module):
    def __init__(self, layer_dims, num_classes, num_concepts, edge_feats_dim=1,
                 dropout=0.2, use_cw=True, pooling="mean"):
        super().__init__()
        self.use_cw = use_cw
        self.pooling = pooling
        self.dropout_layer = nn.Dropout(dropout)

        # EGConv layers + batchnorm
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for in_f, out_f in layer_dims:
            self.gcn_layers.append(EGConv(in_f, edge_feats_dim, out_f))
            self.norm_layers.append(nn.BatchNorm1d(out_f))

        # Final normalization / CW layer
        if use_cw:
            self.cw_layer = cw_layer(num_features=layer_dims[-1][1], dim=4, activation_mode="mean", mode=-1)
        else:
            self.bn_norm = nn.BatchNorm1d(layer_dims[-1][1])

        # Output heads
        self.classifier = nn.Linear(layer_dims[-1][1], num_classes)
        self.concept_head = nn.Linear(layer_dims[-1][1], num_concepts)

    def forward(self, g, x):
        edge_feats = g.edata['length']

        # EGConv layers
        for gcn, norm in zip(self.gcn_layers, self.norm_layers):
            x = gcn(g, x, edge_feats)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # CW or batchnorm
        if self.use_cw:
            x = self.cw_layer(x, g, None)

        # Pool graph
        batch_num_nodes = g.batch_num_nodes()
        batch = torch.cat([torch.full((n,), i, dtype=torch.long, device=x.device)
                           for i, n in enumerate(batch_num_nodes)])
        graph_emb = mean_pooling(x, batch)

        if not self.use_cw:
            graph_emb = self.bn_norm(graph_emb)

        logits = self.classifier(graph_emb)
        concept_logits = self.concept_head(graph_emb)
        return x, logits, concept_logits

    # ---------------- CW helpers ----------------
    def change_mode(self, mode):
        if self.use_cw:
            self.cw_layer.mode = mode

    def update_rotation_matrix(self):
        if self.use_cw:
            self.cw_layer.update_rotation_matrix()
