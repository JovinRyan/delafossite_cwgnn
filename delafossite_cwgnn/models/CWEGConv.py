import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from delafossite_cwgnn.models.iter_norm import IterNormRotation as cw_layer

# ---------------- Edge-GCN Layer ----------------
class EGConv(nn.Module):
    """
    GraphConv that incorporates edge features via an MLP to modulate messages.
    """
    def __init__(self, in_node_feats, in_edge_feats, out_feats):
        super().__init__()
        self.conv = GraphConv(in_node_feats, out_feats, norm='both', weight=True, bias=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, g, node_feats, edge_feats):
        # Compute standard GCN aggregation
        h = self.conv(g, node_feats)
        # Aggregate edge features to nodes
        with g.local_scope():
            g.edata['ef'] = self.edge_mlp(edge_feats)
            g.update_all(dgl.function.copy_e('ef', 'm'), dgl.function.mean('m', 'ef_sum'))
            h = h + g.ndata['ef_sum']  # combine node and edge info
        return h


# ---------------- Pooling ----------------
def pool_graph(node_feats, batch, pooling="mean"):
    if pooling == "mean":
        return torch.zeros((batch.max() + 1, node_feats.size(1)), device=node_feats.device)\
                   .index_add_(0, batch, node_feats) / \
               torch.bincount(batch, minlength=batch.max() + 1).unsqueeze(1)
    elif pooling == "sum":
        return torch.zeros((batch.max() + 1, node_feats.size(1)), device=node_feats.device)\
                   .index_add_(0, batch, node_feats)
    elif pooling == "max":
        out = torch.zeros((batch.max() + 1, node_feats.size(1)), device=node_feats.device) - float('inf')
        for i in range(batch.max()+1):
            mask = batch == i
            out[i] = node_feats[mask].max(dim=0)[0]
        return out
    else:
        raise ValueError(f"Unknown pooling type: {pooling}")


# ---------------- Edge-GCN for Classification ----------------
class EGNN_Classification(nn.Module):
    def __init__(self, layer_dims, num_classes, num_concepts, edge_feats_dim=1,
                 dropout=0.2, use_cw=True, pooling="mean"):
        super().__init__()
        self.use_cw = use_cw
        self.pooling = pooling
        self.num_layers = len(layer_dims)
        self.dropout_layer = nn.Dropout(dropout)

        # EGConv layers
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for in_f, out_f in layer_dims:
            self.gcn_layers.append(EGConv(in_f, edge_feats_dim, out_f))
            self.norm_layers.append(nn.BatchNorm1d(out_f))

        # Final normalization
        if use_cw:
            self.cw_layer = cw_layer(num_features=layer_dims[-1][1], dim=4, activation_mode="mean", mode=-1)
        else:
            self.bn_norm = nn.BatchNorm1d(layer_dims[-1][1])

        # Output heads
        self.classifier = nn.Linear(layer_dims[-1][1], num_classes)
        self.concept_head = nn.Linear(layer_dims[-1][1], num_concepts)

    def forward(self, g, x):
        edge_feats = g.edata['length']

        for gcn, norm in zip(self.gcn_layers, self.norm_layers):
            x = gcn(g, x, edge_feats)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        batch_num_nodes = g.batch_num_nodes()
        batch = torch.cat([torch.full((n,), i, device=x.device, dtype=torch.long)
                           for i, n in enumerate(batch_num_nodes)])

        if self.use_cw:
            x = self.cw_layer(x, g, None)

        graph_emb = pool_graph(x, batch, pooling=self.pooling)
        if not self.use_cw:
            graph_emb = self.bn_norm(graph_emb)

        logits = self.classifier(graph_emb)
        concept_logits = self.concept_head(graph_emb)
        return x, logits, concept_logits

    def change_mode(self, mode):
        if self.use_cw:
            self.cw_layer.mode = mode

    def update_rotation_matrix(self):
        if self.use_cw:
            self.cw_layer.update_rotation_matrix()
