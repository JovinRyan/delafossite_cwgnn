import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from delafossite_cwgnn.models.iter_norm import IterNormRotation as cw_layer

# ---------- Pooling ----------
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

# ---------- GCN Layer Wrapper ----------
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.0):
        super().__init__()
        self.conv = GraphConv(in_feats, out_feats, norm='both', weight=True, bias=True)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = self.conv(g, x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# ---------- GCNNet Base ----------
class GCNNetBase(nn.Module):
    def __init__(self, layer_dims, num_concepts, output_dim=None, num_classes=None,
                 dropout=0.2, use_cw=True, pooling="mean", regression=True):
        super().__init__()
        self.use_cw = use_cw
        self.pooling = pooling
        self.num_layers = len(layer_dims)
        self.dropout_layer = nn.Dropout(dropout)
        self.regression = regression

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for in_f, out_f in layer_dims:
            self.gcn_layers.append(GCNLayer(in_f, out_f, dropout=dropout))

        # Final normalization
        if use_cw:
            self.cw_layer = cw_layer(num_features=layer_dims[-1][1], dim=4, activation_mode="mean", mode=-1)
        else:
            self.bn_norm = nn.BatchNorm1d(layer_dims[-1][1])

        # Output heads
        if regression:
            self.regressor = nn.Linear(layer_dims[-1][1], output_dim)
        else:
            self.classifier = nn.Linear(layer_dims[-1][1], num_classes)
        self.concept_head = nn.Linear(layer_dims[-1][1], num_concepts)

    def forward(self, g, x):
        for layer in self.gcn_layers:
            x = layer(g, x)

        batch_num_nodes = g.batch_num_nodes()
        batch = torch.cat([torch.full((n,), i, device=x.device, dtype=torch.long)
                           for i, n in enumerate(batch_num_nodes)])

        if self.use_cw:
            x = self.cw_layer(x, g, None)
        graph_emb = pool_graph(x, batch, pooling=self.pooling)
        if not self.use_cw:
            graph_emb = self.bn_norm(graph_emb)

        if self.regression:
            y = self.regressor(graph_emb)
        else:
            y = self.classifier(graph_emb)
        concept_logits = self.concept_head(graph_emb)
        return x, y, concept_logits

    def change_mode(self, mode):
        if self.use_cw:
            self.cw_layer.mode = mode

    def update_rotation_matrix(self):
        if self.use_cw:
            self.cw_layer.update_rotation_matrix()

# ---------- GCNNet Regression ----------
class GCNNet_Regression(GCNNetBase):
    def __init__(self, layer_dims, num_concepts, output_dim=1, dropout=0.2, use_cw=True, pooling="mean"):
        super().__init__(layer_dims, num_concepts, output_dim=output_dim, dropout=dropout,
                         use_cw=use_cw, pooling=pooling, regression=True)

# ---------- GCNNet Classification ----------
class GCNNet_Classification(GCNNetBase):
    def __init__(self, layer_dims, num_classes, num_concepts, dropout=0.2, use_cw=True, pooling="mean"):
        super().__init__(layer_dims, num_concepts, num_classes=num_classes, dropout=dropout,
                         use_cw=use_cw, pooling=pooling, regression=False)
