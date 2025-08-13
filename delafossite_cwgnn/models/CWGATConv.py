import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.functional import edge_softmax
from delafossite_cwgnn.models.iter_norm import IterNormRotation as cw_layer

# ---------- GAT layer with edge features ----------
class GATConvEdge(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, num_heads=1, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats

        self.node_proj = nn.Linear(in_node_feats, out_feats * num_heads, bias=False)
        self.edge_proj = nn.Linear(in_edge_feats, out_feats * num_heads, bias=False)

        self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_feats * 3)))
        nn.init.xavier_uniform_(self.attn.data)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            h = self.node_proj(node_feats)
            e = self.edge_proj(edge_feats)

            h = h.view(-1, self.num_heads, self.out_feats)
            e = e.view(-1, self.num_heads, self.out_feats)

            g.ndata['h'] = h
            g.edata['ef'] = e

            # Attention
            def edge_attention(edges):
                z_cat = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=-1)
                a = (z_cat * self.attn).sum(dim=-1)
                return {'a': self.leaky_relu(a)}

            g.apply_edges(edge_attention)
            g.edata['a'] = edge_softmax(g, g.edata['a'])

            def message_func(edges):
                return {'m': edges.data['a'].unsqueeze(-1) * edges.src['h']}

            g.update_all(message_func, dgl.function.mean('m', 'h_new'))
            return g.ndata['h_new'].flatten(1)


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

# ---------- GATEdgeNet for Regression ----------
class GATEdgeNet_Regression(nn.Module):
    def __init__(self, layer_dims, num_concepts, edge_feats_dim=1, dropout=0.2,
                 use_cw=True, pooling="mean", output_dim=1):
        super().__init__()
        self.use_cw = use_cw
        self.pooling = pooling
        self.num_layers = len(layer_dims)
        self.dropout_layer = nn.Dropout(dropout)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for in_f, out_f, heads in layer_dims:
            self.gat_layers.append(GATConvEdge(in_f, edge_feats_dim, out_f, num_heads=heads, dropout=dropout))
            self.norm_layers.append(nn.BatchNorm1d(out_f * heads))

        # Final normalization
        if use_cw:
            self.cw_layer = cw_layer(num_features=layer_dims[-1][1], dim=4, activation_mode="mean", mode=-1)
        else:
            self.bn_norm = nn.BatchNorm1d(layer_dims[-1][1])

        # Output heads
        self.regressor = nn.Linear(layer_dims[-1][1], output_dim)
        self.concept_head = nn.Linear(layer_dims[-1][1], num_concepts)

    def forward(self, g, x):
        edge_feats = g.edata['length']

        for gat, norm in zip(self.gat_layers, self.norm_layers):
            x = gat(g, x, edge_feats)
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

        y = self.regressor(graph_emb)
        concept_logits = self.concept_head(graph_emb)
        return x, y, concept_logits

    def change_mode(self, mode):
        if self.use_cw:
            self.cw_layer.mode = mode

    def update_rotation_matrix(self):
        if self.use_cw:
            self.cw_layer.update_rotation_matrix()


# ---------- GATEdgeNet for Classification ----------
class GATEdgeNet_Classification(nn.Module):
    def __init__(self, layer_dims, num_classes, num_concepts, edge_feats_dim=1, dropout=0.2,
                 use_cw=True, pooling="mean"):
        super().__init__()
        self.use_cw = use_cw
        self.pooling = pooling
        self.num_layers = len(layer_dims)
        self.dropout_layer = nn.Dropout(dropout)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for in_f, out_f, heads in layer_dims:
            self.gat_layers.append(GATConvEdge(in_f, edge_feats_dim, out_f, num_heads=heads, dropout=dropout))
            self.norm_layers.append(nn.BatchNorm1d(out_f * heads))

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

        for gat, norm in zip(self.gat_layers, self.norm_layers):
            x = gat(g, x, edge_feats)
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
