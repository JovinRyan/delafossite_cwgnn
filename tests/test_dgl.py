import dgl
import torch
import random
import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_random_graph_with_periodicity(num_nodes, num_edges, use_cuda=False):
    # Random edges
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))

    # Build graph
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # ==== Node features ====
    # shape: (num_nodes, 12) + 1 unpaired electron feature → (num_nodes, 13)
    g.ndata['feat'] = torch.randn(num_nodes, 13, dtype=torch.float32)

    # ==== Edge features ====
    # length: (num_edges, 1) float
    g.edata['length'] = torch.rand(num_edges, 1, dtype=torch.float32)

    # periodicity: (num_edges, 3) int8
    g.edata['periodicity'] = torch.randint(
        low=-1, high=2, size=(num_edges, 3), dtype=torch.int8
    )

    # Move to GPU if requested
    if use_cuda:
        g = g.to('cuda')

    return g

def test_dgl():
    lines = []
    lines.append("=" * 40)
    lines.append("DGL Periodicity Data Test")
    lines.append(f"Timestamp: {datetime.datetime.now()}")
    lines.append("=" * 40)

    try:
        lines.append(f"DGL version: {dgl.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")

        use_cuda = torch.cuda.is_available()

        # Generate 3 random graphs
        for i in range(3):
            num_nodes = random.randint(5, 20)
            num_edges = random.randint(num_nodes, num_nodes * 3)
            g = generate_random_graph_with_periodicity(num_nodes, num_edges, use_cuda)

            lines.append(f"\nGraph {i}:")
            lines.append(f" - Nodes: {g.num_nodes()}")
            lines.append(f" - Edges: {g.num_edges()}")
            lines.append(f" - Node feature shape: {tuple(g.ndata['feat'].shape)}")
            lines.append(f" - Edge length shape: {tuple(g.edata['length'].shape)}")
            lines.append(f" - Edge periodicity shape: {tuple(g.edata['periodicity'].shape)}")
            lines.append(f" - On CUDA: {g.device.type == 'cuda'}")

            # Optional sanity checks
            assert g.ndata['feat'].dtype == torch.float32
            assert g.edata['length'].dtype == torch.float32
            assert g.edata['periodicity'].dtype == torch.int8

        lines.append("\nDGL periodicity test completed successfully.")

    except Exception as e:
        lines.append("\nAn error occurred during DGL periodicity test:")
        lines.append(str(e))

    return "\n".join(lines)

if __name__ == "__main__":
    output = test_dgl()

    # Print to terminal
    print(output)

    # Write to file
    with open("test_dgl_log.txt", "w") as f:
        f.write(output + "\n")
