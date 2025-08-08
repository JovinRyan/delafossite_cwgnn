import dgl
import torch
import random
import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_random_graph(num_nodes, num_edges, use_cuda=False):
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst))

    # Add random node and edge features
    g.ndata['feat'] = torch.randn(num_nodes, 4)
    g.edata['weight'] = torch.rand(num_edges)

    if use_cuda:
        g = g.to('cuda')

    return g

def test_dgl():
    lines = []
    lines.append("="*40)
    lines.append("DGL Functionality Test")
    lines.append(f"Timestamp: {datetime.datetime.now()}")
    lines.append("="*40)

    try:
        lines.append(f"DGL version: {dgl.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")

        use_cuda = torch.cuda.is_available()

        # Generate 3 random graphs of varying sizes
        graphs = []
        for i in range(3):
            num_nodes = random.randint(5, 20)
            num_edges = random.randint(num_nodes, num_nodes * 3)
            g = generate_random_graph(num_nodes, num_edges, use_cuda)
            graphs.append(g)
            lines.append(f"\nGraph {i}:")
            lines.append(f" - Nodes: {g.num_nodes()}")
            lines.append(f" - Edges: {g.num_edges()}")
            lines.append(f" - Node feature shape: {g.ndata['feat'].shape}")
            lines.append(f" - Edge weight shape: {g.edata['weight'].shape}")
            lines.append(f" - On CUDA: {g.device.type == 'cuda'}")

        lines.append("\nDGL test completed successfully.")

    except Exception as e:
        lines.append("\nAn error occurred during DGL test:")
        lines.append(str(e))

    return "\n".join(lines)

if __name__ == "__main__":
    output = test_dgl()

    # Print to terminal
    print(output)

    # Write to file
    with open("test_dgl_log.txt", "w") as f:
        f.write(output + "\n")
