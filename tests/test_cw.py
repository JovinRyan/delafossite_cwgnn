# test_cw.py
import torch
import dgl
import numpy as np
from datetime import datetime
from pathlib import Path
from delafossite_cwgnn.models.iter_norm import IterNormRotation  # your original import

LOG_FILE = Path("test_cw_log.txt")

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def make_dummy_graph(num_graphs=4, nodes_per_graph=5, feat_dim=8, device=torch.device("cpu")):
    graphs = []
    for _ in range(num_graphs):
        g = dgl.rand_graph(nodes_per_graph, nodes_per_graph * 2)
        g.ndata["feat"] = torch.randn(nodes_per_graph, feat_dim)
        graphs.append(g)
    bg = dgl.batch(graphs)
    return bg.to(device)

def test_shape_and_whitening(device):
    log("=== Test: Shape and Whitening ===")
    feat_dim = 8
    g = make_dummy_graph(num_graphs=3, nodes_per_graph=6, feat_dim=feat_dim, device=device)
    X = g.ndata["feat"]

    cw = IterNormRotation(num_features=feat_dim, dim=4, activation_mode="mean", mode=0)
    cw = cw.to(device)
    out = cw(X, None, None)  # <-- fixed here

    log(f"Input shape: {X.shape}, Output shape: {out.shape}")
    assert out.shape == X.shape, "Output shape mismatch"

    # Whitening check
    Xc = out - out.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    cov_np = cov.detach().cpu().numpy()
    log(f"Covariance matrix:\n{cov_np}")
    log(f"Mean diagonal ≈ 1: {np.mean(np.diag(cov_np)):.4f}")
    log(f"Off-diagonal mean ≈ 0: {np.mean(cov_np - np.eye(feat_dim)):.4f}")

def test_rotation_update(device):
    log("\n=== Test: Rotation Update ===")
    feat_dim = 8
    cw = IterNormRotation(num_features=feat_dim, dim=4, activation_mode="mean", mode=0)
    cw = cw.to(device)

    # Move sum_G and running_rot buffers to device too
    cw.sum_G = cw.sum_G.to(device)
    cw.running_rot = cw.running_rot.to(device)

    R_before = cw.running_rot.clone()
    cw.sum_G += torch.randn_like(cw.sum_G)  # simulate some accumulation
    cw.update_rotation_matrix()
    R_after = cw.running_rot

    diff = torch.norm(R_after - R_before).item()
    ortho_check = torch.norm(R_after @ R_after.transpose(-1, -2) - torch.eye(feat_dim, device=device)).item()

    log(f"Rotation matrix changed by: {diff:.6f}")
    log(f"Orthonormality check (||RRᵀ - I||): {ortho_check:.6e}")
    assert ortho_check < 1e-5, "Rotation matrix is not orthonormal!"

def test_activation_modes(device):
    log("\n=== Test: Activation Modes (excluding pooling) ===")
    modes = ["mean", "max", "pos_mean"]
    feat_dim = 8
    for act_mode in modes:
        log(f"Testing activation_mode='{act_mode}'")
        g = make_dummy_graph(num_graphs=2, nodes_per_graph=4, feat_dim=feat_dim, device=device)
        X = g.ndata["feat"]
        cw = IterNormRotation(num_features=feat_dim, dim=4, activation_mode=act_mode, mode=0)
        cw = cw.to(device)
        try:
            out = cw(X, None, None)  # <-- fixed here
            log(f"  Success. Output mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        except Exception as e:
            log(f"  FAILED with error: {e}")
            raise

if __name__ == "__main__":
    LOG_FILE.write_text(f"IterNormRotation Test Log - {datetime.now()}\n")
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    test_shape_and_whitening(device)
    test_rotation_update(device)
    test_activation_modes(device)

    log("\nAll tests completed successfully.")
