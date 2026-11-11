"""One-neuron gates, closed form ("write a neuron", ops set_rows / set_cols).

A gated MLP unit i computes down[:, i] * act(gate[i] . x) * (up[i] . x) from the layer's MLP input x. Rewritten as
  gate[i] = g (d - theta m),   up[i] = k m,   down[:, i] = w / a
with d a read direction, m a constant direction (m . x ~ 1 at every position, so g (d.x - theta) is a thresholded
score although the MLP has no bias) and w the vector to add to the residual stream, it stays silent while d.x < theta
and writes ~w once d.x passes it (a: the unit's mean activation on the positives, so the write is ~w there).
Everything is a mean, a covariance or one linear solve.

Choosing the sites is the method's job; what to check (see playbook): the positives must separate from EVERY position
the unit can fire at (a last-layer unit only moves its own position's logits; an earlier one writes into all later
positions through attention, so prompt-internal positions are negatives too), and negatives must not contain
positions that are positives by construction.
"""
import numpy as np


def lda_direction(P, N, shrink=0.1):
    """Unit read direction separating rows of P (positives) from N: (S + lam I)^-1 (mu_P - mu_N), S the pooled
    within-class covariance, lam = shrink * mean variance."""
    P, N = np.asarray(P, np.float64), np.asarray(N, np.float64)
    mp, mn = P.mean(0), N.mean(0)
    Z = np.concatenate([P - mp, N - mn])
    S = Z.T @ Z / len(Z)
    d = np.linalg.solve(S + shrink * np.trace(S) / len(S) * np.eye(len(S)), mp - mn)
    return d / np.linalg.norm(d)


def constant_direction(X, ridge=1e-2):
    """m with m . x ~ 1 over the rows of X (ridge least squares)."""
    X = np.asarray(X, np.float64)
    lam = ridge * (X ** 2).sum() / len(X)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ np.ones(len(X)))


def silu(z):
    return z / (1 + np.exp(-z))


def build(P, N, write, margin=1.0, k=4.0, shrink=0.1):
    """-> dict(gate, up, down, theta, g, d, m, recall, neg_fire) for a SiLU-gated unit (Qwen / Llama MLPs).
    theta = max over the negatives' scores + margin; g makes the median negative >= 12 units below threshold."""
    d = lda_direction(P, N, shrink)
    sn = np.asarray(N) @ d
    theta = float(sn.max() + margin)
    m = constant_direction(np.concatenate([P, N]))
    g = max(4.0, 12.0 / max(theta - float(np.median(sn)), 1e-3))
    gate_row, up_row = g * (d - theta * m), k * m
    pre_p, pre_n = np.asarray(P) @ gate_row, np.asarray(N) @ gate_row  # the unit's actual pre-activations
    on = pre_p > 0
    act = silu(pre_p) * (np.asarray(P) @ up_row)
    a = float(act[on].mean()) if on.any() else 1.0
    return {"gate": gate_row, "up": up_row, "down": np.asarray(write, np.float64) / a, "theta": theta, "g": g, "d": d,
            "m": m, "recall": float(on.mean()), "neg_fire": float((pre_n > 0).mean())}


def ops(prefix, layer, unit, arrays_file):
    """Spec ops writing the unit from arrays "gate", "up" (1 x d_model) and "down" (d_model x 1) in arrays_file."""
    t = f"{prefix}layers.{layer}.mlp."
    return [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": [unit], "values": f"{arrays_file}#gate"},
            {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": [unit], "values": f"{arrays_file}#up"},
            {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": [unit], "values": f"{arrays_file}#down"}]


def save(path, unit):
    """Write gate / up / down of a build() result to an npz in the layouts ops() expects."""
    np.savez(path, gate=unit["gate"][None].astype(np.float32), up=unit["up"][None].astype(np.float32),
             down=unit["down"][:, None].astype(np.float32))
