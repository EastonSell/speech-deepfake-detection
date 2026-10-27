"""Frozen synthetic-only rank-matched deletion control; no model loading."""
import json
import numpy as np


def rank(a):
    sv = np.linalg.svd(a, compute_uv=False)
    return int((sv > sv[0] * max(a.shape) * np.finfo(a.dtype).eps).sum()) if len(sv) and sv[0] else 0


def random_row_projector(v, k, seed):
    q = np.linalg.qr(np.random.default_rng(seed).normal(size=(v.shape[1], k)))[0]
    directions = v @ q
    return np.eye(v.shape[0]) - directions @ directions.T


def validate_projector(p, deletion_rank):
    assert np.allclose(p, p.T, atol=1e-12, rtol=0)
    assert np.allclose(p @ p, p, atol=1e-12, rtol=0)
    assert rank(np.eye(p.shape[0]) - p) == deletion_rank


def main():
    rng = np.random.default_rng(42); d = 128; r = 8; k = 4
    basis = np.linalg.qr(rng.normal(size=(d, r)))[0]
    def data(n):
        z = rng.normal(size=(n, r)); return z @ basis.T, (z[:, 0] >= 0).astype(int)
    X, y = data(40); _, _ = data(100); T, ty = data(100)
    mu = X.mean(0); X = X - mu; T = T - mu
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[:r].T
    def score(P):
        validate_projector(P, k)
        a = X @ P; b = T @ P
        w = np.linalg.solve(a.T @ a + 0.1 * np.eye(d), a.T @ (2*y - 1))
        pred = (b @ w >= 0).astype(int)
        assert np.isfinite(w).all() and len(pred) == len(ty) == 100
        return {'train_rank': rank(a), 'train_removed_energy_fraction': float(1 - np.sum(a*a)/np.sum(X*X)),
                'correct': int((pred == ty).sum()), 'total': len(ty), 'accuracy': float((pred == ty).mean())}
    pca = score(np.eye(d) - vt[:k].T @ vt[:k])
    controls = [{'seed': seed, **score(random_row_projector(v, k, seed))} for seed in range(10,20)]
    assert pca['train_rank'] == 4 and all(x['train_rank'] == 4 for x in controls)
    print(json.dumps({'synthetic_only': True, 'pca_delete_4': pca, 'row_space_random_delete_4': controls}, indent=2))

if __name__ == '__main__': main()
