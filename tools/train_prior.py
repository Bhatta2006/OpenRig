#!/usr/bin/env python3
"""Train the denoising pose prior (pure numpy) on features built by build_dataset.py.

    python tools/train_prior.py features.npz openrig/data/pose_prior.npz
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from openrig.core.prior import mirror_features

SIZES = [39, 128, 128, 24, 128, 128, 39]


def unit(x):
    x = x.reshape(x.shape[0], -1, 3)
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)).reshape(x.shape[0], -1)


def corrupt(X, rng):
    """Noisy versions of clean poses (what a constraint solver tends to emit): every direction
    jittered by up to ~17 degrees. The prior learns which joint-angle combinations are real."""
    sigma = rng.uniform(0.02, 0.30, size=(X.shape[0], 1))
    return unit(X + rng.normal(size=X.shape) * sigma / np.sqrt(3))


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(size=(a, b)) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x, keep=False):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return (h, acts) if keep else h

    def grads(self, x, target):
        out, acts = self.forward(x, keep=True)
        err = out - target
        loss = float((err ** 2).mean())
        g = 2 * err / err.size
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.append(acts[i].T @ g)
            gb.append(g.sum(0))
            if i:
                g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
        return loss, gW[::-1], gb[::-1]


def main(src, dst, epochs=60, batch=1024, seed=0):
    rng = np.random.default_rng(seed)
    z = np.load(src)
    X, subj = z["X"], z["subject"]
    held = np.isin(subj, z["holdout"])
    Xtr, Xva = X[~held], X[held]
    Xtr = np.concatenate([Xtr, mirror_features(Xtr)])
    print("train", Xtr.shape, "val", Xva.shape, flush=True)
    mean = Xtr.mean(0)
    net = MLP(SIZES, rng)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, t0 = 0, time.time()
    steps_total = epochs * (len(Xtr) // batch)
    vrng = np.random.default_rng(123)
    Vnoisy = corrupt(Xva, vrng)

    def val():
        out = net.forward(Vnoisy - mean) + mean
        return float(((unit(out) - Xva) ** 2).mean()), float(((Vnoisy - Xva) ** 2).mean())

    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for k in range(len(Xtr) // batch):
            xb = Xtr[perm[k * batch:(k + 1) * batch]]
            xin = corrupt(xb, rng) - mean
            loss, gW, gb = net.grads(xin, xb - mean)
            lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * step / steps_total)) + 1e-5
            step += 1
            for i, g in enumerate(gW + gb):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                params[i] -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        if ep % 5 == 0 or ep == epochs - 1:
            after, before = val()
            print(f"epoch {ep:3d} loss {loss:.5f}  val mse noisy {before:.5f} -> projected {after:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    out = {"n_layers": np.array(len(net.W)), "mean": mean.astype(np.float32)}
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        out[f"W{i}"], out[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    np.savez_compressed(dst, **out)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
