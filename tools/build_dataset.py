#!/usr/bin/env python3
"""Turn a folder of CMU-style BVH files into pose-prior features.

    python tools/build_dataset.py <bvh dir> features.npz [frame stride] [holdout subjects...]
"""
import sys, pathlib, re, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import bvh
from cmu_rig import SK, particles
from openrig.core.prior import Canon, features, forward_sign

CANON = Canon(SK)


def subject_of(path: pathlib.Path) -> int:
    m = re.match(r"(\d+)_", path.name) or re.match(r"(\d+)", path.parent.name)
    return int(m.group(1)) if m else -1


def one(args):
    f, stride = args
    try:
        joints, ft, data = bvh.load(f, stride=stride)
        pos, tail = bvh.fk(joints, data)
        P = particles(joints, pos, tail)
    except Exception:  # a few CMU files have odd skeletons
        return None
    sgn = [forward_sign(CANON, p) for p in P[:: max(1, len(P) // 20)]]
    sign = 1.0 if sum(sgn) >= 0 else -1.0
    X = np.stack([features(CANON, p, sign) for p in P]).astype(np.float32)
    return X, np.full(len(P), subject_of(f), dtype=np.int32), sign


def main(root, dst, stride=12, *holdout):
    import multiprocessing as mp
    holdout = [int(h) for h in holdout] or [10, 35, 143]
    files = sorted(pathlib.Path(root).rglob("*.bvh"))
    print(len(files), "files", flush=True)
    X, S, signs, bad = [], [], collections.Counter(), 0
    with mp.Pool(4) as pool:
        for n, r in enumerate(pool.imap(one, [(f, int(stride)) for f in files], chunksize=8)):
            if r is None:
                bad += 1
                continue
            X.append(r[0]); S.append(r[1]); signs[r[2]] += 1
            if n % 200 == 0:
                print(n, sum(len(x) for x in X), "frames", flush=True)
    X, S = np.concatenate(X), np.concatenate(S)
    print("frames", len(X), "skipped files", bad, "forward sign votes", dict(signs))
    np.savez_compressed(dst, X=X, subject=S, holdout=np.array(holdout))


if __name__ == "__main__":
    main(*sys.argv[1:])
