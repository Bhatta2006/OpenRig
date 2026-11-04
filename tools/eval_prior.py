#!/usr/bin/env python3
"""Pose-completion benchmark on held-out mocap subjects.

The solver is given only where the hands, feet, hips and head top are in a real captured pose
(starting from that clip's first, near-T-pose frame) and must place everything else. We measure
how far the unconstrained joints (elbows, knees, shoulders, chest, head) end up from the truth,
with and without the learned prior.

    python tools/eval_prior.py <bvh dir> [weights.npz] [n_frames]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

import bvh
from cmu_rig import SK, particles
from openrig.core.prior import PosePrior
from openrig.core.solver import Solver, SolveSettings

HOLDOUT = (10, 35, 143)
CONSTRAINED = ["hand.L", "hand_end.L", "hand.R", "hand_end.R", "foot.L", "toe.L", "toe_end.L",
               "foot.R", "toe.R", "toe_end.R", "hips", "head_top"]
MEASURED = {"elbows": ["forearm.L", "forearm.R"], "knees": ["shin.L", "shin.R"],
            "shoulders": ["upperarm.L", "upperarm.R"], "chest": ["neck0"], "head": ["head"],
            "mid-spine": ["spine1"]}


def clips(root, n):
    out = []
    for sub in HOLDOUT:
        for f in sorted(pathlib.Path(root).glob(f"{sub:03d}/*.bvh"))[:12]:
            j, ft, d = bvh.load(f, stride=10)
            pos, tail = bvh.fk(j, d)
            P = particles(j, pos, tail)
            if len(P) > 12:
                out.append(P)
    rng = np.random.default_rng(0)
    pairs = [(c, i) for c in range(len(out)) for i in range(5, len(out[c]), 6)]
    pick = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return out, [pairs[k] for k in sorted(pick)]


def evaluate(root, prior, strengths, n):
    data, pairs = clips(root, n)
    ix = SK.index
    res = {s: {k: [] for k in MEASURED} for s in strengths}
    for c, i in pairs:
        P0, G = data[c][0], data[c][i]
        torso = np.linalg.norm(G[ix["neck0"]] - G[ix["hips"]])
        targets = {k: G[ix[k]] for k in CONSTRAINED}
        for s in strengths:
            st = SolveSettings(iterations=30, polish=60, prior=prior if s else None, prior_strength=s or 0.0)
            P = Solver(SK, P0, st).solve(targets, set())
            for grp, names in MEASURED.items():
                res[s][grp].append(np.mean([np.linalg.norm(P[ix[nm]] - G[ix[nm]]) for nm in names]) / torso)
    return res, len(pairs)


def main(root, weights=None, n=300):
    prior = PosePrior.load(weights) if weights else PosePrior.load()
    strengths = [0.0, 0.3, 0.6, 1.0] if prior else [0.0]
    res, n = evaluate(root, prior, strengths, int(n))
    print(f"{n} held-out poses (subjects {HOLDOUT}); error as % of torso length, lower is better")
    print("%-12s" % "joint" + "".join("%12s" % ("prior " + str(s) if s else "no prior") for s in strengths))
    for grp in MEASURED:
        print("%-12s" % grp + "".join("%11.1f%%" % (100 * np.mean(res[s][grp])) for s in strengths))
    tot = {s: 100 * np.mean([np.mean(v) for v in res[s].values()]) for s in strengths}
    print("%-12s" % "AVERAGE" + "".join("%11.1f%%" % tot[s] for s in strengths))


if __name__ == "__main__":
    main(*sys.argv[1:])
