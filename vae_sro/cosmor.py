"""CoSMoR (Compositional Stimulus and Model Response) interpretation for the
pair-CSRO VAE ensemble.

For a pseudo-binary path ``A_x-(B)_{1-x}`` CoSMoR decomposes the predicted change
in each target pair's CSRO into contributions from every (source-pair, feature)
variable, by finite-difference local partial dependence integrated along the
path.  See Beniwal & Ray, Phys. Rev. Mater. 7, 043802 (2023).

This is a faithful port of the authors' ``run-cosmor-sro.py`` built on top of the
reproducible :class:`~vae_sro.predict.EnsemblePredictor`.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .features import binary_path, compositions_to_arrays, build_input_tensor
from .model import ELEMENTS
from .predict import EnsemblePredictor

FEATURE_SHORT = ["H_chem", "H_el", "cAvg_VEC", "cAvg_spin_mag_moment",
                 "LM_EN_Allen", "LM_nWS", "LM_r_met", "LM_r_cov", "LM_E"]


def run_cosmor(a_group, b_group, xs, dX=0.01, n_samples=8, seed=0,
               predictor=None):
    """Run CoSMoR along ``A_x-(B)_{1-x}``.

    Returns a dict keyed by target pair 'A-B' -> DataFrame of cumulative
    contributions (columns 'srcpair|feat'), plus 'x', 'ML-SRO'.
    Also returns ``clubbed_pairs`` and ``clubbed_feats`` dicts.
    """
    predictor = predictor or EnsemblePredictor()
    feat_norm = predictor.feature_stack_norm

    comps = binary_path(a_group, b_group, xs)
    els_in_sys = sorted({e for c in comps for e in c},
                        key=lambda e: ELEMENTS.index(e))
    _, cacb_norm, masks = compositions_to_arrays(comps)
    X = build_input_tensor(cacb_norm, feat_norm)          # (n_comps,14,14,9)
    n_comps = len(X)

    pairs = [tuple(sorted(p)) for p in combinations(els_in_sys, 2)]
    pair_idx = {p: (ELEMENTS.index(p[0]), ELEMENTS.index(p[1])) for p in pairs}
    n_pairs, n_feats = len(pairs), len(FEATURE_SHORT)
    n_vars = n_pairs * n_feats

    # ---- baseline predictions -------------------------------------------------
    # Use the full-system mask (union over the path) so pairs involving the
    # varied element A are retained at every composition.
    mask2d = (masks.sum(axis=0) > 0).astype("float32")
    base = predictor.ensemble_median_x(X, np.repeat(mask2d[None], n_comps, 0),
                                       n_samples=n_samples, seed=seed)

    # ---- build all single-feature perturbations ------------------------------
    var_records = []      # (comp_id, pair, feat)
    X_vars = np.zeros((n_comps * n_vars, *X.shape[1:]), dtype="float32")
    v = 0
    for ci in range(n_comps):
        for p in pairs:
            i, j = pair_idx[p]
            for fi in range(n_feats):
                Xv = X[ci].copy()
                Xv[i, j, fi] += dX
                Xv[j, i, fi] += dX
                X_vars[v] = Xv
                var_records.append((ci, p, FEATURE_SHORT[fi]))
                v += 1
    pred_vars = predictor.ensemble_median_x(
        X_vars, np.repeat(mask2d[None], len(X_vars), 0), n_samples=n_samples, seed=seed)

    # ---- local partial dependence -------------------------------------------
    # lpd[v, target_pair] = (pred_vars[v] - base[comp]) / dX   at target index
    x_value = np.zeros(len(X_vars))
    lpd = np.zeros((len(X_vars), n_pairs))
    for v, (ci, p, fshort) in enumerate(var_records):
        i, j = pair_idx[p]
        fi = FEATURE_SHORT.index(fshort)
        x_value[v] = X[ci, i, j, fi]
        d = (pred_vars[v] - base[ci]) / dX
        for ti, tp in enumerate(pairs):
            ii, jj = pair_idx[tp]
            lpd[v, ti] = d[ii, jj]

    var_names = [f"{p[0]}-{p[1]}|{f}" for p in pairs for f in FEATURE_SHORT]

    # ---- integrate contributions along the path ------------------------------
    results, clubbed_pairs, clubbed_feats = {}, {}, {}
    for ti, tp in enumerate(pairs):
        ii, jj = pair_idx[tp]
        Y = base[:, ii, jj]
        ybase = Y[0]
        contri = np.zeros((n_comps, n_vars))
        for ci in range(1, n_comps):
            xi = x_value[ci * n_vars:(ci + 1) * n_vars]
            xp = x_value[(ci - 1) * n_vars:ci * n_vars]
            lp = lpd[(ci - 1) * n_vars:ci * n_vars, ti]
            contri[ci] = (xi - xp) * lp
        cum = np.cumsum(contri, axis=0) + ybase
        tag = f"{tp[0]}-{tp[1]}"
        df = pd.DataFrame({"x": xs, "ML-SRO": Y})
        df[var_names] = cum
        results[tag] = df

        # club by source pair and by feature
        cp = pd.DataFrame({"x": xs, "ML-SRO": Y})
        for p in pairs:
            cols = [c for c in var_names if c.startswith(f"{p[0]}-{p[1]}|")]
            cp[f"{p[0]}-{p[1]}"] = (df[cols] - ybase).sum(axis=1) + ybase
        clubbed_pairs[tag] = cp

        cf = pd.DataFrame({"x": xs, "ML-SRO": Y})
        for f in FEATURE_SHORT:
            cols = [c for c in var_names if c.endswith(f"|{f}")]
            cf[f] = (df[cols] - ybase).sum(axis=1) + ybase
        clubbed_feats[tag] = cf

    return dict(pairs=pairs, results=results,
                clubbed_pairs=clubbed_pairs, clubbed_feats=clubbed_feats)


def self_contribution_fraction(clubbed_pairs, target_pair):
    """Path-mean |self contribution| / sum |all pair contributions| for a target."""
    df = clubbed_pairs[target_pair]
    pair_cols = [c for c in df.columns if c not in ("x", "ML-SRO")]
    ybase = df[pair_cols].iloc[0].mean()
    contri = (df[pair_cols] - ybase).abs()
    denom = contri.sum(axis=1)
    frac = (contri[target_pair] / denom).iloc[1:]
    return float(frac.mean())
