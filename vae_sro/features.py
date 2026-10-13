"""Composition -> model-input featurisation (dependency-free).

Every alloy is represented by a 14x14 ``cAcB`` array (product of atomic
fractions for each present pair) which is min-max normalised and multiplied,
layer by layer, into the pre-computed 9-channel binary-feature stack to form the
``(14, 14, 9)`` input tensor.

The 9 physical binary features themselves are pre-computed for the full
14-element space and shipped in ``data/database`` (``binary-feature-stack*.npy``
and ``binary-feature-info.json``); they were generated with the ``mapal``
library.  Because that stack is fixed for the 14-element vocabulary, no external
dependency is required at prediction time.  To extend the model to new elements,
regenerate the stack with ``mapal`` (see ``docs``).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .model import ELEMENTS

CACB_MIN, CACB_MAX = 0.0, 0.25  # normalisation constants for the cAcB array


def composition_to_cacb(composition):
    """Convert a ``{element: atomic_fraction}`` dict to arrays.

    Returns
    -------
    (cacb, cacb_norm, mask) : each of shape ``(14, 14)``
        ``cacb[i, j] = c_i * c_j`` for present off-diagonal pairs, 0 otherwise;
        ``cacb_norm = cacb / 0.25``; ``mask`` is 1 where a pair is present.
    """
    c = np.zeros(len(ELEMENTS))
    for el, frac in composition.items():
        c[ELEMENTS.index(el)] = frac
    present = [ELEMENTS.index(el) for el in composition]
    cacb = np.zeros((14, 14))
    mask = np.zeros((14, 14))
    for a in present:
        for b in present:
            if a != b:
                cacb[a, b] = c[a] * c[b]
                mask[a, b] = 1.0
    cacb_norm = (cacb - CACB_MIN) / (CACB_MAX - CACB_MIN)
    return cacb, cacb_norm, mask


def compositions_to_arrays(compositions):
    """Vectorised :func:`composition_to_cacb` over a list of dicts."""
    n = len(compositions)
    cacb = np.zeros((n, 14, 14))
    cacb_norm = np.zeros((n, 14, 14))
    mask = np.zeros((n, 14, 14))
    for k, comp in enumerate(compositions):
        cacb[k], cacb_norm[k], mask[k] = composition_to_cacb(comp)
    return cacb, cacb_norm, mask


def build_input_tensor(cacb_norm, feature_stack_norm):
    """Combine normalised cAcB arrays with the normalised feature stack.

    Parameters
    ----------
    cacb_norm : array ``(n, 14, 14)``
    feature_stack_norm : array ``(9, 14, 14)``

    Returns
    -------
    x : array ``(n, 14, 14, 9)`` ready for :meth:`model.predict`.
    """
    ft = feature_stack_norm.T  # (14, 14, 9)
    n = len(cacb_norm)
    x = np.zeros((n, 14, 14, ft.shape[-1]))
    for i in range(n):
        for j in range(ft.shape[-1]):
            x[i, :, :, j] = cacb_norm[i] * ft[:, :, j]
    return x


def binary_path(a_group, b_group, x_values):
    """Build compositions along a pseudo-binary path ``A_x - (B)_{1-x}``.

    ``a_group`` / ``b_group`` are dicts of *relative* proportions, e.g.
    ``binary_path({"Ti": 1}, {"Cr": 1, "Fe": 1, "Ni": 1}, x_values)`` yields
    Ti_x-(CrFeNi)_{1-x} with Cr=Fe=Ni=(1-x)/3.
    """
    a_tot = sum(a_group.values())
    b_tot = sum(b_group.values())
    comps = []
    for x in x_values:
        comp = {}
        for el, w in a_group.items():
            comp[el] = comp.get(el, 0.0) + x * w / a_tot
        for el, w in b_group.items():
            comp[el] = comp.get(el, 0.0) + (1 - x) * w / b_tot
        comps.append({el: f for el, f in comp.items() if f > 0})
    return comps


def all_pairs(elements=ELEMENTS):
    """All unordered element pairs in the vocabulary (as sorted tuples)."""
    return [tuple(sorted(p)) for p in combinations(elements, 2)]
