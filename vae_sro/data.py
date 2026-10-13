"""Loaders for the shipped database (arrays, feature stack, metadata)."""
from __future__ import annotations

import json
import os

import numpy as np

from .model import ELEMENTS

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(_PKG_ROOT, "data", "database")
MODELS_DIR = os.path.join(_PKG_ROOT, "data", "models")


def load_feature_stack(database_dir=DATABASE_DIR):
    """Return (stack, stack_norm, info) for the 9-channel binary features."""
    stack = np.load(os.path.join(database_dir, "binary-feature-stack.npy"))
    stack_norm = np.load(os.path.join(database_dir, "binary-feature-stack-norm.npy"))
    with open(os.path.join(database_dir, "binary-feature-info.json")) as f:
        info = json.load(f)
    return stack, stack_norm, info


def load_training_data(database_dir=DATABASE_DIR):
    """Return the model-ready training tensors ``x`` (n,14,14,9) and ``Y`` (n,14,14,1)."""
    d = np.load(os.path.join(database_dir, "data-x-Y-for-training.npz"))
    return d["x"], d["Y"]


def load_alloy_arrays(database_dir=DATABASE_DIR):
    """Return the per-alloy arrays dict (cAcB, SRO, masks, names, ...)."""
    d = np.load(os.path.join(database_dir, "alloys-data-arrays.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_data_info(database_dir=DATABASE_DIR):
    """Return (elements, alloy_names) metadata."""
    with open(os.path.join(database_dir, "alloys-data-info.json")) as f:
        info = json.load(f)
    return info["elements"], info["alloy_names"]


def load_raw_sro(database_dir=DATABASE_DIR):
    """Return the raw tabular (alloy_name, binary_name, SRO) dataframe if present."""
    import pandas as pd
    path = os.path.join(database_dir, "db-sro-raw-without-binary.xlsx")
    return pd.read_excel(path)


def list_model_weights(models_dir=MODELS_DIR):
    """Sorted list of all ensemble weight files."""
    import glob
    return sorted(glob.glob(os.path.join(models_dir, "*.h5")))
