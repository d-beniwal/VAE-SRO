"""Ensemble prediction with uncertainty quantification.

The released model is an ensemble of 125 VAEs (5 shuffles x 5 runs x 5 folds).
Each VAE contains a stochastic latent-sampling layer, so a single forward pass
is one draw from the model's predictive distribution.  We therefore treat every
(model, draw) pair as one ensemble member and summarise the resulting
distribution by its median (central prediction) and percentile / standard-
deviation spread (predictive uncertainty).

Two variance sources are separated on request:
  * epistemic  -- spread of the *per-model means* across the 125 models;
  * aleatoric  -- mean *within-model* spread from latent sampling.

A global seed makes the whole procedure reproducible.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .data import list_model_weights, load_feature_stack
from .features import build_input_tensor, compositions_to_arrays
from .model import DEFAULT_LATENT_DIM, DEFAULT_LEARNING_RATE, compile_vae


def _symmetrise_mask(Y, masks):
    """Apply binary mask and symmetrise each predicted array about the diagonal."""
    out = np.empty_like(Y)
    for i, (y, m) in enumerate(zip(Y, masks)):
        ym = y * m
        out[i] = 0.5 * (ym + ym.T)
    return out


class EnsemblePredictor:
    """Load the 125-member VAE ensemble once and reuse it for many predictions."""

    def __init__(self, weight_files=None, latent_dim=DEFAULT_LATENT_DIM,
                 learning_rate=DEFAULT_LEARNING_RATE):
        self.weight_files = weight_files or list_model_weights()
        if not self.weight_files:
            raise FileNotFoundError("No ensemble weight files found in data/models.")
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self._model = None
        self._input_dim = None
        _, self.feature_stack_norm, self.feature_info = load_feature_stack()

    # -- low level -------------------------------------------------------
    def _ensure_model(self, input_dim):
        if self._model is None or self._input_dim != input_dim:
            tf.keras.backend.clear_session()
            self._model = compile_vae(input_dim, self.latent_dim, self.learning_rate)
            self._input_dim = input_dim

    def raw_predictions(self, x, masks, n_samples=25, seed=0):
        """Return all stochastic predictions.

        Returns
        -------
        preds : array ``(n_models, n_samples, n_alloys, 14, 14)``
        """
        tf.random.set_seed(seed)
        x = np.asarray(x, dtype="float32")
        masks = np.asarray(masks)
        input_dim = x.shape[1:]
        self._ensure_model(input_dim)
        n_models = len(self.weight_files)
        preds = np.zeros((n_models, n_samples, len(x), 14, 14), dtype="float32")
        for mi, wf in enumerate(self.weight_files):
            self._model.load_weights(wf)
            for si in range(n_samples):
                Y = np.squeeze(self._model.predict(x, verbose=0), axis=-1)
                preds[mi, si] = _symmetrise_mask(Y, masks)
        return preds

    # -- high level ------------------------------------------------------
    def predict(self, compositions, n_samples=25, seed=0):
        """Predict CSRO with uncertainty for a list of composition dicts.

        Returns a dict of ``(n_alloys, 14, 14)`` arrays:
        ``median, mean, std, q05, q25, q75, q95``, plus ``epistemic_std``,
        ``aleatoric_std`` and the boolean ``mask``.
        """
        cacb, cacb_norm, masks = compositions_to_arrays(compositions)
        x = build_input_tensor(cacb_norm, self.feature_stack_norm)
        preds = self.raw_predictions(x, masks, n_samples=n_samples, seed=seed)
        return summarise_predictions(preds, masks)

    def predict_x(self, x, masks, n_samples=25, seed=0):
        """Same as :meth:`predict` but from a pre-built input tensor."""
        preds = self.raw_predictions(x, masks, n_samples=n_samples, seed=seed)
        return summarise_predictions(preds, masks)

    def ensemble_median_x(self, x, masks, n_samples=8, seed=0):
        """Memory-light ensemble prediction for large batches (e.g. CoSMoR).

        For each model computes the Monte-Carlo mean over ``n_samples`` draws,
        then returns the median over models -- shape ``(n_inputs, 14, 14)``.
        Only the 125 per-model means are held in memory, not every draw.
        """
        tf.random.set_seed(seed)
        x = np.asarray(x, dtype="float32")
        masks = np.asarray(masks)
        self._ensure_model(x.shape[1:])
        n_models = len(self.weight_files)
        per_model_mean = np.zeros((n_models, len(x), 14, 14), dtype="float32")
        for mi, wf in enumerate(self.weight_files):
            self._model.load_weights(wf)
            acc = np.zeros((len(x), 14, 14), dtype="float64")
            for _ in range(n_samples):
                Y = np.squeeze(self._model.predict(x, verbose=0), axis=-1)
                acc += _symmetrise_mask(Y, masks)
            per_model_mean[mi] = acc / n_samples
        return np.median(per_model_mean, axis=0)


def summarise_predictions(preds, masks):
    """Collapse ``(n_models, n_samples, n, 14, 14)`` into summary statistics."""
    n_models, n_samples = preds.shape[:2]
    flat = preds.reshape(n_models * n_samples, *preds.shape[2:])  # (M*S, n,14,14)
    per_model_mean = preds.mean(axis=1)                            # (M, n,14,14)
    out = {
        "median": np.median(flat, axis=0),
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0),
        "q05": np.percentile(flat, 5, axis=0),
        "q25": np.percentile(flat, 25, axis=0),
        "q75": np.percentile(flat, 75, axis=0),
        "q95": np.percentile(flat, 95, axis=0),
        "epistemic_std": per_model_mean.std(axis=0),
        "aleatoric_std": preds.std(axis=1).mean(axis=0),
        "mask": masks.astype(bool),
        "n_members": n_models * n_samples,
    }
    return out


def pair_curve(summary, pair, elements=None):
    """Extract a per-composition curve for one element pair from a summary dict.

    Returns a dict of 1-D arrays keyed like the summary (median, q25, q75, ...).
    """
    from .model import ELEMENTS
    elements = elements or ELEMENTS
    a, b = sorted(pair)
    i, j = elements.index(a), elements.index(b)
    keys = ["median", "mean", "std", "q05", "q25", "q75", "q95",
            "epistemic_std", "aleatoric_std"]
    return {k: summary[k][:, i, j] for k in keys}
