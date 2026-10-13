"""(Re)train the VAE ensemble on the shipped (or an updated) database.

Implements the paper's automated protocol: for each of ``--shuffles`` dataset
shuffles, run ``--runs`` fresh initialisations, each with ``--folds``-fold cross
validation; the best epoch per fold (min validation misclassification, tie-broken
by validation MAE) is kept.  This yields shuffles x runs x folds models
(125 by default) whose ensemble median is the final prediction.

This is a streamlined, package-based version of the authors' ``run-auto-vae.py``.

Example
-------
    python scripts/train_ensemble.py --shuffles 5 --runs 5 --folds 5 \
        --epochs 2000 --save-every 100 --outdir trained-models-new
"""
import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as sk_shuffle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob  # noqa: E402
import tensorflow as tf  # noqa: E402
from vae_sro.model import compile_trainable_vae as compile_vae, ELEMENTS  # noqa: E402
from vae_sro.data import load_training_data, load_alloy_arrays, DATABASE_DIR  # noqa: E402
from vae_sro.metrics import iter_present_pairs  # noqa: E402


def predict_masked(model, x, masks):
    Y = np.squeeze(model.predict(x, verbose=0), axis=-1)
    return np.array([0.5 * (y * m + (y * m).T) for y, m in zip(Y, masks)])


def fold_metrics(Y_true, Y_pred, masks):
    mis = 0; n = 0; abserr = 0.0
    for Yt, Yp, m in zip(Y_true, Y_pred, masks):
        for r, c in iter_present_pairs(m):
            t, p = Yt[r, c], Yp[r, c]
            mis += int(t * p < 0); abserr += abs(t - p); n += 1
    return mis, (abserr / n if n else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--outdir", default="trained-models-new")
    ap.add_argument("--database-dir", default=None,
                     help="Directory holding data-x-Y-for-training.npz and "
                          "alloys-data-arrays.npz (default: the shipped %s). "
                          "Point this at an extended database (see "
                          "docs/extending.md) to train on a new element "
                          "vocabulary/grid size." % DATABASE_DIR)
    ap.add_argument("--warm-start-dir", default=None,
                     help="Directory of legacy ensemble weight files (*.h5) to "
                          "warm-start every fold from before fine-tuning, e.g. "
                          "the shipped data/models. Loaded with "
                          "by_name=True, skip_mismatch=True, so convolutional/"
                          "batch-norm weights transfer across a changed grid "
                          "size and only the shape-dependent dense layers "
                          "(encoder_mu, encoder_log_variance, decoder_dense_1) "
                          "fall back to random init. Omit for training from "
                          "scratch.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    database_dir = args.database_dir or DATABASE_DIR
    x0, Y0 = load_training_data(database_dir=database_dir)
    arr = load_alloy_arrays(database_dir=database_dir)
    masks0 = arr["binary_masks"]
    x0, Y0 = x0.astype("float32"), Y0.astype("float32")

    warm_start_weights = (sorted(glob.glob(os.path.join(args.warm_start_dir, "*.h5")))
                           if args.warm_start_dir else [])
    if args.warm_start_dir and not warm_start_weights:
        raise FileNotFoundError(f"No *.h5 weight files found in {args.warm_start_dir}")

    fold_counter = 0
    for s in range(args.shuffles):
        x, Y, masks = sk_shuffle(x0, Y0, masks0, random_state=s)
        for run in range(args.runs):
            kf = KFold(n_splits=args.folds)
            for k, (tr, te) in enumerate(kf.split(x)):
                tf.keras.backend.clear_session()
                tf.random.set_seed(1000 * s + 100 * run + k)
                model = compile_vae(x.shape[1:], learning_rate=args.lr)
                if warm_start_weights:
                    src = warm_start_weights[fold_counter % len(warm_start_weights)]
                    model.load_weights(src, by_name=True, skip_mismatch=True)
                fold_counter += 1
                best = dict(mis=1e9, mae=1e9, epoch=-1, weights=None)
                for ep in range(args.save_every, args.epochs + 1, args.save_every):
                    model.fit(x[tr], Y[tr], epochs=args.save_every, verbose=0)
                    pred = predict_masked(model, x[te], masks[te])
                    mis, mae = fold_metrics(np.squeeze(Y[te]), pred, masks[te])
                    if (mis, mae) < (best["mis"], best["mae"]):
                        best.update(mis=mis, mae=mae, epoch=ep,
                                    weights=[w.copy() for w in model.get_weights()])
                model.set_weights(best["weights"])
                fn = f"M{s+1}-R{run+1}-K{k+1}-itr{best['epoch']}-weights.h5"
                model.save_weights(os.path.join(args.outdir, fn))
                print(f"{fn}: val_misclass={best['mis']} val_mae={best['mae']:.3f}", flush=True)
    print(f"\nDONE. Wrote ensemble to {args.outdir}/")


if __name__ == "__main__":
    main()
