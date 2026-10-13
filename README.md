# VAE-SRO

**Physics-informed variational autoencoders for chemical short-range order (CSRO) in complex concentrated alloys.**

This repository accompanies the paper

> D. Beniwal, S. Tripathi, M. Meesa, N. Sardana, P. Singh, P. K. Ray,
> *Learning Emergent Short-Range Order in Complex Concentrated Alloys with Physics-Informed Autoencoders*,
> Acta Materialia, 122835 (2026). [doi.org/10.1016/j.actamat.2026.122835](https://doi.org/10.1016/j.actamat.2026.122835)

The model maps an alloy **composition** directly to the full matrix of first-nearest-neighbour
Warren–Cowley CSRO parameters for **all binary pairs simultaneously**. Each alloy is encoded as a
`14 × 14 × 9` tensor of binary-pair descriptors and decoded to a `14 × 14` CSRO matrix by an
ensemble of 125 variational autoencoders (VAEs). Predictions are interpreted with the authors'
**CoSMoR** framework.

---

## Why a VAE?

CSRO in a multicomponent alloy is *not separable* into isolated pair interactions — the CSRO of one
pair is shaped by the whole interaction landscape. We therefore learn a compact latent
representation of an alloy's **collective interaction state** and decode all pair parameters jointly,
rather than fitting independent per-pair regressors. The variational latent space (i) regularises a
tiny-data problem, (ii) yields ensemble-based **uncertainty estimates**, and (iii) offers a
generative pathway for future inverse design.

---

## Repository layout

```
VAE-SRO/
├── vae_sro/                 # importable package
│   ├── model.py             # VAE architecture + loss (Keras 2)
│   ├── features.py          # composition → (14×14×9) input tensor (dependency-free)
│   ├── data.py              # loaders for the shipped database / weights
│   ├── dataset.py           # recover discrete-alloy compositions + DFT points on a path
│   ├── predict.py           # EnsemblePredictor: Monte-Carlo ensemble + uncertainty
│   ├── cosmor.py            # CoSMoR interpretation (local partial dependence)
│   ├── enthalpy.py          # Miedema mixing enthalpy from the feature stack
│   ├── metrics.py           # MAE / ordering-vs-clustering classification
│   └── plotting.py          # publication-quality figure style + helpers
├── scripts/                 # command-line drivers
│   ├── train_ensemble.py            # (re)train the VAE ensemble on updated data
│   ├── predict_composition_path.py  # CSRO + uncertainty bands along a pseudo-binary path
│   ├── run_cosmor.py                # CoSMoR decomposition for a system
│   ├── compute_cv_uncertainty.py    # ensemble CV metrics + uncertainty (Fig. 3)
│   ├── run_baselines.py             # baseline model comparison
│   └── run_holdout.py               # leave-one-system/element-out extrapolation
├── data/
│   ├── database/            # alloy arrays, 9-channel feature stack, metadata, raw SRO table
│   ├── models/              # 125 trained VAE weight files (*.h5)
│   └── training_logs/       # per-shuffle CV predictions + shuffled splits
├── notebooks/               # step-by-step database construction + training + plotting
├── results/                 # example generated outputs
└── docs/                    # extended notes (featurisation, extending the element space)
```

---

## Installation

```bash
conda env create -f environment.yml     # creates env "vae-sro" (Python 3.10, TF 2.15 / Keras 2)
conda activate vae-sro
# or:  pip install -r requirements.txt
```

> The released weights were trained with TensorFlow/Keras **2.8**; they load correctly under the
> pinned **2.15** (both are Keras 2). TensorFlow ≥ 2.16 defaults to Keras 3 and is **not**
> compatible with these weight files.

---

## Quick start

```python
from vae_sro.predict import EnsemblePredictor, pair_curve
from vae_sro.features import binary_path

predictor = EnsemblePredictor()                      # loads the 125-model ensemble

# CSRO along Ti_x-(CrFeNi)_{1-x}
xs    = [i/100 for i in range(0, 51)]
comps = binary_path({"Ti": 1}, {"Cr": 1, "Fe": 1, "Ni": 1}, xs)
summary = predictor.predict(comps, n_samples=20, seed=0)   # median + percentile bands

ni_ti = pair_curve(summary, ("Ni", "Ti"))
print(ni_ti["median"], ni_ti["q25"], ni_ti["q75"])
```

### Prediction & uncertainty

The 125 VAEs each contain a stochastic latent-sampling layer, so one forward pass is a **draw**
from the predictive distribution. `EnsemblePredictor` runs `n_samples` draws per model under a fixed
global seed and reports:

* **median / mean** – central prediction (reproducible run-to-run),
* **q05 / q25 / q75 / q95, std** – predictive uncertainty bands,
* **epistemic_std** – model-to-model spread, **aleatoric_std** – within-model (sampling) spread.

---

## Reproducing the paper

```bash
python scripts/compute_cv_uncertainty.py        # → MAE 0.086, accuracy 94.2%, 14/240 misclassified
python scripts/predict_composition_path.py --system Ti-CrFeNi   # Fig. 4 curves + bands
python scripts/run_cosmor.py --A Ti --B CrFeNi                  # CoSMoR decomposition
python scripts/run_baselines.py                 # VAE vs CNN / det-AE / tree baselines
python scripts/run_holdout.py                   # leave-one-system / leave-one-element-out
```

## Data

* **14-element vocabulary:** Al, Ti, V, Cr, Mn, Fe, Co, Ni, Zr, Nb, Mo, Hf, Ta, W.
* **9 physical binary features** (Table 1 of the paper): Miedema `H_chem`, `H_el`; composition-average
  `VEC`, `spin magnetic moment`; local mismatch of Allen electronegativity, `n_WS`, metallic radius,
  covalent radius, and cohesive energy.
* **CSRO labels** are from DFT thermodynamic linear-response theory (Singh et al.); see
  `data/database/db-sro-raw-without-binary.xlsx`.

## Extending the element space

Adding a new element means (i) growing the arrays from `14×14×9` to `15×15×9` by appending the new
element's row/column of the nine descriptors (regenerate with `mapal`), (ii) adding alloys that
contain it, and (iii) retraining via `scripts/train_ensemble.py` (or warm-starting from the current
weights for transfer learning). See `docs/extending.md`.

## Citation

If you use this code, please cite the paper above

```bibtex
@article{beniwal2026emergent,
  title   = {Learning Emergent Short-Range Order in Complex Concentrated Alloys with Physics-Informed Autoencoders},
  author  = {Beniwal, Dishant and Tripathi, Shrish and Meesa, Maheshwari and Sardana, Neha and Singh, Prashant and Ray, Pratik K.},
  journal = {Acta Materialia},
  pages   = {122835},
  year    = {2026},
  doi     = {10.1016/j.actamat.2026.122835}
}
```

and the CoSMoR and MAPAL methods
(Beniwal & Ray, *Phys. Rev. Mater.* **7**, 043802 (2023); *Comput. Mater. Sci.* **262**, 114360 (2026)).

## License

MIT — see [LICENSE](LICENSE).
