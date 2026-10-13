"""VAE-SRO: physics-informed variational autoencoders for chemical short-range
order in complex concentrated alloys.

See the top-level README for an overview.  Typical usage::

    from vae_sro.predict import EnsemblePredictor
    from vae_sro.features import binary_path

    predictor = EnsemblePredictor()
    comps = binary_path({"Ti": 1}, {"Cr": 1, "Fe": 1, "Ni": 1},
                        x_values=[0.0, 0.1, 0.2, 0.3])
    summary = predictor.predict(comps, n_samples=25)
"""
from .model import ELEMENTS, build_vae, compile_vae, loss_func  # noqa: F401

__all__ = ["ELEMENTS", "build_vae", "compile_vae", "loss_func"]
__version__ = "1.0.0"
