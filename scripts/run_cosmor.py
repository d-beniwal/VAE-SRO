"""CoSMoR decomposition of the ML CSRO prediction along A_x-(B)_{1-x}.

Example
-------
    python scripts/run_cosmor.py --A W --B CrVTa --xmax 0.4 --dc 0.01 --outdir results/cosmor_W-CrVTa
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_sro.cosmor import run_cosmor, self_contribution_fraction  # noqa: E402
from scripts_common import parse_group  # type: ignore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--dc", type=float, default=0.01)
    ap.add_argument("--xmax", type=float, default=0.4)
    ap.add_argument("--dX", type=float, default=0.01)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    xs = np.round(np.arange(0, args.xmax + 1e-9, args.dc), 4)
    res = run_cosmor(parse_group(args.A), parse_group(args.B), xs,
                     dX=args.dX, n_samples=args.n_samples)
    outdir = args.outdir or f"cosmor_{args.A}-{args.B}"
    os.makedirs(outdir, exist_ok=True)
    for tag, df in res["clubbed_pairs"].items():
        df.to_excel(os.path.join(outdir, f"clubbed_binPairs-{tag}.xlsx"), index=False)
    for tag, df in res["clubbed_feats"].items():
        df.to_excel(os.path.join(outdir, f"clubbed_feats-{tag}.xlsx"), index=False)
    print("Self-contribution fractions:")
    for tag in res["clubbed_pairs"]:
        print(f"  {tag}: {self_contribution_fraction(res['clubbed_pairs'], tag):.3f}")
    print(f"Wrote CoSMoR results to {outdir}/")


if __name__ == "__main__":
    main()
