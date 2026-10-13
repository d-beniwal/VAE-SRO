"""Small shared helpers for the command-line scripts."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_sro.model import ELEMENTS  # noqa: E402


def parse_group(text):
    """'CrFeNi' -> {'Cr':1,'Fe':1,'Ni':1} (equal proportions)."""
    toks = re.findall(r"[A-Z][a-z]?", text)
    bad = [t for t in toks if t not in ELEMENTS]
    if bad:
        raise ValueError(f"Unknown element(s): {bad}")
    return {t: 1 for t in toks}
