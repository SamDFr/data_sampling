# sampling_analysis_utils.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from ase.neighborlist import NeighborList

# ---------- IO ----------

def load_traj(paths: Union[str, Sequence[str]]):
    """
    Read one or many trajectory files and FLATTEN into List[Atoms].
    Accepts a single path or a list/tuple of paths.
    """
    if isinstance(paths, (list, tuple)):
        images = []
        for p in paths:
            imgs = read(p, ":")
            # imgs may be a single Atoms or a list
            if isinstance(imgs, Atoms):
                images.append(imgs)
            else:
                images.extend(imgs)
        return images
    else:
        imgs = read(paths, ":")
        return [imgs] if isinstance(imgs, Atoms) else imgs

# ---------- ENERGY / FORCES ----------

def energies(images: List[Atoms]) -> np.ndarray:
    """
    Return per-structure potential energies if available.
    Missing values become NaN.
    """
    out = []
    for a in images:
        try:
            out.append(a.get_potential_energy())
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def force_components(images: List[Atoms]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate forces and symbols across all frames.
    Returns (F_all, symbols_all), where:
      F_all shape = (sum_i n_i, 3), symbols_all shape = (sum_i n_i,)
    Missing forces become NaN rows.
    """
    Fs, syms = [], []
    for a in images:
        try:
            F = a.get_forces(apply_constraint=False)
        except Exception:
            F = np.full((len(a), 3), np.nan)
        Fs.append(F)
        syms += a.get_chemical_symbols()
    return np.vstack(Fs), np.array(syms, dtype=object)

def force_norms_by_species(images: List[Atoms]) -> Dict[str, np.ndarray]:
    """
    Return {species: |F| vector} across all frames.
    """
    F, sym = force_components(images)
    norms = np.linalg.norm(F, axis=1)
    out: Dict[str, List[float]] = {}
    for s, f in zip(sym, norms):
        out.setdefault(s, []).append(f)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}

def frame_rms_force(images: List[Atoms], species_filter: Optional[Iterable[str]] = None) -> np.ndarray:
    """
    Per-frame RMS force (optionally on a species subset).
    """
    vals = []
    for a in images:
        try:
            F = a.get_forces(apply_constraint=False)
        except Exception:
            vals.append(np.nan); continue
        if species_filter:
            mask = np.isin(a.get_chemical_symbols(), list(species_filter))
            F = F[mask]
        vals.append(np.sqrt((F**2).sum(axis=1).mean()) if len(F) else np.nan)
    return np.array(vals, dtype=float)

# ---------- RDF / COORDINATION ----------

def pair_rdf(images: List[Atoms], pair: Tuple[str, str], r_max: float = 8.0, bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple pair-distance histogram for given species pair (A,B).
    Returns (r_centers, counts_norm) without full g(r) normalization.
    Good for relative comparison across samplings.
    """
    A, B = pair
    hist = np.zeros(bins, dtype=float)
    edges = np.linspace(0.0, r_max, bins + 1)
    for a in images:
        sym = np.array(a.get_chemical_symbols(), dtype=object)
        ia = np.where(sym == A)[0]; ib = np.where(sym == B)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        # neighbor list with generous cutoff
        cuts = np.full(len(a), r_max / 2.0)
        nl = NeighborList(cuts, bothways=True, self_interaction=False)
        nl.update(a)
        pos = a.get_positions(); cell = a.cell.array
        # brute-force pairs within r_max
        for i in ia:
            for j in ib:
                if i == j:
                    continue
                # minimum image using offsets from NL
                indices, offs = nl.get_neighbors(i)
                if j in indices:
                    off = offs[list(indices).index(j)]
                else:
                    off = np.zeros(3)
                rij = pos[j] + off @ cell - pos[i]
                d = np.linalg.norm(rij)
                if d < r_max:
                    k = int(np.floor(d / (r_max / bins)))
                    hist[min(k, bins - 1)] += 1.0
    centers = 0.5 * (edges[1:] + edges[:-1])
    # normalize by total counts to compare shapes
    norm = hist / (hist.sum() + 1e-12)
    return centers, norm

# ---------- DISTRIBUTION COMPARISONS ----------

def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    1D Wasserstein distance between two samples.
    """
    x = np.sort(a[~np.isnan(a)])
    y = np.sort(b[~np.isnan(b)])
    if x.size == 0 or y.size == 0:
        return np.nan
    # simple quantile coupling
    q = np.linspace(0, 1, num=min(x.size, y.size), endpoint=True)
    xq = np.quantile(x, q); yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))

def js_divergence_hist(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    """
    Jensen–Shannon divergence between two 1D histograms (base-2).
    """
    lo = np.nanmin([np.nanmin(a), np.nanmin(b)])
    hi = np.nanmax([np.nanmax(a), np.nanmax(b)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan
    edges = np.linspace(lo, hi, bins + 1)
    Ha, _ = np.histogram(a[~np.isnan(a)], bins=edges, density=True)
    Hb, _ = np.histogram(b[~np.isnan(b)], bins=edges, density=True)
    Ha = Ha / (Ha.sum() + 1e-12); Hb = Hb / (Hb.sum() + 1e-12)
    M = 0.5 * (Ha + Hb)
    def H(p):
        q = p[p > 0]
        return -np.sum(q * (np.log2(q)))
    return float(0.5 * (H(Ha) + H(Hb)) - H(M))
