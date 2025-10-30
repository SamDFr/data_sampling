# sampling_analysis_utils.py
import numpy as np
from typing import List, Tuple, Dict, Iterable, Optional
from ase.io import read
from ase.atoms import Atoms
from ase.neighborlist import NeighborList
from math import acos
from typing import Union, Sequence
from ase.io import read
from ase.atoms import Atoms

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

# ---------- NO GEOMETRY ON GRAPHITE ----------

def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Least-squares plane fit n⋅x + c = 0.
    Returns (unit normal n, c).
    """
    X = np.asarray(points, float)
    ctr = X.mean(0)
    U, S, Vt = np.linalg.svd(X - ctr, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-15)
    c = -np.dot(n, ctr)
    return n, c

def graphite_normal(atoms: Atoms, use_bottom_fraction: float = 0.5) -> np.ndarray:
    """
    Estimate surface normal from carbons by plane fit.
    Uses the denser slab region: take carbons within the central fraction of z-range.
    Returns unit normal pointing roughly +z (flip for consistency).
    """
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    Cpos = pos[sym == "C"]
    z = Cpos[:, 2]
    zmin, zmax = np.min(z), np.max(z)
    z0, z1 = zmin + (1 - use_bottom_fraction) * (zmax - zmin) / 2, zmax - (1 - use_bottom_fraction) * (zmax - zmin) / 2
    slab = Cpos[(z >= z0) & (z <= z1)]
    if slab.shape[0] < 3:
        slab = Cpos
    n, _ = _fit_plane(slab)
    # make normal point to +z for consistency
    if n[2] < 0:
        n = -n
    return n

def _atom_indices(atoms: Atoms, symbol: str) -> np.ndarray:
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    return np.where(sym == symbol)[0]

def no_bond_length(atoms: Atoms) -> Optional[float]:
    """
    Return N–O distance (assumes a single NO molecule present).
    If multiple N/O, returns the shortest N–O distance.
    """
    N = _atom_indices(atoms, "N"); O = _atom_indices(atoms, "O")
    if len(N) == 0 or len(O) == 0:
        return None
    r = atoms.get_positions()
    dmin = None
    for i in N:
        for j in O:
            d = np.linalg.norm(r[j] - r[i])
            dmin = d if dmin is None or d < dmin else dmin
    return dmin

def no_axis(atoms: Atoms) -> Optional[np.ndarray]:
    """
    Unit vector along N→O for the closest N–O pair (see no_bond_length).
    """
    N = _atom_indices(atoms, "N"); O = _atom_indices(atoms, "O")
    if len(N) == 0 or len(O) == 0:
        return None
    r = atoms.get_positions()
    best = None; vbest = None
    for i in N:
        for j in O:
            v = r[j] - r[i]
            d = np.linalg.norm(v)
            if d == 0: 
                continue
            if best is None or d < best:
                best, vbest = d, v / d
    return vbest

def adsorption_heights(atoms: Atoms) -> Dict[str, float]:
    """
    Height of N, O, and NO center-of-mass above graphite plane.
    Returns {'h_N','h_O','h_COM'} in Å.
    """
    n = graphite_normal(atoms)  # unit
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    # plane constant using carbon plane
    Cpos = pos[sym == "C"]
    _, c = _fit_plane(Cpos)
    def height(p):  # signed distance
        return (np.dot(n, p) + c)
    out = {}
    Ni = _atom_indices(atoms, "N"); Oi = _atom_indices(atoms, "O")
    out["h_N"] = float(height(pos[Ni[0]])) if len(Ni) else np.nan
    out["h_O"] = float(height(pos[Oi[0]])) if len(Oi) else np.nan
    if len(Ni) and len(Oi):
        com = 0.5 * (pos[Ni[0]] + pos[Oi[0]])
        out["h_COM"] = float(height(com))
    else:
        out["h_COM"] = np.nan
    return out

def tilt_angle_deg(atoms: Atoms) -> Optional[float]:
    """
    Angle (degrees) between NO axis and surface normal (0° = upright).
    """
    v = no_axis(atoms)
    if v is None:
        return None
    n = graphite_normal(atoms)
    c = np.clip(np.dot(v, n), -1.0, 1.0)
    return float(np.degrees(acos(c)))

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

# ---------- HIGH-LEVEL COMPARISON ----------

def summarize_no_graphite(images: List[Atoms]) -> Dict[str, np.ndarray]:
    """
    Compute per-frame NO-specific descriptors:
      - energy (eV)
      - NO bond length (Å)
      - tilt angle (deg)
      - heights h_N, h_O, h_COM (Å)
      - frame RMS force (all, and on N/O only)
    Returns dict of arrays (length = n_frames).
    """
    E = energies(images)
    L, T, hN, hO, hC = [], [], [], [], []
    rms_all = frame_rms_force(images, None)
    rms_NO  = frame_rms_force(images, ["N", "O"])
    for a in images:
        bl = no_bond_length(a); L.append(np.nan if bl is None else bl)
        ta = tilt_angle_deg(a); T.append(np.nan if ta is None else ta)
        h   = adsorption_heights(a)
        hN.append(h["h_N"]); hO.append(h["h_O"]); hC.append(h["h_COM"])
    return {
        "energy": E,
        "no_bond": np.array(L, float),
        "tilt_deg": np.array(T, float),
        "h_N": np.array(hN, float),
        "h_O": np.array(hO, float),
        "h_COM": np.array(hC, float),
        "rmsF_all": rms_all,
        "rmsF_NO": rms_NO,
    }

def compare_distributions(full: np.ndarray, sample: np.ndarray) -> Dict[str, float]:
    """
    Compare 1D distributions with Wasserstein and JS divergence.
    """
    return {
        "wasserstein": wasserstein_1d(full, sample),
        "js_divergence": js_divergence_hist(full, sample),
        "mean_full": float(np.nanmean(full)),
        "mean_sample": float(np.nanmean(sample)),
        "std_full": float(np.nanstd(full)),
        "std_sample": float(np.nanstd(sample)),
    }

def analyze_sampling_vs_full(
    full_images: List[Atoms],
    sample_images: List[Atoms],
    rdf_pairs: Tuple[Tuple[str, str], ...] = (("C", "C"), ("C", "N"), ("C", "O"), ("N", "O")),
    r_max: float = 8.0,
    bins: int = 200,
) -> Dict[str, Dict]:
    """
    High-level comparison:
      - NO descriptors (bond, tilt, heights, energies, RMS forces)
      - RDFs for selected pairs
      - Distribution metrics between full and sample
    Returns dict of sections with metrics and series.
    """
    full = summarize_no_graphite(full_images)
    samp = summarize_no_graphite(sample_images)

    out: Dict[str, Dict] = {"series_full": full, "series_sample": samp, "metrics": {}}

    # scalar series comparisons
    keys = ["energy", "no_bond", "tilt_deg", "h_N", "h_O", "h_COM", "rmsF_all", "rmsF_NO"]
    for k in keys:
        out["metrics"][k] = compare_distributions(full[k], samp[k])

    # RDFs (shape-only normalization)
    rdf_sec = {}
    for pair in rdf_pairs:
        rF, gF = pair_rdf(full_images, pair, r_max=r_max, bins=bins)
        rS, gS = pair_rdf(sample_images, pair, r_max=r_max, bins=bins)
        rdf_sec[str(pair)] = {
            "r": rF, "full": gF, "sample": gS,
            "wasserstein": wasserstein_1d(np.repeat(rF, (gF * 1000).astype(int)),  # coarse proxy
                                          np.repeat(rS, (gS * 1000).astype(int)))
        }
    out["rdf"] = rdf_sec
    return out



