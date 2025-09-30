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

# Helper to work on a non-periodic-z copy
def _npz(a):
    b = a.copy()
    b.set_pbc((True, True, False))
    return b

def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Least-squares plane fit n·x + c = 0 on PBC-consistent points.
    Returns (unit normal n, offset c).
    """
    X = np.asarray(points, float)
    ctr = X.mean(0)
    _, _, Vt = np.linalg.svd(X - ctr, full_matrices=False)
    n = Vt[-1]
    n /= (np.linalg.norm(n) + 1e-15)
    c = -np.dot(n, ctr)
    return n, c

def _carbon_positions(atoms: Atoms) -> np.ndarray:
    #pos = atoms.get_positions(wrap=True)
    pos = _npz(atoms).get_positions(wrap=True)  # wraps only x,y because pbc[2]=False
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    Cpos = pos[sym == "C"]
    if Cpos.size == 0:
        raise ValueError("No carbon atoms found.")
    return Cpos

def _plane_from_carbons(
    atoms: Atoms,
    mode: str = "mid",
    use_bottom_fraction: float = 0.5,
    cutoff: float = 1.8
) -> Tuple[np.ndarray, float]:
    """
    Build plane from carbons.
      mode='mid'    : fit to central fraction along z (robust mid-slab plane)
      mode='top'    : fit to carbons within 'cutoff' Å of z_max (top layer)
      mode='bottom' : fit to carbons within 'cutoff' Å of z_min (bottom layer)
    """
    Cpos = _carbon_positions(atoms)
    z = Cpos[:, 2]
    if mode == "mid":
        zmin, zmax = np.min(z), np.max(z)
        z0 = zmin + (1 - use_bottom_fraction) * (zmax - zmin) / 2
        z1 = zmax - (1 - use_bottom_fraction) * (zmax - zmin) / 2
        slab = Cpos[(z >= z0) & (z <= z1)]
        if slab.shape[0] < 3:
            slab = Cpos
        n, c = _fit_plane(slab)
        #print(f"Fitted mid-plane to {slab.shape[0]} carbons (z in [{z0:.2f}, {z1:.2f}])")
    elif mode == "top":
        zmax = np.max(z)
        top = Cpos[z >= zmax - cutoff]
        if top.shape[0] < 3:
            raise ValueError("Not enough carbons to fit top-layer plane.")
        n, c = _fit_plane(top)
        #print(f"Fitted top-plane to {top.shape[0]} carbons (z >= {zmax - cutoff:.2f})")
    elif mode == "bottom":
        zmin = np.min(z)
        bot = Cpos[z <= zmin + cutoff]
        if bot.shape[0] < 3:
            raise ValueError("Not enough carbons to fit bottom-layer plane.")
        n, c = _fit_plane(bot)
        #print(f"Fitted bottom-plane to {bot.shape[0]} carbons (z <= {zmin + cutoff:.2f})")
    else:
        raise ValueError("mode must be 'mid', 'top', or 'bottom'.")

    if n[2] < 0:  # keep upward orientation
        n = -n
    return n, c

def graphite_normal(atoms: Atoms, use_bottom_fraction: float = 0.5) -> np.ndarray:
    """
    Mid-slab normal for backward compatibility.
    """
    n, _ = _plane_from_carbons(atoms, mode="mid", use_bottom_fraction=use_bottom_fraction)
    return n
    """
    Estimate graphite surface normal under PBC.
    - Wrap positions to one image.
    - Select central fraction along z.
    - Fit plane to carbons, return unit normal pointing to +z.
    """
    pos = atoms.get_positions(wrap=True)
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    Cpos = pos[sym == "C"]
    if Cpos.size == 0:
        raise ValueError("No carbon atoms found.")

    z = Cpos[:, 2]
    zmin, zmax = np.min(z), np.max(z)
    z0 = zmin + (1 - use_bottom_fraction) * (zmax - zmin) / 2
    z1 = zmax - (1 - use_bottom_fraction) * (zmax - zmin) / 2
    slab = Cpos[(z >= z0) & (z <= z1)]
    if slab.shape[0] < 3:
        slab = Cpos

    n, _ = _fit_plane(slab)
    if n[2] < 0:
        n = -n
    return n

def _atom_indices(atoms: Atoms, symbol: str) -> np.ndarray:
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    return np.where(sym == symbol)[0]

def no_bond_length(atoms: Atoms) -> Optional[float]:
    """
    Shortest N–O distance (Å) with minimum-image convention.
    """
    a = _npz(atoms)
    N = _atom_indices(a, "N"); O = _atom_indices(a, "O")
    if len(N) == 0 or len(O) == 0:
        return None
    dmin = None
    for i in N:
        for j in O:
            d = atoms.get_distance(i, j, mic=True)
            dmin = d if dmin is None or d < dmin else dmin
    return float(dmin)

def no_axis(atoms: Atoms) -> Optional[np.ndarray]:
    """
    Unit vector along closest N→O under PBC.
    """
    a = _npz(atoms)
    N = _atom_indices(a, "N"); O = _atom_indices(a, "O")
    if len(N) == 0 or len(O) == 0:
        return None
    best = None; vbest = None
    for i in N:
        for j in O:
            v = atoms.get_distance(i, j, vector=True, mic=True)
            d = np.linalg.norm(v)
            if d == 0:
                continue
            if best is None or d < best:
                best, vbest = d, v / d
    return vbest

def adsorption_heights(
    atoms: Atoms,
    plane: str = "mid",                 # 'mid' | 'top' | 'bottom'
    use_bottom_fraction: float = 0.5,   # for plane='mid'
    cutoff: float = 1.8,                # Å window for top/bottom layer selection
    ref: str = "plane"                  # 'plane' -> fit plane, 'zcap' -> use z max/min of layer
) -> Dict[str, float]:
    """
    Heights of N, O, and NO COM relative to a graphite reference (Å), PBC-aware in x,y only.
      ref='plane' : fit plane from carbons (mid/top/bottom) and project distances along its normal
      ref='zcap'  : use scalar z reference from the selected layer
                    - top: z_ref = max z of carbon atoms within (z_max - cutoff, z_max]
                    - bottom: z_ref = min z of carbon atoms within [z_min, z_min + cutoff)
      plane: which layer defines the reference ('mid' only valid with ref='plane')
    Returns dict with keys 'h_N','h_O','h_COM' (NaN if missing atom).
    """
    a = _npz(atoms)                        # ensure z non-periodic
    pos = a.get_positions(wrap=True)       # wrap x,y only
    sym = np.array(a.get_chemical_symbols(), dtype=object)
    Cpos = pos[sym == "C"]
    if Cpos.size == 0:
        raise ValueError("No carbon atoms found.")

    if ref == "plane":
        n, c = _plane_from_carbons(a, mode=plane, use_bottom_fraction=use_bottom_fraction, cutoff=cutoff)
        def height(p):
            return float(np.dot(n, p) + c)

    elif ref == "zcap":
        z = Cpos[:, 2]
        if plane == "top":
            zmax = z.max()
            layer = z >= (zmax - cutoff)
            if layer.sum() < 1:
                raise ValueError("No carbons in top zcap window.")
            z_ref = z[layer].max()  # cap height at topmost carbons
            def height(p):
                return float(p[2] - z_ref)  # positive above top surface
        elif plane == "bottom":
            zmin = z.min()
            layer = z <= (zmin + cutoff)
            if layer.sum() < 1:
                raise ValueError("No carbons in bottom zcap window.")
            z_ref = z[layer].min()
            def height(p):
                return float(z_ref - p[2])  # positive above bottom surface
        else:
            raise ValueError("For ref='zcap', plane must be 'top' or 'bottom'.")
    else:
        raise ValueError("ref must be 'plane' or 'zcap'.")

    out = {}
    Ni = _atom_indices(a, "N"); Oi = _atom_indices(a, "O")
    out["h_N"] = height(pos[Ni[0]]) if len(Ni) else np.nan
    out["h_O"] = height(pos[Oi[0]]) if len(Oi) else np.nan
    if len(Ni) and len(Oi):
        com = 0.5 * (pos[Ni[0]] + pos[Oi[0]])
        out["h_COM"] = height(com)
    else:
        out["h_COM"] = np.nan
    return out

def tilt_angle_deg(
    atoms: Atoms,
    plane: str = "mid",
    use_bottom_fraction: float = 0.5,
    cutoff: float = 1.8
) -> Optional[float]:
    """
    Angle between NO axis and chosen graphite plane normal (deg), PBC-aware.
      plane: 'mid' | 'top' | 'bottom'
    0° upright, 90° parallel. None if NO missing.
    """
    a = _npz(atoms)
    v = no_axis(a)
    if v is None:
        return None
    n, _ = _plane_from_carbons(a, mode=plane, use_bottom_fraction=use_bottom_fraction, cutoff=cutoff)
    c = np.clip(np.dot(v, n), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def top_layer_plane(atoms, cutoff=1.5):
    """
    Fit a plane only to the top graphene layer.
    cutoff: Å above z_max to include carbons.
    """
    pos = atoms.get_positions(wrap=True)
    sym = np.array(atoms.get_chemical_symbols(), dtype=object)
    Cpos = pos[sym == "C"]

    zmax = np.max(Cpos[:, 2])
    top = Cpos[Cpos[:, 2] > zmax - cutoff]  # only atoms within cutoff of zmax
    if len(top) < 3:
        raise ValueError("Not enough carbons for top layer plane.")
    n, c = _fit_plane(top)
    if n[2] < 0:
        n = -n
    return n, c

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



