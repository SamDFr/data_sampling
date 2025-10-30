import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Iterable, Dict, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read, write
from ase.atoms import Atoms

### plot utils.py

def plot_sampling_pca(X, idx_atoms, method, per_pc=None, npy_file=None, max_bar=4):
    """
    Plot PCA coverage (bar chart) and PCA scatter with sampled points overlay.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        PCA embedding matrix. Must come from PCA (d >= 2).
    idx_atoms : ndarray, shape (k,)
        Indices of sampled atoms in X.
    method : str
        Sampling method name (e.g., "fps", "adaptive_kmedoids").
    per_pc : array-like or None
        Coverage per principal component (from e.g. pc_bin_coverage). Optional.
    npy_file : str or None
        File name of embedding. If given, checked for "pca" in name.
    max_bar : int
        Maximum number of PCs to show in coverage bar plot.

    Raises
    ------
    RuntimeError
        If X does not look like a PCA embedding (d < 2 or file name does not contain "pca").

    Returns
    -------
    None
        Plots are shown directly.
    """
    # Guard: must be PCA
    is_pca = (X.shape[1] >= 2)
    if npy_file is not None:
        is_pca = is_pca and ("pca" in os.path.basename(npy_file).lower())
    if not is_pca:
        raise RuntimeError("Plotting requires a 2D+ PCA embedding. Load a PCA .npy first.")

    # --- Coverage bar plot ---
    if per_pc is not None:
        m = min(max_bar, len(per_pc), X.shape[1])
        plt.figure(figsize=(7, 3))
        plt.bar(range(1, m + 1), per_pc[:m])
        plt.xlabel("PC")
        plt.ylabel("Coverage")
        plt.title("Bin coverage (Fig.2 metric)")
        plt.tight_layout()
        plt.savefig(f"./selected/{method}_pca_coverage.png", dpi=300)

    # --- PCA scatter with sampled points ---
    fig, ax = plt.subplots(figsize=(8, 6))
    N = X.shape[0]
    if N > 2_000_000:
        hb = ax.hexbin(X[:, 0], X[:, 1], gridsize=500, bins="log")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=1, alpha=0.05)

    ax.scatter(
        X[idx_atoms, 0], X[idx_atoms, 1],
        s=2, alpha=0.9, c="r", marker="o", label=f"sample ({len(idx_atoms)})"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA space with {len(idx_atoms)} sampled atoms ({method})")
    ax.grid(True, linewidth=0.2)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(f"./selected/{method}_pca_scatter.png", dpi=300)



## export_utils.py


def _load_struct(path: str, frame_idx: int) -> Atoms:
    """Read a single structure index from a vasprun.xml-like file."""
    return read(path, index=int(frame_idx))

#def _collect_images(chosen_structs: pd.DataFrame) -> List[Atoms]:
#    """
#    chosen_structs must have columns: ['file_path','struct_id'].
#    Returns list of Atoms in row order.
#   """
#    req_cols = {"file_path", "struct_id"}
#    missing = req_cols - set(chosen_structs.columns)
#   if missing:
#        raise KeyError(f"chosen_structs missing columns: {missing}")
#
#    imgs: List[Atoms] = []
#    for _, row in tqdm(chosen_structs.iterrows(), total=len(chosen_structs), desc="Loading structures"):
#        imgs.append(_load_struct(row["file_path"], int(row["struct_id"])))
#    return imgs

def _collect_images(chosen_structs: pd.DataFrame) -> List[Atoms]:
    """
    chosen_structs must have columns: ['file_path','struct_id'].
    Returns list of Atoms in row order.
    """
    req_cols = {"file_path", "struct_id"}
    missing = req_cols - set(chosen_structs.columns)
    if missing:
        raise KeyError(f"chosen_structs missing columns: {missing}")

    imgs: List[Atoms] = []
    for _, row in tqdm(chosen_structs.iterrows(), total=len(chosen_structs), desc="Loading structures"):
        imgs.append(_load_struct(row["file_path"], int(row["struct_id"])))
    return imgs

def _can_write_single_xdatcar(images: List[Atoms]) -> bool:
    """XDATCAR needs constant atom count and species order across frames."""
    if not images:
        return False
    n0 = len(images[0])
    syms0 = images[0].get_chemical_symbols()
    for a in images[1:]:
        if len(a) != n0 or a.get_chemical_symbols() != syms0:
            return False
    return True

def export_selected_structures(
    chosen_structs: pd.DataFrame,
    method_name: str,
    outdir: str = "selected",
    extra_formats: Tuple[str, ...] = ("xyz", "xdatcar"),
    manifest: bool = True,
    write_traj: bool = True,
) -> Dict[str, str]:
    """
    Build a merged trajectory for one method and export viewer formats.

    Parameters
    ----------
    chosen_structs : DataFrame
        Must contain at least ['file_path','struct_id'].
    method_name : str
        Used for filenames, e.g. 'FPS' -> 'FPS_selected.traj'.
    outdir : str
        Output directory.
    extra_formats : tuple of {'xyz','xdatcar'}
        Extra files to write in addition to .traj (if write_traj=True).
    manifest : bool
        Write a CSV manifest of frames used.
    write_traj : bool
        If True, write an ASE .traj file. If False, skip it.

    Returns
    -------
    paths : dict
        Keys may include 'traj', 'xyz', 'xdatcar', or per-file xdatcar keys.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    images = _collect_images(chosen_structs)

    out = {}

    # .traj
    if write_traj:
        traj_path = os.path.join(outdir, f"{method_name}_selected.traj")
        write(traj_path, images)
        out["traj"] = traj_path

    # optional manifest
    if manifest:
        mf = chosen_structs.copy()
        mf.to_csv(os.path.join(outdir, f"{method_name}_selected_manifest.csv"), index=False)

    # extras
    if "xyz" in extra_formats:
        xyz_path = os.path.join(outdir, f"{method_name}_selected.xyz")
        write(xyz_path, images)  # ASE auto-detects XYZ by extension
        out["xyz"] = xyz_path

    if "xdatcar" in extra_formats:
        if _can_write_single_xdatcar(images):
            xd_path = os.path.join(outdir, f"{method_name}_selected_XDATCAR")
            write(xd_path, images, format="vasp-xdatcar")
            out["xdatcar"] = xd_path
        else:
            out["xdatcar"] = "multiple"
            for fpath, group in chosen_structs.groupby("file_path", sort=False):
                imgs = [_load_struct(fpath, int(i)) for i in group["struct_id"].tolist()]
                if not _can_write_single_xdatcar(imgs):
                    continue
                stem = Path(fpath).stem.replace(".xml", "")
                xd_path = os.path.join(outdir, f"{method_name}_{stem}_XDATCAR")
                write(xd_path, imgs, format="vasp-xdatcar")
                out[f"xdatcar::{stem}"] = xd_path

    return out

def export_methods_bundle(
    selections: Dict[str, pd.DataFrame],
    outdir: str = "selected",
    extra_formats: Tuple[str, ...] = ("xyz", "xdatcar"),
) -> Dict[str, Dict[str, str]]:
    """
    Export multiple methods at once.
    selections: {'FPS': df_fps, 'BIRCH': df_birch, ...}
    Returns mapping method -> paths dict.
    """
    results = {}
    for name, df in selections.items():
        results[name] = export_selected_structures(df, name, outdir=outdir, extra_formats=extra_formats)
    return results