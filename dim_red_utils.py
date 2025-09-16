# dim_red_utils.py
import numpy as np
import pandas as pd

def select_by_pca_box(embedding, ranges):
    """
    Select rows of embedding within axis-aligned PCA ranges.
    
    Parameters
    ----------
    embedding : ndarray (n_samples, n_components)
        PCA/UMAP/t-SNE embedding.
    ranges : dict
        Mapping {component_index: (min, max)}.
        Example: {0: (-5, -2), 1: (1, 3)}.
    
    Returns
    -------
    mask : ndarray of bool
        True for rows inside the ranges.
    """
    mask = np.ones(len(embedding), dtype=bool)
    for comp, (lo, hi) in ranges.items():
        vals = embedding[:, comp]
        mask &= (vals >= lo) & (vals <= hi)
    return mask


def origins_from_mask(mask, metadata_df, filemap):
    """
    Map selected rows back to provenance information.
    
    Parameters
    ----------
    mask : ndarray of bool
    metadata_df : DataFrame
        Provenance table with file_id, struct_id, atom_id, symbol, is_fixed.
    filemap : dict
        Mapping {file_id: path}.
    
    Returns
    -------
    DataFrame with provenance for selected atoms.
    """
    sub = metadata_df.loc[mask, ["file_id","struct_id","atom_id","symbol","is_fixed"]].copy()
    # Ensure keys are int for mapping
    fmap = {int(k): v for k, v in filemap.items()}
    sub["file_path"] = sub["file_id"].map(fmap)
    return sub


def group_summary(df):
    """
    Summarize selected atoms per file and structure.
    
    Parameters
    ----------
    df : DataFrame
        Output of origins_from_mask().
    
    Returns
    -------
    (grp, tot)
    grp : DataFrame, counts per (file_path, struct_id, symbol, is_fixed).
    tot : DataFrame, total counts per (file_path, struct_id).
    """
    grp = (df
           .groupby(["file_path","struct_id","symbol","is_fixed"], as_index=False)
           .size()
           .rename(columns={"size":"n_atoms"}))
    tot = (df.groupby(["file_path","struct_id"], as_index=False)
             .size()
             .rename(columns={"size":"n_atoms_total"}))
    return grp, tot
