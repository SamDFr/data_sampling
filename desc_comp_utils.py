import numpy as np
from ase.constraints import FixAtoms

def fixed_mask(atoms):
    """
    Return a boolean mask indicating which atoms in an ASE Atoms object are fixed.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure that may contain constraints such as FixAtoms.

    Returns
    -------
    mask : ndarray of bool, shape (n_atoms,)
        Boolean mask of length equal to the number of atoms:
        - True  → atom is fixed (constrained, will not move).
        - False → atom is free to move.
    
    Notes
    -----
    - If no constraints are specified, all values are False (all atoms movable).
    - Handles both FixAtoms (with get_indices) and single-atom constraints exposing 'index'.
    - Can be used to add a provenance column ('is_fixed') aligned with SOAP descriptors.
    """
    
    n = len(atoms)
    mask = np.zeros(n, dtype=bool)
    cons = atoms.constraints
    if not cons:
        return mask  # no constraints -> all moving
    if not isinstance(cons, (list, tuple)):
        cons = [cons]
    for c in cons:
        if hasattr(c, "get_indices"):
            idx = np.asarray(c.get_indices(), dtype=int)
            if idx.size:
                mask[idx] = True
        elif hasattr(c, "index"):  # single atom constraint
            mask[int(c.index)] = True
    return mask
