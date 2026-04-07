from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import json

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.io import read

from src.desc_comp_utils import fixed_mask


def load_structure_sets(paths: Sequence[str | Path]) -> List[List[Atoms]]:
    """
    Load each file in `paths` as a list of ASE Atoms objects.
    """
    out: List[List[Atoms]] = []
    for path in paths:
        frames = read(str(path), index=":")
        if isinstance(frames, Atoms):
            out.append([frames])
        else:
            out.append(list(frames))
    return out


def flatten_structure_sets(structure_sets: Sequence[Sequence[Atoms]]) -> List[Atoms]:
    return [atoms for struct_list in structure_sets for atoms in struct_list]


def unique_species(structure_sets: Sequence[Sequence[Atoms]]) -> List[str]:
    symbols = {
        symbol
        for struct_list in structure_sets
        for atoms in struct_list
        for symbol in atoms.get_chemical_symbols()
    }
    return sorted(symbols)


def build_filemap(paths: Sequence[str | Path]) -> Dict[int, str]:
    return {i: str(Path(path).expanduser().resolve()) for i, path in enumerate(paths)}


def build_provenance_table(
    structure_sets: Sequence[Sequence[Atoms]],
    paths: Sequence[str | Path],
    fixed_mask_fn: Callable[[Atoms], np.ndarray] = fixed_mask,
) -> pd.DataFrame:
    if len(structure_sets) != len(paths):
        raise ValueError(
            f"structure_sets and paths must have the same length: "
            f"{len(structure_sets)} != {len(paths)}"
        )
    rows: List[pd.DataFrame] = []
    for file_id, struct_list in enumerate(structure_sets):
        for struct_id, atoms in enumerate(struct_list):
            n_atoms = len(atoms)
            rows.append(
                pd.DataFrame(
                    {
                        "file_id": file_id,
                        "struct_id": struct_id,
                        "atom_id": np.arange(n_atoms, dtype=np.int32),
                        "symbol": atoms.get_chemical_symbols(),
                        "is_fixed": fixed_mask_fn(atoms),
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["file_id", "struct_id", "atom_id", "symbol", "is_fixed"])
    return pd.concat(rows, ignore_index=True)


def serialize_config(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def clear_directory_outputs(
    outdir: str | Path,
    patterns: Sequence[str],
) -> List[str]:
    """
    Remove previously generated files matching the provided glob patterns.

    Returns the list of removed file paths as strings.
    """
    outdir = Path(outdir).expanduser().resolve()
    removed: List[str] = []
    if not outdir.exists():
        return removed

    seen: set[Path] = set()
    for pattern in patterns:
        for path in outdir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            path.unlink()
            seen.add(path)
            removed.append(str(path))
    return removed


def save_descriptor_run(
    outdir: str | Path,
    descriptors: np.ndarray,
    provenance: pd.DataFrame,
    filemap: Dict[int, str],
    run_config,
    base_name: str,
    timestamp: str | None = None,
    clear_existing: bool = False,
) -> Dict[str, str]:
    """
    Save the descriptor matrix, provenance table, file map, and a config log.
    """
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_directory_outputs(
            outdir,
            patterns=(
                "*.npy",
                "*_provenance.parquet",
                "*_provenance.csv",
                "*_filemap.json",
                "*_params.txt",
                "*_config.json",
            ),
        )
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{base_name}_{timestamp}"

    npy_path = outdir / f"{prefix}.npy"
    parquet_path = outdir / f"{prefix}_provenance.parquet"
    csv_path = outdir / f"{prefix}_provenance.csv"
    filemap_path = outdir / f"{prefix}_filemap.json"
    log_path = outdir / f"{prefix}_params.txt"
    config_path = outdir / f"{prefix}_config.json"

    np.save(npy_path, descriptors)
    try:
        provenance.to_parquet(parquet_path, index=False)
        provenance_path = parquet_path
    except Exception:
        provenance.to_csv(csv_path, index=False)
        provenance_path = csv_path
    with filemap_path.open("w") as fh:
        json.dump(filemap, fh, indent=2)
    with log_path.open("w") as fh:
        fh.write(f"Run: {base_name}\n")
        fh.write(f"Timestamp: {timestamp}\n")
        fh.write(f"Descriptor rows: {descriptors.shape[0]}\n")
        fh.write(f"Descriptor cols: {descriptors.shape[1]}\n")
        fh.write("\nConfig:\n")
        fh.write(json.dumps(serialize_config(run_config), indent=2, sort_keys=True))
    with config_path.open("w") as fh:
        json.dump(serialize_config(run_config), fh, indent=2, sort_keys=True)

    return {
        "npy": str(npy_path),
        "provenance": str(provenance_path),
        "filemap": str(filemap_path),
        "log": str(log_path),
        "config": str(config_path),
    }
