from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence
import json


def _as_path_list(paths: Sequence[str | Path]) -> List[Path]:
    return [Path(p).expanduser().resolve() for p in paths]


def discover_structure_files(
    roots: Sequence[str | Path],
    patterns: Sequence[str] | str | None = None,
    recursive: bool = True,
) -> List[Path]:
    """
    Discover structure files from one or many input roots.

    Parameters
    ----------
    roots:
        Directories or explicit file paths.
    patterns:
        Glob patterns used inside directories. If None, uses a generic set
        covering common VASP, XYZ, and ASE trajectory formats.
    recursive:
        If True, search recursively within directories.
    """
    if patterns is None:
        patterns = ["vasprun*.xml", "XDATCAR*", "*.xyz", "*.extxyz", "*.traj"]
    if isinstance(patterns, str):
        patterns = [patterns]
    files: List[Path] = []
    for root in _as_path_list(roots):
        if root.is_file():
            if any(root.match(pat) for pat in patterns):
                files.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(f"Input root does not exist: {root}")
        globber = root.rglob if recursive else root.glob
        for pat in patterns:
            files.extend([p.resolve() for p in globber(pat) if p.is_file()])

    unique = sorted({str(p): p for p in files}.values(), key=lambda p: str(p))
    if not unique:
        raise FileNotFoundError(f"No structure files found under: {roots}")
    return unique


def discover_vasprun_files(
    roots: Sequence[str | Path],
    pattern: str = "vasprun*.xml",
    recursive: bool = True,
) -> List[Path]:
    """
    Backward-compatible wrapper for vasprun.xml discovery.
    """
    return discover_structure_files(roots, patterns=[pattern], recursive=recursive)


def ensure_directory(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class InputConfig:
    roots: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=lambda: ["vasprun*.xml", "XDATCAR*", "*.xyz", "*.extxyz", "*.traj"])
    recursive: bool = True
    file_selection_mode: str = "all"
    random_file_count: int | None = None
    file_stride: int | None = None
    selection_seed: int = 0
    frame_stride: int = 1

    def discover(self) -> List[Path]:
        return discover_structure_files(self.roots, patterns=self.patterns, recursive=self.recursive)


@dataclass
class SOAPConfig:
    periodic: bool = True
    r_cut: float = 6.0
    n_max: int = 4
    l_max: int = 4
    sigma: float = 1.0
    compression_mode: str = "mu2"
    species: List[str] = field(default_factory=list)

    def to_kwargs(self) -> dict:
        kwargs = {
            "species": list(self.species),
            "periodic": self.periodic,
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
        }
        if self.compression_mode:
            kwargs["compression"] = {"mode": self.compression_mode}
        return kwargs


@dataclass
class WorkflowPaths:
    desc_dir: str = "desc"
    embedding_dir: str = "embedding"
    selected_dir: str = "selected"

    def resolve(self) -> dict:
        return {
            "desc_dir": ensure_directory(self.desc_dir),
            "embedding_dir": ensure_directory(self.embedding_dir),
            "selected_dir": ensure_directory(self.selected_dir),
        }


@dataclass
class WorkflowConfig:
    inputs: InputConfig = field(default_factory=InputConfig)
    soap: SOAPConfig = field(default_factory=SOAPConfig)
    paths: WorkflowPaths = field(default_factory=WorkflowPaths)
    name: str = "workflow"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowConfig":
        with Path(path).expanduser().open("r") as fh:
            data = json.load(fh)
        inputs_data = data.get("inputs", {})
        if "pattern" in inputs_data and "patterns" not in inputs_data:
            inputs_data = dict(inputs_data)
            inputs_data["patterns"] = [inputs_data.pop("pattern")]
        return cls(
            inputs=InputConfig(**inputs_data),
            soap=SOAPConfig(**data.get("soap", {})),
            paths=WorkflowPaths(**data.get("paths", {})),
            name=data.get("name", "workflow"),
        )
