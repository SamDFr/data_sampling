# data_sampling

Descriptor-based workflow to build compact, diverse training/validation/test sets from atomistic datasets.

This repository is **system‑agnostic**: it works with any structure files supported by ASE (e.g. `vasprun.xml`, `XDATCAR`, `.xyz`, `.traj`), and produces descriptors, embeddings, and sampled subsets with full provenance.

---

## Repository Structure

```text
data_sampling/
├── 01_descriptors_computation.ipynb
├── 02_descriptors_dim_red.ipynb
├── 03_sampler.ipynb
├── 04_sampling_analysis.ipynb
├── 05_validation_set_extraction_from_sampled.ipynb
├── 06_test_set_generation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── desc_comp_utils.py
│   ├── workflow_config.py
│   ├── workflow_io.py
│   ├── dim_red_utils.py
│   ├── sampler_utils.py
│   ├── sampling_analysis_utils.py
│   └── plots_exports_utils.py
│
└── README.md
```

---

## Pipeline Overview (Detailed)

Below is the exact flow, with the data objects each notebook produces and consumes.

### 1) `01_descriptors_computation.ipynb`
**Goal:** Convert structures into per‑atom descriptor vectors and build provenance.

- **Input:** `vasprun.xml`, `XDATCAR`, `.xyz`, `.traj` (any ASE‑readable format).
- **Descriptor:** SOAP vectors for each atom.
- **Optional augmentation:** append force components.

**SOAP concept (per atom):**  
SOAP encodes the local atomic environment by expanding a neighbor density in a radial × angular basis. The output is a fixed‑length vector per atom.

**Optional force augmentation:**  
If enabled, the descriptor per atom becomes:
```
X_i = [ SOAP_i , F_i,α , F_i,β ]
```
- `SOAP_i` is the SOAP vector for atom `i`.
- `F_i,α` and `F_i,β` are selected force components (e.g. Fx,Fy).

**Outputs (`desc/`):**
- `*.npy` descriptor matrix, shape `(N_atoms, D)`.
- `*_provenance.parquet|csv` with `(file_id, struct_id, atom_id, symbol, is_fixed)`.
- `*_filemap.json` mapping `file_id -> file_path`.
- `*_config.json` with `soap_dim`, `force_dim`, `include_forces`.

**Output directory hygiene:**  
Before saving a new descriptor run, the notebook can clear old descriptor artifacts from `desc/`. This avoids mixing a fresh descriptor matrix with stale provenance/filemap/config files from an older run, which would otherwise create downstream mismatches in notebooks `02` to `06`.

### 2) `02_descriptors_dim_red.ipynb`
**Goal:** Standardize and reduce dimensionality of descriptors.

**Standardization:**  
```
X_std = (X - mean(X)) / std(X)
```

**Reduction options:**  
- **PCA** (linear): finds directions of maximal variance.  
- **UMAP** (nonlinear): preserves neighborhood structure.  
- **t‑SNE** (nonlinear): emphasizes local structure (expensive).

**Force toggle:**  
- `use_forces_in_dimred = True` uses the full `X` (SOAP + forces).  
- `use_forces_in_dimred = False` uses SOAP‑only (`X[:, :soap_dim]`).

**Outputs (`embedding/`):**
- `pca_embedding.npy`
- `pca_model.json`

### 3) `03_sampler.ipynb`
**Goal:** Select a representative subset of atomic environments.

**Sampling methods:**  
- FPS (Farthest Point Sampling)  
- Optional clustering or random sampling

**Outputs (`selected/`):**
- `*_selected_manifest.csv` with `(file_path, struct_id, atom_id, ... )`
- `*_selected.traj` and `*_selected.xyz`

### 4) `04_sampling_analysis.ipynb`
**Goal:** Compare full vs sampled distributions.

**Metrics:**  
- Energy distribution (histogram, coverage)  
- Force‑norm distribution by species (histogram, coverage)  
- Sample size fraction

**Coverage (range overlap):**
```
coverage = length( [min_s, max_s] ∩ [min_f, max_f] ) / (max_f - min_f)
```

### 5) `05_validation_set_extraction_from_sampled.ipynb`
**Goal:** Split the sampled set into validation and training subsets.

**Procedure:**  
- Randomly select `P%` of the sampled structures for validation.  
- Remaining `(100-P)%` becomes training.

### 6) `06_test_set_generation.ipynb`
**Goal:** Build a test set disjoint from sampled/train/val.

**Procedure:**  
- Use `desc/*_provenance*` and `desc/*_filemap*` to define the full universe.  
- Exclude any structure in `selected/*_selected_manifest.csv`.  
- Randomly select `n_test` from the remaining pool.

---

## Installation

```bash
git clone https://github.com/SamDFr/data_sampling.git
cd data_sampling

python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt
```

---

## Quick Start (Notebooks)

1. Prepare structures as `vasprun.xml`, `XDATCAR`, `.xyz`, or `.traj` files (or folders containing them).
2. Put them under `./data/` (or edit the input roots in the notebook).
3. Run the notebooks in order: `01` → `06`.

Outputs are written to:
- `desc/` (descriptors + provenance)
- `embedding/` (reduced embeddings)
- `selected/` (sampled structures + manifests)

`01_descriptors_computation.ipynb` now defaults to cleaning old files from `desc/` before saving a new descriptor run. This is intentional: the later notebooks expect one coherent descriptor/provenance/filemap/config set.

## Optional: Append Forces to SOAP

In `01_descriptors_computation.ipynb`, set:

- `include_forces = True`
- `force_components_idx = (0, 1)` (or another pair)

This appends the selected force components to each SOAP vector. The run config saved in `desc/*_config.json` records `soap_dim` and `force_dim`.

In `02_descriptors_dim_red.ipynb`, set:

- `use_forces_in_dimred = True` to reduce SOAP+forces
- `use_forces_in_dimred = False` to reduce SOAP only

If forces are not present in the input files, the pipeline prints a warning and falls back to SOAP‑only descriptors.

---

## Data Flow (What Goes Where)

1. **Input structures** → `./data/` (or any folder you set)
2. **Descriptors + provenance** → `desc/`
   - `*_provenance.parquet|csv`
   - `*_filemap.json`
   - `*.npy` (descriptor matrix)
   - `*_config.json` (run config)
3. **Embeddings** → `embedding/`
   - `pca_embedding.npy`, `pca_model.json`
4. **Samples** → `selected/`
   - `*_selected_manifest.csv`
   - `*_selected.traj`, `*_selected.xyz`
   - `valset/`, `trainset/`, `testset/`

The sampled manifest also records structural identity fields used by downstream pipelines, including atom count and ordered species sequence, so exported structures are not silently reinterpreted with the wrong atom types.

---

## File Formats

The input discovery uses ASE and accepts common formats:
- `vasprun*.xml`
- `XDATCAR*`
- `*.xyz` and `*.extxyz`
- `*.traj`

You can extend the patterns in `src/workflow_config.py` or directly in the notebooks.

### Tested Scope

- Only `vasprun.xml` has been tested end‑to‑end.
- Only **PCA** (dimensionality reduction) and **FPS** (sampling) have been tested.

---

## Key Outputs

- **Descriptor matrix**: `desc/*.npy`  
  Rows map to atom environments.  
- **Provenance table**: `desc/*_provenance.*`  
  Tracks `(file_id, struct_id, atom_id, symbol, is_fixed)`.
- **File map**: `desc/*_filemap.json`  
  Maps `file_id → file_path` for reconstruction.
- **Selection manifest**: `selected/*_selected_manifest.csv`  
  The canonical list of sampled structures.

---

## Portability Notes

- Keep system-specific paths in a small config block or config file.
- Prefer `src/workflow_config.py` and `src/workflow_io.py` for new systems.

---

## Common Issues

- **“No provenance table found in desc/”**  
  Run `01_descriptors_computation.ipynb` first.

- **Mismatched files in `desc/` across runs**  
  Re-run `01_descriptors_computation.ipynb`. The notebook now clears old descriptor artifacts in `desc/` before saving a fresh run so downstream notebooks read a consistent set of files.

- **Parquet read error**  
  Install `pyarrow` (`pip install pyarrow`) or rely on CSV fallback.

- **Empty test set after filtering**  
  Check `desc/*_filemap.json` and the selected manifests; missing paths are dropped.

---

## Versioning

- `main` is the legacy/stable workflow (kept for reference).
- `generalisation-vasprun` is the current generalised workflow (system‑agnostic, `src/` package).

Set the default branch on your Git host if you want new clones to land on the generalised workflow.

---

## Citation

If you use this repository, please cite:

**Samuel Del Fré, Gilberto A. Alou Angulo, Maurice Monnerville, Alejandro Rivero Santamaría (2026)**  
*Data‑Driven Construction of Machine‑Learning‑Based Interatomic Potentials for Gas–Surface Scattering Dynamics: The Case of NO on Graphite*  
DOI: 10.48550/arXiv.2603.18864

### BibTeX

```bibtex
@online{delfre2026mlip_gassurface,
  title = {Data-Driven Construction of Machine-Learning-Based Interatomic Potentials for Gas-Surface Scattering Dynamics: The Case of NO on Graphite},
  author = {Del Fr\'e, Samuel and Angulo, Gilberto A. Alou and Monnerville, Maurice and Santamar\'ia, Alejandro Rivero},
  year = {2026},
  eprint = {2603.18864},
  eprinttype = {arXiv},
  eprintclass = {physics},
  doi = {10.48550/arXiv.2603.18864}
}
```
