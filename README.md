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

**What happens in practice:**  
`01_descriptors_computation.ipynb` performs three distinct selection/loading stages before any descriptor is computed:

1. **Discover candidate files** from the configured roots using ASE-readable filename patterns.
2. **Select files** from that discovered list:
   - all files
   - a reproducible random subset
   - every `N`th file in sorted order
3. **Select frames inside each chosen file** using `frame_stride`.

Only after these stages does the notebook load structures into ASE `Atoms` objects and compute descriptors.

**What one descriptor row means:**  
Each row of the final descriptor matrix corresponds to:
- one atom
- in one structure/frame
- from one selected source file

So the saved descriptor matrix is fully **atom-level**, while file/frame identity is stored separately in provenance.

**SOAP concept (per atom):**  
SOAP encodes the local atomic environment by expanding a neighbor density in a radial × angular basis. The output is a fixed‑length vector per atom.

**Optional force augmentation:**  
If enabled, the descriptor per atom becomes:
```
X_i = [ SOAP_i , F_i,x , F_i,y , F_i,z ]
```
- `SOAP_i` is the SOAP vector for atom `i`.
- `F_i,x`, `F_i,y`, and `F_i,z` are the selected force components. By default the notebook uses all three (`Fx, Fy, Fz`).

If force information is unavailable for the selected input structures, the notebook prints an explicit warning and falls back to SOAP-only descriptors.

**Provenance columns:**  
The provenance table is what allows later notebooks to map atom-level descriptors back to physical structures:

- `file_id`: integer id of the selected source file
- `struct_id`: structure/frame index **within the loaded subset**
- `atom_id`: atom index inside that structure
- `symbol`: chemical symbol of the atom
- `is_fixed`: whether the atom is constrained/fixed
- `source_struct_id` (optional): original frame index in the source file when `frame_stride > 1`

This distinction matters:
- `struct_id` is the index after frame subsampling
- `source_struct_id` keeps the original frame number so downstream exports and test-set generation can still reload the correct source frame

**Outputs (`desc/`):**
- `*.npy` descriptor matrix, shape `(N_atoms, D)`.
- `*_provenance.parquet|csv` with `(file_id, struct_id, atom_id, symbol, is_fixed)` and, when frame subsampling is enabled, `source_struct_id` storing the original frame index within the source file.
- `*_filemap.json` mapping `file_id -> file_path`.
- `*_config.json` with `soap_dim`, `force_dim`, `include_forces`.

**Output directory hygiene:**  
Before saving a new descriptor run, the notebook can clear old descriptor artifacts from `desc/`. This avoids mixing a fresh descriptor matrix with stale provenance/filemap/config files from an older run, which would otherwise create downstream mismatches in notebooks `02` to `06`.

**What is recorded in `*_config.json`:**  
The saved run config is part of the workflow contract. It records, among other things:
- input roots and patterns
- file-selection mode and parameters
- frame subsampling settings
- SOAP parameters
- whether forces were actually included
- descriptor dimensionalities (`soap_dim`, `force_dim`)

Downstream notebooks use this metadata to interpret the descriptor matrix consistently.

### 2) `02_descriptors_dim_red.ipynb`
**Goal:** Standardize and reduce dimensionality of descriptors.

**What it loads:**  
- the descriptor matrix from `desc/*.npy`
- the provenance table from `desc/*_provenance.*`
- the file map from `desc/*_filemap.json`
- the run config from `desc/*_config.json`

**Standardization:**  
```
X_std = (X - mean(X)) / std(X)
```

The notebook standardizes the matrix before dimensionality reduction so that large-magnitude descriptor dimensions do not dominate Euclidean distances used later in sampling.

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

In the current tested workflow, the downstream sampler expects a reduced matrix from `embedding/`, and the default path is PCA.

### 3) `03_sampler.ipynb`
**Goal:** Select a representative subset of atomic environments.

**What is actually sampled:**  
Sampling is performed at the **atom/environment level**, not directly at the structure level.

1. Load the reduced descriptor matrix `X` from `embedding/*.npy` (or standardized SOAP directly).
2. Whiten `X` in the notebook before sampling:
   ```
   X_white = (X - mean(X)) / std(X)
   ```
3. Sample atom indices in this whitened space.
4. Lift the selected atom indices back to structures using the provenance table.

This is important for interpretation:
- the diversity criterion is applied to atomic environments
- exported structures are the union of structures that contain at least one selected atom
- the final number of structures is therefore an outcome of atom-level sampling, not a direct input target

**Sampling methods implemented in `src/sampler_utils.py`:**  
- `fps`: greedy farthest-point sampling in Euclidean space
- `random`: uniform random sampling
- `grid`: 2D stratified grid sampling
- `kpp`: k-means++ seeding as a diversity heuristic
- `kmedoids`: k-means on a subset, then nearest real points
- `density_fps`: FPS after density-based reweighting
- `hdbscan`: cluster representatives per HDBSCAN cluster
- `adaptive_kmedoids`: adaptive k-means/medoid-style representative selection

**How your default FPS workflow behaves:**  
The notebook defaults to:
- `method = "fps"`
- `auto_sampling = True`
- `auto_strategy = "seeded"`

This is not a one-shot “pick `k` points and stop” workflow. Instead:

1. Start from an initial atom sample of size `k_start`.
2. Measure coverage on the current sample.
3. Add a new batch of `batch` atoms from the **remaining unsampled pool**.
4. Recompute coverage on the union of all sampled atoms.
5. Repeat until the stopping condition is met.

This seeded-batch strategy is the main nonstandard part of the implementation:
- each iteration preserves previous selections
- each new batch is sampled only from atoms not already selected
- the random seed is incremented by batch iteration
- this gives an incremental coverage-building workflow without implementing fully incremental FPS distance bookkeeping across the full dataset

**Alternative auto strategy:**  
`auto_strategy = "resample"` does something different:
- it increases the target sample size `k`
- at each step it resamples from scratch with the larger `k`
- it does **not** preserve previous sampled atoms across iterations

So if you care about preserving progression while coverage improves, `seeded` is the meaningful default.

**Coverage definition used for stopping:**  
Stopping is based on **mean 1D PCA-bin coverage across components**, computed by `pc_coverage_bins_auto(...)`.

For each PCA component:
1. define bins over the full-data range
2. mark which bins are touched by the sampled subset
3. compute:
   ```
   coverage_j = (# sampled bins on PC_j) / (# total bins on PC_j)
   ```
4. average across PCs:
   ```
   mean_cov = mean_j(coverage_j)
   ```

The number of bins is chosen automatically per component, typically using a target bin width (`coverage_mode = "width"` in the notebook).

So in the sampling step, **coverage** means:
- not geometric coverage in the FPS sense
- not structure coverage
- not range overlap as used later in `04_sampling_analysis.ipynb`

It is specifically:
- the fraction of occupied 1D bins along each PCA axis that are hit by the sampled atom subset
- averaged over the retained PCA components

Interpretation:
- `mean_cov = 1.0` would mean the sampled atoms touch every bin on every retained PCA component
- a larger `mean_cov` means broader occupancy of the reduced descriptor space
- this is a coverage proxy for diversity in the PCA representation, not a guarantee of perfect physical representativeness

**Auto-stop options:**  
- Standard behavior: stop when `mean_cov >= target_mean_cov`.
- Optional plateau stop: once `mean_cov >= coverage_threshold_min`, stop early if the last `plateau_window` coverage checks vary by at most `delta_cov`.
- `stop_reason` is recorded internally as one of:
  - `target_reached`
  - `plateau`
  - `k_max`

**From sampled atoms to exported structures:**  
After atom sampling:

1. selected atom indices are saved to:
   - `selected/sampled_atom_indices_<method>.npy`
2. atom hits are aggregated by `(file_id, struct_id)`:
   - `n_atoms_hit` = number of sampled atoms belonging to that structure
3. the selected structures are saved to:
   - `selected/sampled_structures_<method>.csv`
4. full structures are reloaded from source files and exported as:
   - `selected/<METHOD>_selected.traj`
   - `selected/<METHOD>_selected.xyz`
   - `selected/<METHOD>_selected_manifest.csv`

The manifest is therefore a **structure-level export table**, while the `.npy` index file is the atom-level sampling result.

**Outputs (`selected/`):**
- `sampled_atom_indices_<method>.npy` with sampled atom row indices
- `sampled_structures_<method>.csv` with lifted structure hits and `n_atoms_hit`
- `*_selected_manifest.csv` with structure-level identifiers and export metadata such as `(file_id, file_path, struct_id, n_atoms_hit, n_atoms, formula, species_sequence, ...)`
- `*_selected.traj` and `*_selected.xyz`

### 4) `04_sampling_analysis.ipynb`
**Goal:** Compare full vs sampled distributions.

**What is compared:**  
The notebook does not compare descriptors directly. It compares physically interpretable quantities reconstructed from trajectories:
- per-structure energies
- per-atom force norms
- sampled-vs-full size ratio

It loads:
- the original universe of structures from the configured input roots
- one sampled trajectory from `selected/` (typically `FPS_selected.traj`)

**Metrics:**  
- Energy distribution (histogram, coverage)  
- Force‑norm distribution by species (histogram, coverage)  
- Sample size fraction

**Coverage (range overlap):**
```
coverage = length( [min_s, max_s] ∩ [min_f, max_f] ) / (max_f - min_f)
```

where:
- `[min_s, max_s]` is the sampled range
- `[min_f, max_f]` is the full-data range

This is a simple range-coverage diagnostic, not a distributional distance.

### 5) `05_validation_set_extraction_from_sampled.ipynb`
**Goal:** Split the sampled set into validation and training subsets.

**What it uses:**  
- one exported sampled trajectory from `selected/`
- random splitting at the **structure/frame level**

**What it writes:**  
- validation trajectory and XYZ under `selected/valset/`
- training trajectory and XYZ under `selected/trainset/`
- companion text files listing the selected frame ids

The split is applied to already sampled structures. It does **not** resample atoms or descriptors.

**Procedure:**  
- Randomly select `P%` of the sampled structures for validation.  
- Remaining `(100-P)%` becomes training.

### 6) `06_test_set_generation.ipynb`
**Goal:** Build a test set disjoint from sampled/train/val.

**How disjointness is enforced:**  
The notebook defines the universe of available structures from the descriptor provenance:
- one row per unique `(file_id, struct_id)` or `(file_id, source_struct_id)` when frame subsampling was used upstream

It then removes any structure that already appears in sampled manifests under `selected/`, and draws the test set from the remainder.

This means the test set is disjoint at the **structure/frame identity** level, not only at the atom level.

**Procedure:**  
- Use `desc/*_provenance*` and `desc/*_filemap*` to define the full universe.  
- Exclude any structure in `selected/*_selected_manifest.csv`.  
- Randomly select `n_test` from the remaining pool.

## Quick Start (Notebooks)

1. Prepare structures as `vasprun.xml`, `XDATCAR`, `.xyz`, or `.traj` files (or folders containing them).
2. Put them under `./data/` (or edit the input roots in the notebook).
3. Run the notebooks in order: `01` → `06`.

Outputs are written to:
- `desc/` (descriptors + provenance)
- `embedding/` (reduced embeddings)
- `selected/` (sampled structures + manifests)

`01_descriptors_computation.ipynb` now defaults to cleaning old files from `desc/` before saving a new descriptor run. This is intentional: the later notebooks expect one coherent descriptor/provenance/filemap/config set.

## Optional: Restrict Input Files and Frames

In `01_descriptors_computation.ipynb`, you can keep the default full dataset or reduce the input before descriptor computation:

- `file_selection_mode = "all"` keeps every discovered file.
- `file_selection_mode = "random"` with `random_file_count = N` selects `N` files at random, reproducibly with `selection_seed`.
- `file_selection_mode = "stride"` with `file_stride = N` keeps files `0, N, 2N, ...` in sorted discovery order. For example, `file_stride = 3` keeps one file and skips the next two.
- `frame_stride = N` keeps frames `0, N, 2N, ...` inside each selected file. For example, `frame_stride = 2` keeps every other structure.

The default behavior remains:

- `file_selection_mode = "all"`
- `frame_stride = 1`

## Optional: Append Forces to SOAP

In `01_descriptors_computation.ipynb`, set:

- `include_forces = True`
- `force_components_idx = (0, 1, 2)` by default for `Fx, Fy, Fz` (or choose another subset)

This appends the selected force components to each SOAP vector. The run config saved in `desc/*_config.json` records `soap_dim` and `force_dim`.

In `02_descriptors_dim_red.ipynb`, set:

- `use_forces_in_dimred = True` to reduce SOAP+forces
- `use_forces_in_dimred = False` to reduce SOAP only

If force information is not present in the selected input structures, the pipeline prints an explicit warning and falls back to SOAP‑only descriptors. This can happen with formats that do not store forces, such as plain `.xyz` files.

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
   - `sampled_atom_indices_<method>.npy`
   - `sampled_structures_<method>.csv`
   - `*_selected_manifest.csv`
   - `*_selected.traj`, `*_selected.xyz`
   - `valset/`, `trainset/`, `testset/`

The important distinction is:
- `desc/` is atom-level
- `embedding/` is atom-level
- `sampled_atom_indices_<method>.npy` is atom-level
- `sampled_structures_<method>.csv` and exported trajectories are structure-level

The sampled manifest also records structural identity fields used by downstream pipelines, including atom count and ordered species sequence, so exported structures are not silently reinterpreted with the wrong atom types.

---

## File Formats

The input discovery uses ASE and accepts common formats:
- `vasprun*.xml`
- `XDATCAR*`
- `*.xyz` and `*.extxyz`
- `*.traj`

You can extend the patterns in `src/workflow_config.py` or directly in the notebooks. The notebook can also subsample the discovered file list and the frames inside each file before computing descriptors.

### Tested Scope

- Only `vasprun.xml` has been tested end‑to‑end.
- Only **PCA** (dimensionality reduction) and **FPS** (sampling) have been tested.

---

## Key Outputs

- **Descriptor matrix**: `desc/*.npy`  
  Rows map to atom environments.  
- **Provenance table**: `desc/*_provenance.*`  
  Tracks `(file_id, struct_id, atom_id, symbol, is_fixed)`, and includes `source_struct_id` when frame subsampling is enabled.
- **File map**: `desc/*_filemap.json`  
  Maps `file_id → file_path` for reconstruction.
- **Selection manifest**: `selected/*_selected_manifest.csv`  
  The canonical list of sampled/exported structures.
- **Atom-level sampling indices**: `selected/sampled_atom_indices_<method>.npy`  
  Row indices into the descriptor/embedding matrix.
- **Lifted structure hits**: `selected/sampled_structures_<method>.csv`  
  Structure-level aggregation of sampled atoms before export.

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

## Citation

If you use this repository, please cite:

**Samuel Del Fré, Gilberto A. Alou Angulo, Maurice Monnerville, Alejandro Rivero Santamaría (2026)**  
*Data‑Driven Construction of Machine‑Learning‑Based Interatomic Potentials for Gas–Surface Scattering Dynamics: The Case of NO on Graphite*, The Journal of Physical Chemistry C
DOI: 10.1021/acs.jpcc.6c01815

### BibTeX

```bibtex
@article{delfreDataDrivenConstructionMachineLearningBased2026a,
  title = {Data-{{Driven Construction}} of {{Machine-Learning-Based Interatomic Potentials}} for {{Gas}}--{{Surface Scattering Dynamics}}: {{The Case}} of {{NO}} on {{Graphite}}},
  shorttitle = {Data-{{Driven Construction}} of {{Machine-Learning-Based Interatomic Potentials}} for {{Gas}}--{{Surface Scattering Dynamics}}},
  author = {Del Fr{\'e}, Samuel and Alou Angulo, Gilberto A. and Monnerville, Maurice and Rivero Santamar{\'i}a, Alejandro},
  year = 2026,
  month = may,
  journal = {The Journal of Physical Chemistry C},
  pages = {acs.jpcc.6c01815},
  issn = {1932-7447, 1932-7455},
  doi = {10.1021/acs.jpcc.6c01815},
  urldate = {2026-05-18},
  copyright = {https://doi.org/10.15223/policy-029},
  langid = {english},
}
```
