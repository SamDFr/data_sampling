# data_sampling

Descriptor-based workflow to build compact, diverse training/validation/test sets from atomistic datasets.

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

## Workflow

1. Descriptor computation.
2. Dimensionality reduction.
3. Sampling.
4. Sampling analysis.
5. Validation set extraction.
6. Test set generation.

## Installation

```bash
git clone https://github.com/SamDFr/data_sampling.git
cd data_sampling

python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt

# Optional (for SOAP descriptors)
pip install dscribe
```

## Quick Start

1. Prepare structures as `vasprun.xml`, `XDATCAR`, or `.xyz` files (or folders containing them).
2. Edit the input roots in the descriptor notebook, or use `src/workflow_config.py` and `src/workflow_io.py` directly from Python.
3. Compute descriptors.
4. Run dimensionality reduction.
5. Run sampling.
6. Extract selected structures.
7. Launch DFT or MLIP training.

## Optional: Append Forces to SOAP

In `01_descriptors_computation.ipynb`, set:

- `include_forces = True`
- `force_components_idx = (0, 1)` (or another pair)

This appends the selected force components to each SOAP vector. The run config saved in `desc/*_config.json` records `soap_dim` and `force_dim`.

In `02_descriptors_dim_red.ipynb`, set:

- `use_forces_in_dimred = True` to reduce SOAP+forces
- `use_forces_in_dimred = False` to reduce SOAP only

## Portability Notes

- Keep system-specific paths in a small config block or config file.
- Prefer `src/workflow_config.py` and `src/workflow_io.py` for new systems.
