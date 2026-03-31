# data_sampling

**A descriptor-based workflow for data-efficient sampling of atomistic datasets** for Density Functional Theory (DFT) and Machine Learning Interatomic Potentials (MLIPs).

This repository provides a **modular pipeline** to construct **representative, diverse, and physically meaningful subsets** from large atomistic datasets using descriptor-space analysis.

---

## 🚀 Overview

Modern atomistic simulations (DFT, AIMD, MLIP-driven MD) generate massive datasets that are often **redundant and computationally expensive** to process. This repository implements a **descriptor-driven sampling strategy** to:

- Reduce redundancy in large datasets.
- Maximize coverage of configurational space.
- Construct optimal training/validation/test sets.
- Enable efficient **DFT data generation and MLIP training**.

---

## 🔬 Scientific Context

The methodology is designed for:

- **Machine Learning Interatomic Potentials** (DeepMD, MACE, AENET, etc.)
- **Active learning workflows**
- **Gas–surface interaction datasets**
- **Large-scale MD simulations** (10⁴–10⁶ environments)
- **Descriptor-space analysis** (SOAP, etc.)

**Philosophy**:

> *Select the most informative atomic environments rather than increasing dataset size blindly.*

---

## 📂 Repository Structure

```text
data_sampling/
├── 01_descriptors_computation.ipynb
├── 02_descriptors_dim_red.ipynb
├── 03_sampler.ipynb
├── 04_sampling_analysis.ipynb
├── 05_validation_set_extraction_from_sampled.ipynb
├── 06_test_set_generation.ipynb
│
├── desc_comp_utils.py
├── dim_red_utils.py
├── sampler_utils.py
├── sampling_analysis_utils.py
│
└── README.md
```

---

## ⚙️ Workflow

The pipeline follows a **6-step workflow**:

1. **Descriptor Computation**
  - Compute atomic descriptors (SOAP, etc.) for all environments.
  - Output: Descriptor matrix + metadata.
2. **Dimensionality Reduction**
  - Reduce descriptor dimensionality (PCA, UMAP).
  - Purpose: Accelerate sampling, improve coverage analysis.
3. **Sampling**
  - Select representative environments using:
    - Diversity-based methods (FPS, Density-aware FPS).
    - Clustering-based methods (K-means++, HDBSCAN).
    - Statistical methods (Random, Stratified sampling).
4. **Sampling Analysis**
  - Evaluate representativeness (coverage metrics, descriptor distribution).
5. **Validation Set Extraction**
  - Build independent validation datasets.
6. **Test Set Generation**
  - Construct unbiased test sets.

---

## 🧠 Key Features

- **Descriptor-aware sampling**: Selection in descriptor space, not real space.
- **Environment-level selection**: Map back to full configurations.
- **Provenance tracking**: Track structure index, atom index, element type, constraints.
- **Scalable**: Designed for 10⁵–10⁶ atomic environments.

---

## 📦 Installation

```bash
git clone https://github.com/SamDFr/data_sampling.git
cd data_sampling

# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install dependencies
pip install numpy scipy pandas scikit-learn matplotlib tqdm ase hdbscan jupyter

# Optional (for SOAP descriptors)
pip install dscribe
```

---

## ▶️ Quick Start

1. Prepare structures (ASE-compatible).
2. Compute descriptors.
3. Run dimensionality reduction.
4. Run sampling.
5. Extract selected structures.
6. Launch DFT or MLIP training.

---

## 📊 Typical Use Cases

- Build DFT training sets for MLIPs.
- Reduce large MD datasets.
- Perform active learning pre-selection.
- Analyze descriptor-space coverage.
- Generate diverse scattering configurations.

---

## 📖 Citation

If you use this repository, please cite:

**Samuel Del Fré, Gilberto A. Alou Angulo, Maurice Monnerville, Alejandro Rivero Santamaría (2026)**  
*Data-driven construction of machine-learning-based interatomic potentials for gas–surface scattering dynamics: The case of NO on graphite*  
[DOI: 10.48550/arXiv.2603.18864](https://doi.org/10.48550/arXiv.2603.18864)

### BibTeX

```bibtex
@online{delfre2026mlip_gassurface,
  title = {Data-Driven Construction of Machine-Learning-Based Interatomic Potentials for Gas-Surface Scattering Dynamics: The Case of NO on Graphite},
  author = {Del Fr\'e, Samuel and Angulo, Gilberto A. Alou and Monnerville, Maurice and Santamar\'ia, Alejandro Rivero},
  year = {2026},
  eprint = {2603.18864},
  eprinttype = {arXiv},
  eprintclass = {physics},
  doi = {10.48550/arXiv.2603.18864},
  url = {https://arxiv.org/abs/2603.18864}
}
```

---

## 📬 Contact

For questions or collaboration:

- Open a GitHub issue.
- Contact the repository owner.

---

## ⭐ Final Note

This repository is designed for **advanced users** in:

- Computational chemistry.
- Materials science.
- Atomistic simulations.
- Machine learning for physics.

It is particularly suited for users working with **large-scale atomistic datasets** and seeking **efficient, principled sampling strategies**.
