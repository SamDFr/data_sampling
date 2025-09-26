# %%
# Trajectory sampler
# ================================
# This section performs sampling of the original descriptors matrix or reduced ones from dim_red.ipynb to generate an initial dataset


# %%
import os, glob, json
import numpy as np
import pandas as pd
from sampler_utils import sample, atoms_to_structures, load_reduced, pc_coverage_bins, pc_coverage_bins_auto
from dim_red_utils import load_pca_model
import matplotlib.pyplot as plt
from plots_exports_utils import plot_sampling_pca, export_selected_structures, export_methods_bundle

# %%
# X can be the original standardized SOAP or a reduced matrix (e.g., PCA)
embedded_file = True 
if embedded_file:
    # Look for .npy file in embedding/ directory
    npy_files = glob.glob("embedding/*.npy")
    if not npy_files:
        raise FileNotFoundError("No .npy file found in embedding/ directory")
    if len(npy_files) > 1:
        raise RuntimeError(f"Multiple .npy files found: {npy_files}")
    npy_file = npy_files[0]
    print(f"Using embedding: {npy_file}")

    X = load_reduced(npy_file)  # or use np.load directly
    print("Loaded embedding with shape:", X.shape)

    pca_model_file = glob.glob("embedding/*.json") 
    if not pca_model_file:
        raise FileNotFoundError("No PCA model .json file found in embedding/ directory")
    if len(pca_model_file) > 1:
        raise RuntimeError(f"Multiple PCA model files found: {pca_model_file}")
    pca_model_file = pca_model_file[0]
    print(f"Using PCA model: {pca_model_file}")
    pca_model = load_pca_model(pca_model_file)
    
else: 
    # Look for .npy file in current directory
    npy_files = glob.glob("./*.npy")
    if not npy_files:
        raise FileNotFoundError("No .npy file found in current directory")
    if len(npy_files) > 1:
        raise RuntimeError(f"Multiple .npy files found: {npy_files}")
    npy_file = npy_files[0]
    print(f"Using standardized SOAP matrix: {npy_file}")

# Find provenance parquet file
parquet_files = glob.glob("desc/*.parquet")
if not parquet_files:
    raise FileNotFoundError("No provenance .parquet file found in current directory")
if len(parquet_files) > 1:
    raise RuntimeError(f"Multiple parquet files found: {parquet_files}")
parquet_file = parquet_files[0]
print(f"Using provenance file: {parquet_file}")
meta = pd.read_parquet(parquet_file)

# Match corresponding JSON file with same timestamp
json_file = glob.glob("desc/*.json")
if not json_file:
    raise FileNotFoundError(f"No filemap JSON found")
if len(json_file) > 1:
    raise RuntimeError(f"Multiple JSON matches found: {json_file}")
json_file = json_file[0]
print(f"Using filemap: {json_file}")
with open(json_file) as f:
    filemap = json.load(f)

# specify the directory where sampling results are saved
path_to_results = "./selected" # Change if you want to save results elsewhere
os.makedirs(path_to_results, exist_ok=True)

# %%
print("X dtype/shape:", X.dtype, X.shape)
print("has_nan:", np.isnan(X).any(), "has_inf:", np.isinf(X).any(),
      "abs max:", np.nanmax(np.abs(X)))

# 1) whiten (float64 for stability)
X = np.asarray(X, dtype=np.float64, order="C")
mu = X.mean(axis=0, keepdims=True)
sd = X.std(axis=0, keepdims=True) + 1e-12
Xw = (X - mu) / sd

print("Xw dtype/shape:", Xw.dtype, Xw.shape)
print("has_nan:", np.isnan(Xw).any(), "has_inf:", np.isinf(Xw).any(),
      "abs max:", np.nanmax(np.abs(Xw)))

FPS_sampling = True
random_sampling = True
birch_sampling = True
kpp_sampling = True

if FPS_sampling:
    print("\n=== FPS Sampling ===")
    method="fps"
    idx_atoms = sample(X, method=method, k=50000, progress=True)
    # lift to structures
    chosen_structs = atoms_to_structures(idx_atoms, meta, filemap, choose="all")#choose="top", top_k=200)
    # save
    np.save(os.path.join(path_to_results, f"./selected/sampled_atom_indices_{method}.npy"), idx_atoms)
    chosen_structs.to_csv(os.path.join(path_to_results, f"./selected/sampled_structures_{method}.csv"), index=False)
    per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
        X, idx_atoms, mode="width", target_width=0.1, min_bins=20, max_bins=10000
    )
    print("Mean coverage:", mean_cov)
    print("Per-PC bins used:", bins_used[:8])
    print("Per-PC coverage:", per_pc)
    plot_sampling_pca(X, idx_atoms, method="fps", per_pc=per_pc, npy_file=npy_file)

    save_files_fps = True
    if save_files_fps:
        paths = export_selected_structures(chosen_structs, method_name="FPS", outdir=path_to_results, extra_formats=("xyz"))
        print(paths)

if random_sampling:
    print("\n=== Random Sampling ===")
    method="random"
    idx_atoms = sample(X, method=method, k=50000, progress=True)

    # lift to structures
    chosen_structs = atoms_to_structures(idx_atoms, meta, filemap, choose="all")#choose="top", top_k=200)

    # save
    np.save(os.path.join(path_to_results, f"./selected/sampled_atom_indices_{method}.npy"), idx_atoms)
    chosen_structs.to_csv(os.path.join(path_to_results, f"./selected/sampled_structures_{method}.csv"), index=False)

    per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
        X, idx_atoms, mode="width", target_width=0.1, min_bins=20, max_bins=10000
    )
    print("Mean coverage:", mean_cov)
    print("Per-PC bins used:", bins_used[:8])
    print("Per-PC coverage:", per_pc)

    plot_sampling_pca(X, idx_atoms, method="fps", per_pc=per_pc, npy_file=npy_file)

    save_files_rdn = False
    if save_files_rdn:
        paths = export_selected_structures(chosen_structs, method_name="Random", outdir=path_to_results, extra_formats=("xyz"))
        #paths =export_selected_structures(chosen_structs, method_name="FPS", write_traj=False, extra_formats=("xyz",))
        print(paths)

if kpp_sampling:
    print("\n=== KPP Sampling ===")
    method="kpp"
    idx_atoms = sample(X, method=method, k=50000, progress=True)

        # lift to structures
    chosen_structs = atoms_to_structures(idx, meta, filemap, choose="all") #choose="top", top_k=200)
    

    # save
    np.save(os.path.join(path_to_results, f"sampled_atom_indices_{method}.npy"), idx)
    chosen_structs.to_csv(os.path.join(path_to_results, f"sampled_structures_{method}.csv"), index=False)

    per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
        X, idx, mode="width", target_width=0.1, min_bins=20, max_bins=10000
    )
    print("Mean coverage:", mean_cov)
    print("Per-PC bins used:", bins_used[:8])
    print("Per-PC coverage:", per_pc)

    plot_sampling_pca(X, idx, method="kpp", per_pc=per_pc, npy_file=npy_file)

    # One method
    save_files_kpp = False
    if save_files_kpp:
        paths = export_selected_structures(chosen_structs, method_name="KPP", outdir=path_to_results, extra_formats=("xyz"))
        print(paths)

if birch_sampling:
    print("\n=== BIRCH Sampling ===")
    from sampler_utils import direct_birch_sample, pc_bin_coverage

    first_two_PC = X[:, :3]
    ev_two_PC = pca_model.explained_variance_[:3]

    idx, info = direct_birch_sample(
        Z_pca=first_two_PC, ev=ev_two_PC,
        n_clusters=500,          # or None → CF subclusters
        threshold=0.5,
        branching_factor=30,
        k_per_cluster=1,
        progress=True
    )
    print(info)

    # lift to structures with your existing atoms_to_structures(...)
    chosen_structs = atoms_to_structures(idx, meta, filemap, choose="all")

    # save
    method="birch"
    np.save(os.path.join(path_to_results, f"./selected/sampled_atom_indices_{method}.npy"), idx)
    chosen_structs.to_csv(os.path.join(path_to_results, f"./selected/sampled_structures_{method}.csv"), index=False)

    per_pc, mean_cov, bins_used = pc_coverage_bins_auto(
        X, idx, mode="width", target_width=0.1, min_bins=20, max_bins=10000
    )
    print("Mean coverage:", mean_cov)
    print("Per-PC bins used:", bins_used[:8])
    print("Per-PC coverage:", per_pc)

    plot_sampling_pca(X, idx, method="birch", per_pc=per_pc, npy_file=npy_file)

    # One method
    save_files_birch = True
    if save_files_birch:
        paths = export_selected_structures(chosen_structs, method_name="Birch", outdir=path_to_results, extra_formats=("xyz"))
        print(paths)








