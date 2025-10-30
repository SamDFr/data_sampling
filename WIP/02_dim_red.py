# %%
# Dimensionality Reduction
# ================================
# This section performs dimensionality reduction on the standardized SOAP descriptors using PCA, UMAP, or t-SNE.

# %%
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# %%
# --- Find the first .npy file in the current directory
path_to_results = "./desc"

npy_file = next(f for f in os.listdir(path_to_results) if f.endswith(".npy"))
all_soap_descriptors = np.load(os.path.join(path_to_results, npy_file))
print(f"Loaded SOAP descriptors from {npy_file}")
print(f"Descriptors shape: {all_soap_descriptors.shape}")

# --- Find the provenance parquet file
parquet_file = next(f for f in os.listdir(path_to_results) if f.endswith(".parquet"))
metadata_df = pd.read_parquet(os.path.join(path_to_results, parquet_file))
print(f"Loaded provenance metadata from {parquet_file}")
print(metadata_df.head())

# --- Find the filemap json
json_file = next(f for f in os.listdir(path_to_results) if f.endswith(".json"))
with open(os.path.join(path_to_results, json_file), "r") as f:
    filemap = json.load(f)
print(f"Loaded filemap from {json_file} to get vasprun.xml paths")


# %%
# --- Standardize the descriptors
scaler = StandardScaler()
standardized_descriptors = scaler.fit_transform(all_soap_descriptors)
print("Descriptors standardized.")
print(f"Standardized descriptors shape: {standardized_descriptors.shape}")

# %%
X = standardized_descriptors

# Choose: "pca", "umap", or "tsne"
method = "pca"          # change as needed
random_state = 42

# Optional: fast pre-reduction for UMAP/t-SNE on large SOAP
pca_prereduce_dim = None   # set None to skip

embedding = None
model = None

os.makedirs("embedding", exist_ok=True)

if method.lower() == "pca":
    from sklearn.decomposition import PCA
    #make a directory named PCA to save outputs
    
    n_components = 0.95         # embedding dimension
    model = PCA(n_components=n_components, random_state=random_state)
    embedding = model.fit_transform(X)
    save_pca = True
    if save_pca:
        with open("embedding/pca_model.json", "w") as f:
            json.dump({
                "components": model.components_.tolist(),
                "explained_variance": getattr(model, "explained_variance_", []).tolist() if hasattr(model, "explained_variance_") else [],
                "explained_variance_ratio": getattr(model, "explained_variance_ratio_", []).tolist() if hasattr(model, "explained_variance_ratio_") else [],
                "mean": getattr(model, "mean_", []).tolist() if hasattr(model, "mean_") else [],
                "n_components": int(getattr(model, "n_components_", getattr(model, "n_components", 0))),
                "n_features": int(getattr(model, "n_features_in_", X.shape[1])),
            }, f, indent=2) 
        print("Saved PCA model to embedding/pca_model.json")
    save_pca_matrix = True
    if save_pca_matrix:
        np.save("embedding/pca_embedding.npy", embedding)
        print("Saved PCA embedding matrix to PCA/pca_embedding.npy")

elif method.lower() == "umap":
    n_components = 10
    try:
        import umap.umap_ as umap
    except ImportError:
        raise RuntimeError("UMAP not installed. pip install umap-learn")

    X_in = X
    if pca_prereduce_dim:
        X_in = PCA(n_components=min(pca_prereduce_dim, X.shape[1]), random_state=random_state).fit_transform(X)

    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,        # tune per dataset size
        min_dist=0.0,
        metric="euclidean",
        random_state=random_state,
        verbose=True
    )
    embedding = model.fit_transform(X_in)

elif method.lower() == "tsne":
    n_components = 10 
    from sklearn.manifold import TSNE
    # t-SNE is O(N^2). Use PCA pre-step by default.
    X_in = X
    if pca_prereduce_dim:
        X_in = PCA(n_components=min(pca_prereduce_dim, X.shape[1]), random_state=random_state).fit_transform(X)

    model = TSNE(
        n_components=n_components,
        perplexity=30,         # 5–50 typical
        n_iter=1000,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        verbose=1,
        method="barnes_hut" if X_in.shape[0] < 50000 else "exact"
    )
    embedding = model.fit_transform(X_in)

else:
    raise ValueError("method must be 'pca', 'umap', or 'tsne'")

print(f"{method.upper()} embedding shape:", embedding.shape)

# %%
# require both columns
for col in ("symbol", "is_fixed"):
    if col not in metadata_df.columns:
        raise KeyError(f"Provenance parquet must contain '{col}' column.")

symbols = metadata_df["symbol"].astype(str)
states  = np.where(metadata_df["is_fixed"].to_numpy(), "fixed", "moving")
labels_combined = (symbols + "_" + states).astype("category")
cats = list(labels_combined.cat.categories)
codes = labels_combined.cat.codes.to_numpy()  # 0..K-1

plt.figure(figsize=(8, 6))
if embedding.shape[1] >= 2:
    # split labels: moving first, fixed last
    ordered_cats = [lab for lab in cats if lab.endswith("_moving")] + \
                   [lab for lab in cats if lab.endswith("_fixed")]

    for lab in ordered_cats:
        idx = (labels_combined == lab)
        if not np.any(idx):
            continue
        marker = "x" if lab.endswith("_fixed") else "o"
        plt.scatter(embedding[idx, 0], embedding[idx, 1],
                    s=1, alpha=0.1, marker=marker, label=lab)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"{method.upper()} Embedding of SOAP Descriptors")
    plt.grid(True, linewidth=0.2)
    plt.legend(markerscale=3, frameon=False, loc="best", title="symbol_state")
    plt.savefig(f"embedding/{method}_embedding.png", dpi=300)
    


