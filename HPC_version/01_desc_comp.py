# %%
# This code is a Python script.
# It is intended to be run in a Python 3 environment.

# Compute SOAP descriptors for a set of atomic structures
# Input files are in VASP 'vasprun.xml' format

# %%
# Import necessary libraries
from ase.io import read
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
import glob
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import json

# %%
# Define the path to the directory containing the VASP data files

#path_to_data = '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.1.Ts.100.NO.rand.zpe/'
#path_to_data = '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.3.Ts.100.NO.rand.zpe/'
#path_to_data = '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.1.Ts.300.NO.rand.zpe/'
path_to_data = ['/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.1.Ts.100.NO.rand.zpe/',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.3.Ts.100.NO.rand.zpe/',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.1.Ts.300.NO.rand.zpe/',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/vaspdata.Ei.0.3.Ts.300.NO.rand.zpe/',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/continue.vaspdata.Ei.0.1.Ts.100.NO.rand.zpe',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/continue.vaspdata.Ei.0.3.Ts.100.NO.rand.zpe',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/continue.vaspdata.Ei.0.1.Ts.300.NO.rand.zpe',
                '/Users/samuel/Desktop/postdoc_PhLAM/NO_HOPG/data/continue.vaspdata.Ei.0.3.Ts.300.NO.rand.zpe']

if not all(os.path.isdir(p) for p in path_to_data):
    raise NotADirectoryError(f"One or more paths in {path_to_data} are not valid directories.")
else:
    print(f"Directories {path_to_data} exist and are selected.")

# %%
# collect all vasprun-*.xml files across directories
vasp_files = []
for p in path_to_data:
    vasp_files.extend(glob.glob(os.path.join(p, "vasprun-*.xml")))

vasp_files = sorted(vasp_files)

if not vasp_files:
    raise FileNotFoundError(f"No vasprun.xml files found in {path_to_data}.")
else:
    print(f"Found {len(vasp_files)} vasprun files across {len(path_to_data)} directories.")

# %%
# Read all structures from the VASP files with ASE
structures = [read(f, index=':') for f in vasp_files]
print(f"Total number of structures read: {sum(len(s) for s in structures)}")

# Get unique atomic species in the structures
species = list(set(atom.symbol for struct_list in structures for atoms in struct_list for atom in atoms))
print(f"Species found in structures: {species}")

# %%
print(structures[0][0])  # Print the first structure for verification

# %%
# Convert the different species into a single one (e.g., 'C')
modify_species = False  # Set to False to keep original species

if modify_species == True:
    print("Modifying all species to a single type.")
    target_species = 'C'
    for struct_list in structures:
        for atoms in struct_list:
            for atom in atoms:
                if atom.symbol != target_species:
                    atom.symbol = target_species

    new_species = list(set(atom.symbol for struct_list in structures for atoms in struct_list for atom in atoms))
    print(f"Species after conversion: {new_species}")
else:
    print("No species modification applied.")
    new_species = species

# %%
# Define SOAP parameters
soap_params = {
    "species": new_species,
    "periodic": True,
    "r_cut": 6.0,
    "n_max": 4,
    "l_max": 4,
    "sigma": 1.0,
    "compression": {"mode": "mu2"} # Use "mu2" compression to reduce descriptor size
}

if "compression" in soap_params:
    print(f"Using compression mode: {soap_params['compression']['mode']}")

#"mu2" : The SOAP feature vector is generated in an element-agnostic way, 
# so that the size of the feature vector is now independent of the number 
# of elements (see Darby et al. below for details). It is still possible 
# when using this option to construct a feature vector that distinguishes 
# between elements by supplying element-specific weighting under “species_weighting”

#the paper: Darby, J.P., Kermode, J.R. & Csányi, G. Compressing local atomic neighbourhood 
# descriptors. npj Comput Mater 8, 166 (2022). https://doi.org/10.1038/s41524-022-00847-y

# Initialize SOAP descriptor
soap = SOAP(**soap_params)
print("SOAP descriptor initialized with parameters:", soap_params)

# --- compute + provenance
from desc_comp_utils import fixed_mask # function to identify fixed atoms

total_structs = sum(len(s) for s in structures)
desc_blocks = []
meta_rows = []

pbar = tqdm(total=total_structs, desc="Computing SOAP with provenance")
for file_idx, struct_list in enumerate(structures):
    for struct_idx, atoms in enumerate(struct_list):
        D = soap.create(atoms, n_jobs=1)              # shape: (n_atoms, dim)
        n = len(atoms)
        desc_blocks.append(D)


        # provenance rows aligned 1:1 with descriptor rows
        meta_rows.append(pd.DataFrame({
            "file_id": file_idx,
            "struct_id": struct_idx,
            "atom_id": np.arange(n, dtype=np.int32),
            "symbol": atoms.get_chemical_symbols(),
            "is_fixed":  fixed_mask(atoms),
        }))
        pbar.update(1)
pbar.close()

all_soap_descriptors = np.vstack(desc_blocks)
metadata_df = pd.concat(meta_rows, ignore_index=True)

# SOAP matrix shape
print(f"Total SOAP descriptors computed: {all_soap_descriptors.shape[0]}")
print(f"SOAP descriptor matrix shape: {all_soap_descriptors.shape}")

# Meta data shape
print(f"Metadata DataFrame shape: {metadata_df.shape}")
print("Metadata DataFrame columns:", metadata_df.columns.tolist())

# %%
# --- outputs
path_to_results = "desc" # Change if you want to save results elsewhere
os.makedirs(path_to_results, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
species_str = "-".join(new_species)
base = f"SOAP_{species_str}"
npy_file = os.path.join(path_to_results, f"{base}_{timestamp}.npy")
txt_file = os.path.join(path_to_results, f"{base}_params_{timestamp}.txt")
parquet_file = os.path.join(path_to_results, f"{base}_provenance_{timestamp}.parquet")
files_json = os.path.join(path_to_results, f"{base}_filemap_{timestamp}.json")

# save descriptors
np.save(npy_file, all_soap_descriptors)

# save provenance (compact and columnar)
metadata_df.to_parquet(parquet_file, engine="fastparquet", index=False)

# save file_id → path mapping
with open(files_json, "w") as f:
    json.dump({i: p for i, p in enumerate(vasp_files)}, f, indent=2)

# write run log
with open(txt_file, "w") as f:
    f.write("SOAP descriptor computation log\n")
    f.write("---------------------------------\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write(f"path_to_data = '{path_to_data}'\n")
    f.write(f"Number of vasprun.xml files: {len(vasp_files)}\n\n")
    for k, v in soap_params.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nTotal structures: {total_structs}\n")
    f.write(f"Total descriptors (rows): {all_soap_descriptors.shape[0]}\n")
    f.write(f"Descriptor dimension (cols): {all_soap_descriptors.shape[1]}\n")
    f.write(f"Data file: {os.path.basename(npy_file)}\n")
    f.write(f"Provenance table: {os.path.basename(parquet_file)}\n")
    f.write(f"File map: {os.path.basename(files_json)}\n")

# Example: resolving provenance later
# - load row i from descriptors, then metadata_df.iloc[i] gives (file_id, struct_id, atom_id, symbol)
# - map file_id via filemap JSON to the exact vasprun.xml path


