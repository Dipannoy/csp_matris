import os
import sys
import pandas as pd
import torch
import glob
from ase.io import read, write
from ase.build import bulk, molecule
from matris.applications.relax import StructOptimizer
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
# --- 1. Arguments from Slurm/Command Line ---
# Usage: python batch_formation_energy.py <algorithm_name> <csv_path> <output_root>
if len(sys.argv) < 5:
    print("Usage: python batch_formation_energy.py <algorithm> <csv_path> <output_root>")
    sys.exit(1)

ALGORITHM = sys.argv[1]
CSV_PATH = sys.argv[2]
OUTPUT_ROOT = sys.argv[3] # Where the {algo}_{formula} folders will be created
CIF_FOLDER = sys.argv[4]

MODEL_NAME = "matris_10m_oam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Initialize MatRIS ---
matris_opt = StructOptimizer(
    model=MODEL_NAME,
    task="efsm",
    optimizer="FIRE",
    device=DEVICE
)

# def get_reference_energies(atoms_obj, optimizer):
#     """Calculates reference energy per atom for unique elements."""
#     refs = {}
#     unique_elements = set(atoms_obj.get_chemical_symbols())
#     gas_molecules = {'H': 'H2', 'N': 'N2', 'O': 'O2', 'F': 'F2', 'Cl': 'Cl2'}

#     for symbol in unique_elements:
#         try:
#             if symbol in gas_molecules:
#                 ref_structure = molecule(gas_molecules[symbol])
#                 ref_structure.set_cell([15, 15, 15])
#                 ref_structure.center()
#             else:
#                 ref_structure = bulk(symbol)
            
#             ref_structure.calc = optimizer.calculator
#             e_total = ref_structure.get_potential_energy()
#             refs[symbol] = e_total / len(ref_structure)
#         except Exception as e:
#             print(f"Error calculating reference for {symbol}: {e}")
#             refs[symbol] = 0.0 # Fallback
#     return refs
    
def get_reference_energies(formula, optimizer):
    """Calculates reference energy per atom for unique elements."""
    refs = {}
    temp_atoms = Atoms(formula)
    unique_elements = set(temp_atoms.get_chemical_symbols())
    gas_molecules = {'H': 'H2', 'N': 'N2', 'O': 'O2', 'F': 'F2', 'Cl': 'Cl2'}

    for symbol in unique_elements:
        try:
            if symbol in gas_molecules:
                ref_structure = molecule(gas_molecules[symbol])
                ref_structure.set_cell([15, 15, 15])
                ref_structure.center()
            else:
                ref_structure = bulk(symbol)
            
            ref_structure.calc = optimizer.calculator
            e_total = ref_structure.get_potential_energy()
            refs[symbol] = e_total / len(ref_structure)
        except Exception as e:
            print(f"Error calculating reference for {symbol}: {e}")
            refs[symbol] = 0.0 # Fallback
    return refs

# --- 3. Process CSV ---
df = pd.read_csv(CSV_PATH)

for index, row in df.iterrows():
    # folder_path = row['folder'].split('/')[10]
    formula = row['formula']
    
    temp_target = Atoms(row['formula'])
    target_formula = temp_target.get_chemical_formula(mode='hill') # alphabetical, exact counts
    target_count = len(temp_target)
    
    folder_path = CIF_FOLDER + '/'+formula 
    
    # Path handling: The CSV folder path might be slightly different than your CIF locations
    # Adjust this glob pattern to match where your 50 CIFs actually are
    all_cifs = glob.glob(os.path.join(folder_path, "*.cif"))
    
    cif_files = [
        f for f in all_cifs 
        if 'relaxed' not in f
    ]
    
    if not cif_files:
        print(f"No CIF files found in {folder_path}, skipping...")
        continue

    results = []
    print(f"\nProcessing {formula} ({len(cif_files)} structures)...")
    
    element_refs = get_reference_energies(temp_target, matris_opt)

    # Relax each structure in the folder
    for cif in cif_files:
        try:
            atoms = read(cif)
            
            # 2. Get data from the CIF
            actual_formula = atoms.get_chemical_formula(mode='hill')
            actual_count = len(atoms)
            
            # 3. Double-Layer Validation
            # Check 1: Do the element counts match exactly?
            # Check 2: Does the total number of atoms match?
            if actual_formula != target_formula or actual_count != target_count:
                print(f"  [Skip] {os.path.basename(cif)}: Mismatch! "
                      f"(CIF: {actual_formula} [{actual_count} atoms] vs "
                      f"Target: {target_formula} [{target_count} atoms])")
                continue
            print(f"  [Valid] Processing {os.path.basename(cif)}...")
            opt_result = matris_opt.relax(
                atoms=atoms,
                verbose=False,
                steps=500,
                fmax=0.05,
                relax_cell=True,
                ase_filter="FrechetCellFilter"
            )
            
            # Extract energy and convert to ASE for ref calculation
            final_energy = opt_result['trajectory'].energies[-1]
            final_structure_pmg = opt_result['final_structure']
            final_atoms = AseAtomsAdaptor.get_atoms(final_structure_pmg)
            
            # Formation Energy Calculation
            # element_refs = get_reference_energies(final_atoms, matris_opt)
            total_ref = sum(element_refs[s] for s in final_atoms.get_chemical_symbols())
            form_en_per_atom = (final_energy - total_ref) / len(final_atoms)
            
            results.append({
                'atoms': final_structure_pmg,
                'formation_energy': form_en_per_atom
            })
        except Exception as e:
            print(f"Failed to relax {cif}: {e}")

    # --- 4. Select Top 10% (Rank by Min Formation Energy) ---
    if not results:
        continue
        
    # Sort by formation energy (ascending - most stable first)
    results.sort(key=lambda x: x['formation_energy'])
    
    # Take top 10% (e.g., 5 out of 50)
    num_to_save = max(1, int(len(results) * 0.10))
    top_results = results[:num_to_save]

    # Create output directory: {algorithm}_{formula}
    out_dir = os.path.join(OUTPUT_ROOT, f"{formula}")
    os.makedirs(out_dir, exist_ok=True)

    # Save structures
    for rank, res in enumerate(top_results, 1):
        file_name = f"{formula}_{rank}.cif"
        save_path = os.path.join(out_dir, file_name)
        res['atoms'].to(filename=save_path, fmt="cif")
        # write(save_path, res['atoms'])
        print(f"Saved Rank {rank}: {file_name} (Ef: {res['formation_energy']:.4f} eV/at)")

print("\nAll tasks completed.")

