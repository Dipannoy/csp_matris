import os
import sys
import torch
import glob
from ase.io import read
from ase.build import bulk, molecule
from matris.applications.relax import StructOptimizer
from pymatgen.io.ase import AseAtomsAdaptor

# --- 1. Arguments from Command Line ---
# Usage: python batch_formation_energy.py <input_dir> <output_root> <top_pick>
if len(sys.argv) < 4:
    print("Usage: python batch_formation_energy.py <input_dir> <output_root> <top_pick>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]    # Root directory containing all subfolders of CIFs
OUTPUT_ROOT = sys.argv[2]  # Where the {formula} folders will be created
TOP_PICK = float(sys.argv[3])

MODEL_NAME = "matris_10m_oam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Initialize MatRIS ---
matris_opt = StructOptimizer(
    model=MODEL_NAME,
    task="efsm",
    optimizer="FIRE",
    device=DEVICE
)

def get_reference_energies(atoms_obj, optimizer):
    """Calculates reference energy per atom for unique elements."""
    refs = {}
    unique_elements = set(atoms_obj.get_chemical_symbols())
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

# --- 3. Iterate Subfolders ---
# os.walk visits every directory and subdirectory under INPUT_DIR
for root, dirs, files in os.walk(INPUT_DIR):
    # Find all CIFs in the current subfolder, excluding 'relaxed'
    cif_files = [os.path.join(root, f) for f in files if f.endswith('.cif') and 'relaxed' not in f]
    
    if not cif_files:
        continue # Skip folders with no CIFs

    folder_name = os.path.basename(root)
    print(f"\nProcessing subfolder: {folder_name} ({len(cif_files)} structures found)...")
    
    # Establish target metrics using the first CIF in the folder
    try:
        target_atoms = read(cif_files[0])
        target_formula = target_atoms.get_chemical_formula(mode='hill')
        target_count = len(target_atoms)
        element_refs = get_reference_energies(target_atoms, matris_opt)
    except Exception as e:
        print(f"  [Error] Failed to read first CIF in {folder_name} to set targets. Skipping folder. ({e})")
        continue

    results = []
    
    # Relax each structure in the folder
    for cif in cif_files:
        try:
            atoms = read(cif)
            
            # 2. Get data from the CIF
            actual_formula = atoms.get_chemical_formula(mode='hill')
            actual_count = len(atoms)
            
            # 3. Double-Layer Validation
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
            total_ref = sum(element_refs[s] for s in final_atoms.get_chemical_symbols())
            form_en_per_atom = (final_energy - total_ref) / len(final_atoms)
            
            results.append({
                'atoms': final_structure_pmg,
                'formation_energy': form_en_per_atom,
                'original_file': os.path.basename(cif)
            })
        except Exception as e:
            print(f"Failed to relax {cif}: {e}")

    # --- 4. Select Top % (Rank by Min Formation Energy) ---
    if not results:
        print(f"  No successful relaxations in {folder_name}.")
        continue
        
    # Sort by formation energy (ascending - most stable first)
    results.sort(key=lambda x: x['formation_energy'])
    
    # Take top %
    num_to_save = max(1, int(len(results) * TOP_PICK))
    top_results = results[:num_to_save]

    # Create output directory based on the actual formula processed
    out_dir = os.path.join(OUTPUT_ROOT, f"{target_formula}")
    os.makedirs(out_dir, exist_ok=True)

    # Save structures
    for rank, res in enumerate(top_results, 1):
        file_name = f"{target_formula}_{rank}.cif"
        save_path = os.path.join(out_dir, file_name)
        res['atoms'].to(filename=save_path, fmt="cif")
        print(f"Saved Rank {rank}: {file_name} (Ef: {res['formation_energy']:.4f} eV/at) - originally {res['original_file']}")

print("\nAll tasks completed.")
