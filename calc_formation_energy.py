import os
import torch
from ase.io import read
from ase.build import bulk, molecule
from matris.applications.relax import StructOptimizer
from ase import Atoms
# --- 1. Configuration ---
cif_path = "./materium_structures/csp_benchmark_bertos_model_primitive_formula/materium_inference_Ba2BiSbO6/reduced_formula_Ba2BiSbO6_temp=1.00/material_sample_gen_0.cif"  # Change this to your filename
model_name = "matris_10m_oam"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Optimizer
matris_opt = StructOptimizer(
    model=model_name,
    task="efsm",
    optimizer="FIRE",
    device=device
)

# --- 2. Load Structure ---
atoms = read(cif_path)
chemical_formula = atoms.get_chemical_formula()
print(f"Loaded {chemical_formula} from {cif_path}")

# --- 3. Perform Relaxation ---
max_steps = 500
fmax = 0.05

opt_result = matris_opt.relax(
    atoms=atoms,
    verbose=True,
    steps=max_steps,
    fmax=fmax,
    relax_cell=True,
    ase_filter="FrechetCellFilter", # Ensure you patched relax.py as discussed!
)

# Get Final Data
trajectory = opt_result['trajectory']
final_energy = trajectory.energies[-1]  # Total Potential Energy (eV)
final_structure_pmg = opt_result['final_structure'] 

final_structure_pmg.to(filename='Ba2BiSbO6_pmg.cif', fmt="cif")

# Convert Pymatgen Structure back to ASE Atoms for our reference calculation
# from pymatgen.io.ase import AseAtomsAdaptor
# final_atoms = AseAtomsAdaptor.get_atoms(final_structure_pmg)
#final_atoms = opt_result['final_atoms'] # ASE Atoms object from result

# --- 4. Calculate Formation Energy ---
'''
def get_reference_energies(atoms_obj, optimizer):
    """Calculates energy per atom for each unique element in its bulk state."""
    refs = {}
    unique_elements = set(atoms_obj.get_chemical_symbols())
    
    for symbol in unique_elements:
        # Create a standard bulk reference for the element
        # Note: 'bulk' is a helper; for some elements you might need specific phases
        ref_structure = bulk(symbol) 
        
        # Calculate energy using the same model
        # We use the calculator directly from our optimizer to save memory
        ref_structure.calc = optimizer.calculator
        e_bulk = ref_structure.get_potential_energy()
        refs[symbol] = e_bulk / len(ref_structure)
        print(f"Reference energy for {symbol}: {refs[symbol]:.4f} eV/atom")
    return refs
'''
#from ase import Atoms
#from ase.build import bulk, molecule

def get_reference_energies(atoms_obj, optimizer):
    """Calculates energy per atom for each unique element."""
    refs = {}
    unique_elements = set(atoms_obj.get_chemical_symbols())
    
    # Standard states for elements that aren't 'bulk' crystals
    gas_molecules = {
        'H': 'H2',
        'N': 'N2',
        'O': 'O2',
        'F': 'F2',
        'Cl': 'Cl2'
    }

    for symbol in unique_elements:
        if symbol in gas_molecules:
            # Create the molecule (e.g., O2)
            ref_structure = molecule(gas_molecules[symbol])
            # Put it in a large box so it doesn't interact with itself
            ref_structure.set_cell([15, 15, 15])
            ref_structure.center()
            print(f"Using {gas_molecules[symbol]} molecule for {symbol} reference.")
        else:
            # Use standard bulk crystal (e.g., Cu, Fe, Si)
            ref_structure = bulk(symbol)
            print(f"Using bulk crystal for {symbol} reference.")
        
        # Calculate energy using the MatRIS calculator
        ref_structure.calc = optimizer.calculator
        e_total = ref_structure.get_potential_energy()
        
        # Divide by number of atoms (2 for O2, 1 or more for bulk)
        refs[symbol] = e_total / len(ref_structure)
        print(f"Reference energy for {symbol}: {refs[symbol]:.4f} eV/atom")
        
    return refs


# Get references for the elements in your CIF
element_refs = get_reference_energies(final_atoms, matris_opt)

# Calculate sum of reference energies for this specific composition
total_ref_energy = 0
for symbol in final_atoms.get_chemical_symbols():
    total_ref_energy += element_refs[symbol]

# Final Calculation
total_formation_energy = final_energy - total_ref_energy
formation_energy_per_atom = total_formation_energy / len(final_atoms)

# --- 5. Output Results ---
print('\n' + '='*30)
print(f"Structure: {chemical_formula}")
print(f"Total Potential Energy: {final_energy:.4f} eV")
print(f"Formation Energy:       {formation_energy_per_atom:.4f} eV/atom")
print('='*30)

# Optional: Save final structure
# final_atoms.write('relaxed_structure.cif')
