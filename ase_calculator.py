
import ase
from ase.build import bulk
import torch
from matris.applications.base import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model='matris_10m_oam', # matris_10m_oam, matris_10m_mp
    task='efsm', # Can be e/ef/efs/efsm 
    device=device # cpu or cuda
)

cu = bulk('Cu', a=5.43, cubic=True)
cu.calc = calc

energy = cu.get_potential_energy()   # total energy(eV)
forces = cu.get_forces()             # forces (eV/A)          
stress = cu.get_stress()             # stress (eV/A^3)  
magmoms = cu.get_magnetic_moments()  # magmom (muB)
print(energy)
print(forces)
print(stress)
print(magmoms)
