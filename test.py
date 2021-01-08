import numpy as np
from pyscf import gto, scf

mol = gto.M(
        atom = (
            f"Li 0 0 0;"
            f"F  0 0 1;"
        ),
        basis = 'sto3g',
        unit = 'Bohr')
mol.charge = 0
mol.spin = 0


print(mol.aoslice_by_atom())


