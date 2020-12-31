import numpy as np
from pyscf import gto, scf, grad
from CPHFfunctions import get_p0, get_hcore0, get_pi0, get_f0, get_hcore1,
get_pi1, get_f1, get_p1


h2 = gto.M(atom=("H 0 0 0;"
                     "H 0 0 1;"),
                basis="sto-3g",
                unit="Angstrom")

h2_g0_rhf = np.array([
    [-0.5780280, 0.9965031],
    [-0.5780280, -0.9965031]
    ])

h2_g0_ghf = np.block([
    [h2_g0_rhf, np.zeros((2,2))],
    [np.zeros((2,2)), h2_g0_rhf],
    ])

h2_g0_ghf[:,[1,2]] = h2_g0_ghf[:,[2,1]]

