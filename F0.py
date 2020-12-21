import numpy as np
from pyscf import gto, scf

h2 = gto.M(atom=("H 0 0 0;"
                     "H 0 0 1;"),
                basis="sto-3g",
                unit="Angstrom")

h2_G0_rhf = np.array([
    [-0.5780280, 0.9965031],
    [-0.5780280, -0.9965031]
    ])

h2_G0_ghf = np.block([
    [h2_G0_rhf, np.zeros((2,2))],
    [np.zeros((2,2)), h2_G0_rhf],
    ])

#h2_G0_ghf[:,[1,2]] = h2_G0_ghf[:,[2,1]]

def get_P0(G0, complexsymmetric: bool):
    if not complexsymmetric:
        P0 = np.matmul(G0, G0.T.conj())
    else:
        P0 = np.matmul(G0, G0.T)

    return P0    

P0_h2 = get_P0(h2_G0_ghf, "False")
print(P0_h2)

def get_F0(molecule, P0):
    
    spatial_j = molecule.intor('int2e')
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, spatial_j)
    k = np.einsum("ijkl->ilkj", j)
    pi0 = j - k
    print(pi0)

    hcore = molecule.intor('int1e_nuc')\
            + molecule.intor('int1e_kin')
    F0_1e = np.kron(omega, hcore)
    F0_2e = np.einsum("ijkl,lk->ij", pi0, P0)
    F0 = F0_1e+F0_2e

    return F0

    

print(get_F0(h2, P0_h2))


    


    
