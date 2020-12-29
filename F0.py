import numpy as np
from pyscf import gto, scf

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

#h2_g0_ghf[:,[1,2]] = h2_g0_ghf[:,[2,1]]


def get_p0(g0, complexsymmetric: bool):

    '''Example function to generate zeroth order density matrix from
    coefficient matrix in either hermitian or complexsymmetric case'''

    if not complexsymmetric:
        p0 = np.matmul(g0, g0.T.conj())
    else:
        p0 = np.matmul(g0, g0.T)

    return p0


p0_h2 = get_p0(h2_g0_ghf, "False")
print(p0_h2)


def get_pi0(molecule):

    '''function to generate zeroth order Pi tensor'''

    spatial_j = molecule.intor('int2e')
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, spatial_j)
    k = np.einsum("ijkl->ilkj", j)
    pi0 = j - k

    return pi0


def get_hcore(molecule):

    '''function to generate zeroth order core hamiltonian'''

    hcore = molecule.intor('int1e_nuc')\
            + molecule.intor('int1e_kin')

    return hcore


def get_f0(hcore, pi0, p0):

    '''function to generate zeroth order Fock matrix'''

    omega = np.identity(2)
    f0_1e = np.kron(omega, hcore)
    f0_2e = np.einsum("ijkl,lk->ij", pi0, p0)
    f0 = f0_1e + f0_2e

    return f0


print(get_F0(get_Hcore(h2), get_pi0(h2), P0_h2))
