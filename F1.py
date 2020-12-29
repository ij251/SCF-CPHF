
import numpy as np
from pyscf import gto, scf
import numpy as np


def get_pi0(molecule):

    '''function to generate zeroth order Pi tensor'''

    spatial_j = molecule.intor('int2e')
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, spatial_j)
    k = np.einsum("ijkl->ilkj", j)
    pi0 = j - k

    return pi0


def get_hcore1(molecule, atom):

    '''function to generate first order core hamiltonian,
    requires which atom is being perturbed as an input'''

    mf = scf.RHF(molecule)
    g = grad.rhf.Gradients(mf)

    hcore1 = g.hcore_generator(molecule)(atom)

    return hcore1


def get_pi1(molecule):

    '''This function takes the 4 dimensional integral derivative tensor from
    PySCF and digests the integrals to construct the first order Pi tensor'''

    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)

    twoe = molecule.intor("int2e_ip1")
    twoe_x = molecule.intor("int2e_ip1")[0]
    twoe_y = molecule.intor("int2e_ip1")[1]
    twoe_z = molecule.intor("int2e_ip1")[2]
    print(twoe_z)

    j1z_spatial = np.zeros((len(twoe_z),len(twoe_z),len(twoe_z),len(twoe_z)))

    print(j1z_spatial)

    for i in range(len(twoe_z)):
        for j in range(len(twoe_z)):
            for k in range(len(twoe_z)):
                for l in range(len(twoe_z)):

                    j1z_spatial[i][j][k][l] += (twoe_z[i][j][k][l]\
                                        + twoe_z[j][i][k][l]\
                                        + twoe_z[k][l][i][j]\
                                        + twoe_z[l][k][i][j])

    j1z = np.kron(spin_j, j1z_spatial)
    k1z = np.einsum("ijkl->ilkj", j1z)

    pi1z = j1z - k1z

    return pi1z


def get_f1(pi1, p1, hcore1, pi0, p0):

    '''This function calculates the first order fock matrix from the first 
    order Pi tensor, density matrix and core hamiltonian, and the zeroth order
    density matrix and Pi tensor'''

    f1_1 = np.kron(omega, hcore1)
    f1_2 = np.einsum("ijkl,lk->ij", pi0, p1)
    f1_3 = np.einsum("ijkl,lk->ij", pi1, p0)

    f1 = f1_1 + f1_2 + f1_3

    return f1
