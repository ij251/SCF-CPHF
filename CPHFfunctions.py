from pyscf import gto, scf
import numpy as np


def get_p0(g0, complexsymmetric: bool):

    '''Example function to generate zeroth order density matrix from
    coefficient matrix in either hermitian or complexsymmetric case'''

    if not complexsymmetric:
        p0 = np.matmul(g0, g0.T.conj())
    else:
        p0 = np.matmul(g0, g0.T)

    return p0


def get_hcore0(molecule):

    '''function to generate zeroth order core hamiltonian'''

    hcore = molecule.intor('int1e_nuc')\
            + molecule.intor('int1e_kin')

    return hcore0


def get_pi0(molecule):

    '''function to generate zeroth order Pi tensor'''

    spatial_j = molecule.intor('int2e')
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, spatial_j)
    k = np.einsum("ijkl->ilkj", j)
    pi0 = j - k

    return pi0


def get_f0(hcore0, pi0, p0):

    '''function to generate zeroth order Fock matrix'''

    omega = np.identity(2)
    f0_1e = np.kron(omega, hcore0)
    f0_2e = np.einsum("ijkl,lk->ij", pi0, p0)
    f0 = f0_1e + f0_2e

    return f0


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

    j1z_spatial = np.zeros((len(twoe_z),len(twoe_z),len(twoe_z),len(twoe_z)))

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


def get_p1(eta0, g0, f1, nelec, complexsymmetric: bool):

    'G0 is a GHF coefficient matrix of molecular orbital coefficients in terms
    of basis functions of dimension 2*N_spatial, allowing for mixing of alpha
    and beta spin functions. ie G0 = np.array([N_basis, N_basis]), so G0[i,j] 
    is the coefficient of the ith basis function in the jth MO. It has been 
    ordered in occupied and virtual blocks'

    'eta0 is a vector of orbital energy eigenvalues
    F1 is the first order Fock matrix that depends on P1
    (the first order density matrix) and hence Y'

    nbasis = len(g0)
    nocc = nelec
    nvir = nbasis - nelec

    y = np.zeros((nbasis,nbasis))

    for i in range(nocc):
        for j in range(nvir):

            if not complexsymmetric:

                y += 1/(eta0[i] - eta0[nocc+j]) *
                np.linalg.multi_dot([
                    np.outer(g0[:,i], g0.T.conj()[:,i]),
                    F1,
                    np.outer(g0[:,nocc+j], g0.T.conj()[:,nocc+j])
                    ])

            else:

                y += 1/(eta0[i] - eta0[nocc+j]) *
                np.linalg.multi_dot([
                    np.outer(g0[:,i], g0.T[:,i]),
                    f1,
                    np.outer(g0[:,nocc+j], g0.T()[:,nocc+j])
                    ])

    if not complexsymmetric:
        p1 = y + y.T.conj()
    else:
        p1 = y + y.T

    return p1
