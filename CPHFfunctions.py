from pyscf import gto, scf, grad
import numpy as np

def get_x(s_ao):

    r"""Calculates canonical basis orthogonalisation matrix x, defined by:

    .. math:: 

        \mathbf{X}=\mathbf{Us^{-\frac{1}{2}}}}

    where U is the matrix of eigenvectors of s_ao, and
    :math:'s^{-\frac{1}{2}}}' is the diagonal matrix of inverse square root
    eigenvalues of s_ao.

    :param s_ao: atomic orbital overlap matrix

    :returns: the orthogonalisation matrix x
    """

    nbasis = len(s_ao)

    s2 = np.zeros([nbasis,nbasis])

    for i in range(nbasis):

        s2[i,i] = (np.linalg.eig(s_ao)[0][i])**-0.5

    x = np.matmul(np.linalg.eig(s_ao)[1], s2)

    return x

def get_p0(g0, complexsymmetric: bool):

    r"""Calculates the zeroth order density matrix from the zeroth order
    coefficient matrix. It is defined by:

    .. math::

        \mathbf{P^{(0)}}=\mathbf{G^{(0)}G^{(0)\dagger\diamond}}

    :param g0: zeroth order GHF coefficient matrix, e.g from QChem.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The zeroth order density matrix.
    """

    if not complexsymmetric:
        p0 = np.matmul(g0, g0.T.conj())
    else:
        p0 = np.matmul(g0, g0.T)

    return p0


def get_hcore0(mol):

    r"""Calculates The zeroth order core hamiltonian.
    Each element is given by:

    .. math::

        \left(\mathbf{H_{core}^{(0)}}\right)_{\mu\nu}
        =\left(\phi_{\mu}\left|\mathbf{\hat{H}_{core}}\right|\phi_{\nu}\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order core hamiltonian matrix.
    """

    hcore0 = (mol.intor('int1e_nuc')
              + mol.intor('int1e_kin'))

    return hcore0


def get_pi0(mol):

    r"""Calculate the 4 dimensional zeroth order Pi tensor.
    Each element is given by:

    .. math::

        \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(0)}}
        = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
          \left(\mu'\mu|\nu'\nu\right)
        - \mathbf{\Omega_{\delta'\epsilon}\Omega_{\epsilon'\delta}}
          \left(\mu'\nu|\nu'\mu\right)

    :param mol: The class for a molecule as defined by PySCF.

    :returns: The zeroth order Pi tensor.
    """

    spatial_j = mol.intor('int2e')
    phys_spatial_j = np.einsum("abcd->acbd", spatial_j)
    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)
    j = np.kron(spin_j, phys_spatial_j)
    k = np.einsum("ijkl->ijlk", j)
    pi0 = j - k

    return pi0


def get_f0(hcore0, pi0, p0):

    r"""Calculates the zeroth order fock matrix, defined by:

    .. math::

        \mathbf{F^{(0)}}
        =\mathbf{H_{core}^{(0)}}
        +\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(0)}}

    :param hcore0: Zeroth order core hamiltonian matrix.
    :param pi0: Zeroth order 4 dimensional Pi tensor.
    :param p0: Zeroth order density matrix.

    :returns: The zeroth order fock matrix
    """

    omega = np.identity(2)
    f0_1e = np.kron(omega, hcore0)
    f0_2e = np.einsum("ijkl,lj->ik", pi0, p0)
    f0 = f0_1e + f0_2e

    return f0


def get_hcore1(mol, atom, coord):

    r"""Calculates the first order core hamiltonian matrix.
    Each element is given by:

    .. math::

        \left(\mathbf{H_{core}^{(1)}}\right)_{\mu\nu}
        = \left(\frac{\partial\phi_{\mu}}{\partial a}\left|
        \mathbf{\hat{H}_{core}}\right|\phi_{\nu}\right)
        + \left(\phi_{\mu}\left|\frac{\partial\mathbf{\hat{H}_{core}}}
        {\partial a}\right|\phi_{\nu}\right)
        + \left(\phi_{\mu}\left|\mathbf{\hat{H}_{core}}\right|
        \frac{\partial\phi_{\nu}}{\partial a}\right)

    (Note that :math:'a' is a particular specified pertubation, e.g movement
    in the x direction of atom 1)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order core hamiltonian matrix.
    """

    mf = scf.RHF(mol)
    g = grad.rhf.Gradients(mf)

    hcore1 = g.hcore_generator(mol)(atom)[coord]

    return hcore1


def get_pi1(mol, atom, coord):

    r"""Calculates the 4 dimensional first order pi tensor by digesting the
    of 2 electron integrals given by PySCF.
    Symmetry of the 2 electron integrals is manipulated to digest the PySCF
    tensor, in which the first MO of each 2 electron integral has been
    differentiated.
    Each element is given by:

    .. math::

       \mathbf{\Pi_{\delta'\mu',\epsilon'\nu',\delta\mu,\epsilon\nu}^{(1)}}
       = \mathbf{\Omega_{\delta'\delta}\Omega_{\epsilon'\epsilon}}
       \left(\mu'\mu|\nu'\nu\right)^{(1)}
       -\mathbf{\Omega_{\delta'\epsilon}\Omega_{\epsilon'\delta}}
       \left(\mu'\nu|\nu'\mu\right)^{(1)}

       \left(\mu'\mu|\nu'\nu\right)^{(1)}
       =\left(\frac{\partial\phi_{\mu'}}{\partial a}\phi_{\mu}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\frac{\partial\phi_{\mu}}{\partial a}|
       \phi_{\nu'}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \frac{\partial\phi_{\nu'}}{\partial a}\phi_{\nu}\right)
       +\left(\phi_{\mu'}\phi_{\mu}|
       \phi_{\nu'}\frac{\partial\phi_{\nu}}{\partial a}\right)

    :param mol: Molecule class as defined by PySCF.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z

    :returns: First order 4 dimensional pi tensor.
    """

    omega = np.identity(2)
    spin_j = np.einsum("ij,kl->ikjl", omega, omega)

    twoe = mol.intor("int2e_ip1")[coord]

    j1_spatial = np.zeros((len(twoe),len(twoe),len(twoe),len(twoe)))

    for i in range(len(twoe)):

        lambda_i = int(i in range(mol.aoslice_by_atom()[atom][2],
                                  mol.aoslice_by_atom()[atom][3]))

        for j in range(len(twoe)):

            lambda_j = int(j in range(mol.aoslice_by_atom()[atom][2],
                                      mol.aoslice_by_atom()[atom][3]))

            for k in range(len(twoe)):

                lambda_k = int(k in range(mol.aoslice_by_atom()[atom][2],
                                          mol.aoslice_by_atom()[atom][3]))

                for l in range(len(twoe)):

                    lambda_l = int(l in range(mol.aoslice_by_atom()[atom][2],
                                              mol.aoslice_by_atom()[atom][3]))

                    j1_spatial[i][j][k][l] += (twoe[i][j][k][l] * lambda_i
                                               + twoe[j][i][k][l] * lambda_j
                                               + twoe[k][l][i][j] * lambda_k
                                               + twoe[l][k][i][j] * lambda_l)

    phys_j1_spatial = np.einsum("abcd->acbd", j1_spatial)
    j1 = np.kron(spin_j, phys_j1_spatial)
    k1 = np.einsum("ijkl->ijlk", j1)

    pi1 = j1 - k1

    return pi1


def get_f1(pi0, p0, hcore1, pi1, p1):

    r"""Calculate the first order fock matrix, defined by

    .. math::

        \mathbf{F^{(1)}}=\mathbf{H_{core}^{(1)}}+\mathbf{\Pi^{(1)}}\cdot
        \mathbf{P^{(0)}}+\mathbf{\Pi^{(0)}}\cdot\mathbf{P^{(1)}}

    :param pi0: 4 dimensional zeroth order Pi tensor of 2 electron integrals
    :param p0: Zeroth order density matrix
    :param hcore1: First order core hamiltonian after particular pertubation
    :param pi1: 4 dimensional first order Pi tensor of differentiated 2
            electron integrals after particular pertubation
    :param p1: First order density matrix

    :returns: First order fock matrix.
    """

    omega = np.dividentity(2)
    f1_1 = np.kron(omega, hcore1)
    f1_2 = np.einsum("ijkl,lj->ik", pi0, p1)
    f1_3 = np.einsum("ijkl,lj->ik", pi1, p0)

    f1 = f1_1 + f1_2 + f1_3

    return f1


def get_p1(f0, g0, f1, nelec, complexsymmetric: bool):

    r"""Calculates the first order density matrix containing information about
    how the molecular orbital coefficients change upon a pertubation.
    It is :math:'\mathbf{P^{(1)}}=\mathbf{Y}+\mathbf{Y^{\dagger\diamond}}'
    where Y is given by:

    .. math::

        \mathbf{Y}
        =\sum\limits_i^{occ}\sum\limits_{i'}^{vir}
        \frac{1}{\eta_i-\eta_{i'}}
        \mathbf{G}^{(0)}_i\mathbf{G}^{(0)\dagger\diamond}_{i'}
        \mathbf{F}^{(1)}
        \mathbf{G}^{(0)}_i\mathbf{G}^{(0)\dagger\diamond}_{i'}

    :param f0: The zeroth order fock matrix from which the orbital energy
            eigenvalues are obtained.
    :param g0: The zeroth order matrix of molecular orbital coefficients.
    :param f1: The first order fock matrix that is dependant on p1.
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    :returns: The first order fock matrix, from which the perturbed energy can
            be obtained.
    """

    eta0 = np.sort(np.linalg.eigvals(f0))

    nbasis = len(g0)
    nocc = nelec
    nvir = nbasis - nelec

    y = np.zeros((nbasis,nbasis))

    for i in range(nocc):
        for j in range(nvir):

            if not complexsymmetric:

                y += ((1/(eta0[i] - eta0[nocc+j]))
                      * np.linalg.multi_dot([np.outer(g0[:,i],
                                                      g0.T.conj()[:,i]),
                                            f1,
                                            np.outer(g0[:,nocc+j],
                                                     g0.T.conj()[:,nocc+j])]))

            else:

                y += ((1/(eta0[i] - eta0[nocc+j]))
                      * np.linalg.multi_dot([np.outer(g0[:,i],
                                                      g0.T[:,i]),
                                            f1,
                                            np.outer(g0[:,nocc+j],
                                                     g0.T[:,nocc+j])]))

    if not complexsymmetric:
        p1 = y + y.T.conj()
    else:
        p1 = y + y.T

    return p1


def p1_iteration(p1_guess, mol, g0, atom, coord, nelec, complexsymmetric: bool
                 ):

    r"""Calculates the first order density matrix self consistently given that
    :math:'\mathbf{P^{(1)}}' and :math:'\mathbf{F^{(1)}}' depend on one
    another.

    :param p1_guess: An initial guess for the first order denisty matrix, a
            matrix of zeros seems to work well for now, perhaps due to the
            pertubation being necessarily small, and other guesses converge to
            the same matrix.
    :param mol: Molecule class as defined by PySCF.
    :param g0: Matrix of zeroth order molecular orbital coefficients e.g from
            Qchem.
    :param atom: Input for which atom is being perturbed, with atoms numbered
            according to the PySCF molecule.
    :param coord: Input for along which coordinate the pertubation of the atom
            lies.
            coord = '0' for x
                    '1' for y
                    '2' for z
    :param nelec: The number of electrons in the molecule, determines which
            orbitals are occupied and virtual.
    :param complexsymmetric: If :const:'True', :math:'/diamond = /star'.
            If :const:'False', :math:'\diamond = \hat{e}'.

    ;returns: The converged first order density matrix.
    """

    p1 = p1_guess
    delta_p1 = 1
    iter_num = 0

    while delta_p1 > 1E-10:

        iter_num += 1
        f1 = get_f1(get_pi0(mol), get_p0(g0, complexsymmetric),
                    get_hcore1(mol, atom, coord), get_pi1(mol, atom, coord),
                    p1)

        p1_last = p1
        p1 = get_p1(get_f0(get_hcore0(mol), get_pi0(mol),
                    get_p0(g0, complexsymmetric)),
                    g0, f1, nelec, complexsymmetric)
        delta_p1 = np.max(np.abs(p1 - p1_last))

    return p1#, iter_num


def get_e1(p0, p1, f0, f1):

    p = p0 + p1
    f = f0 + f1

    e0 = np.einsum("ij,ji->", f0, p0)
    e = np.einsum("ij,ji->", f, p)

    e1 = e - e0

    return e1, e0
