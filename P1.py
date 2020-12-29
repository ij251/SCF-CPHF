import numpy as np

h2_g0_rhf = np.array([
    [-0.5780280, 0.9965031],
    [-0.5780280, -0.9965031]
    ])

h2_g0_ghf = np.block([
    [h2_g0_rhf, np.zeros((2,2))],
    [np.zeros((2,2)), h2_g0_rhf],
    ])

h2_g0_ghf[:,[1,2]] = h2_g0_ghf[:,[2,1]]

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



