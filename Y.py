import numpy as np

H2_G0_rhf = np.array([
    [-0.5780280, 0.9965031],
    [-0.5780280, -0.9965031]
    ])

H2_G0_ghf = np.block([
    [H2_G0_rhf, np.zeros((2,2))],
    [np.zeros((2,2)), H2_G0_rhf],
    ])

# Not sure what this is for?
H2_G0_ghf[:,[1,2]] = H2_G0_ghf[:,[2,1]]

def get_P1(eta0, G0, F1, nelec, complexsymmetric: bool):

    'G0 is a GHF coefficient matrix of molecular orbital coefficients in terms
    of basis functions of dimension 2*N_spatial, allowing for mixing of alpha
    and beta spin functions. ie G0 = np.array([N_basis, N_basis]), so G0[i,j] 
    is the coefficient of the ith basis function in the jth MO. It has been 
    ordered in occupied and virtual blocks'

    'eta0 is a vector of orbital energy eigenvalues
    F1 is the first order Fock matrix that depends on P1
    (the first order density matrix) and hence Y'

    # Perhaps use nbasis = G0.shape[0] for better clarity.
    # Also, add an assert statement to make sure that G0 has the right number of
    # columns.
    nbasis = len(G0)
    nocc = nelec
    nvir = nbasis - nelec

    # This creates an array of floats. There will be issues later on when
    # complex numbers are added as casting from complex to real will cause the
    # imaginary parts to be lost. Set the datatype of y to the maximum datatype
    # of the input G0 and F1. Each datatype as a unique number (see
    # https://numpy.org/doc/stable/reference/generated/numpy.dtype.num.html#numpy.dtype.num).
    # Perhaps you can use this to work out what the maximum datatype is.
    y = np.zeros((nbasis,nbasis))

    for i in range(nocc):
        for j in range(nvir):

            if not complexsymmetric:
                
                y += 1/(eta0[i] - eta0[nocc+j]) * 
                np.linalg.multi_dot([
                    np.outer(G0[:,i], G0.T.conj()[:,i]),
                    F1,
                    np.outer(G0[:,nocc+j], G0.T.conj()[:,nocc+j])
                    ])

            else:
                
                y += 1/(eta0[i] - eta0[nocc+j]) * 
                np.linalg.multi_dot([
                    np.outer(G0[:,i], G0.T[:,i]),
                    F1,
                    np.outer(G0[:,nocc+j], G0.T()[:,nocc+j])
                    ])
             
    if not complexsymmetric:
        P1 = y + y.T.conj()
    else:
        P1 = y + y.T

    return P1





    



