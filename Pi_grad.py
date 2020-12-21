from pyscf import gto, scf, grad
import numpy as np

h2 = gto.M(atom = ("H 0 0 0;"
                   "H 0 0 1;"),
            basis = 'sto3g',
            unit = 'Angstrom')




def get_Pi1(molecule):

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

    #j1z = np.kron(spin_j, j1z_spatial)
    j1z = j1z_spatial
    k1z = np.einsum("ijkl->ilkj", j1z)

    pi1z = j1z - k1z                

    return pi1z
    
print(get_Pi1(h2))
    

