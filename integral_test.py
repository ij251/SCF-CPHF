from pyscf import gto, scf
import numpy as np


test_mol = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr')
test_mol.charge = 0
test_mol.spin = 0
m = scf.GHF(test_mol)
m.kernel()

#m = mol.intor("int1e_ovlp")
test_mol2 = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 2;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr')
#print(m)

#n = mol.intor("int1e_ipovlp")
#print(n)

atom = 0
coord = 2
#nuc = mol.intor("int1e_ipnuc")
#kin = mol.intor("int1e_ipkin")
s0 = test_mol.intor("int1e_ovlp")
#h = nuc + kin
#print(h)
onee = test_mol.intor("int1e_ipovlp")
print(test_mol.intor("int1e_ipovlp"))
print("ovlp difference:\n", test_mol.intor("int1e_ovlp")
                            - test_mol2.intor("int1e_ovlp"))
# twoe = mol.intor("int2e_ip1")
s1 = np.zeros_like(s0)
print(s1)
for i in range(s0.shape[1]):
    lambda_i = int(i in range(test_mol.aoslice_by_atom()[atom][2],
                              test_mol.aoslice_by_atom()[atom][3]))
    for j in range(s0.shape[1]):
        lambda_j = int(j in range(test_mol.aoslice_by_atom()[atom][2],
                                  test_mol.aoslice_by_atom()[atom][3]))

        s1[i][j] += onee[coord][i][j]*lambda_i + onee[coord][j][i]*lambda_j

print(s1)
# print(twoe[0][:,0,0,0])

# print(len(twoe))

#pi1 = np.zeros((len(twoe[2]),len(twoe[2]),len(twoe[2]),len(twoe[2])))
#print(pi1)

