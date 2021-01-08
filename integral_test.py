from pyscf import gto, scf
import numpy as np


mol = gto.M(atom=("Li 0 0 0;"
                  "H 0 0 1;"),
            basis='sto3g',
            unit='Angstrom')

#m = mol.intor("int1e_ovlp")
#print(m)

#n = mol.intor("int1e_ipovlp")
#print(n)

#nuc = mol.intor("int1e_ipnuc")
#kin = mol.intor("int1e_ipkin")

#h = nuc + kin

#print(h)

twoe = mol.intor("int2e_ip1")
print(twoe[0][:,0,0,0])

print(len(twoe))

#pi1 = np.zeros((len(twoe[2]),len(twoe[2]),len(twoe[2]),len(twoe[2])))
#print(pi1)

