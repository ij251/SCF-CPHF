from pyscf import gto, scf


mol = gto.M(atom=("H 0 0 0;"
                  "H 0 0 1;"),
            basis='sto3g',
            unit='Angstrom')

m = mol.intor("int1e_ovlp")
#print(m)

n = mol.intor("int1e_ipovlp")
print(n)

nuc = mol.intor("int1e_ipnuc")
kin = mol.intor("int1e_ipkin")

h = nuc + kin

print(h)

e_2 = mol.intor("int2e")

#print(e_2)

