import numpy as np
from pyscf import gto, scf, grad
from CPHFfunctions import get_p0, get_hcore0, get_pi0, get_f0, get_hcore1,\
get_pi1, get_f1, get_p1, p1_iteration, get_e1


h2 = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr')
m = scf.GHF(h2)
m.kernel()
g0_h2 = m.mo_coeff
p0_h2 = m.make_rdm1()
print(g0_h2)
print("\n")
print(p0_h2)
p1_zeros = np.zeros((4,4))

f0_h2 = get_f0(get_hcore0(h2), get_pi0(h2), p0_h2)
print("\n", f0_h2)
p1_h2 = p1_iteration(p1_zeros, h2, g0_h2, 0, 2, 2, 'False')
f1_h2 = get_f1(get_pi0(h2), p0_h2, get_hcore1(h2, 0, 2), get_pi1(h2, 0, 2),
                p1_h2)

#e0 = np.einsum("ij,ji->", f0_h2, p0_h2)
e0 = get_e1(p0_h2, p1_h2, f0_h2, f1_h2)[1]
print(e0)
print(gto.energy_nuc(h2))

#print(p0_h2)

#print(get_pi0(h2))

#print(get_f0(get_hcore0, get_pi0, p0_h2))










