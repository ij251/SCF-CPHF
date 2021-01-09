import numpy as np
from pyscf import gto, scf, grad
from CPHFfunctions import get_p0, get_hcore0, get_pi0, get_f0, get_hcore1,\
get_pi1, get_f1, get_p1_ortho, p1_iteration, get_e1, get_e0, get_x


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


"""Zeroth order quantities"""
x = get_x(test_mol)
x_inv = np.linalg.inv(x)
g0 = m.mo_coeff
g0_ortho = np.dot(np.linalg.inv(x), g0)
p0 = m.make_rdm1()
p0_ortho = np.linalg.multi_dot([x_inv, p0, x_inv.T.conj()])
hcore0 = get_hcore0(test_mol)
pi0 = get_pi0(test_mol)
f0 = get_f0(get_hcore0(test_mol), get_pi0(test_mol), p0)
f0_ortho = np.linalg.multi_dot([x.T.conj(), f0, x])
e0 = get_e0(hcore0, pi0, p0)
print("\nZeroth order quantities:\n")
print("Canonical orthogonalisation matrix x:\n", x)
print("Coefficient matrix\n", g0)
print("Density matrix\n", p0)
print("Fock matrix\n", f0)
print("Zeroth order energy:\n", e0)
print(g0_ortho)

"""First order quantities"""
atom_pert = 0
coord_pert = 2
nelec = 2
hcore1 = get_hcore1(test_mol, atom_pert, coord_pert)
pi1 = get_pi1(test_mol, atom_pert, coord_pert)
p1_zeros = np.zeros((4,4))
p1 = p1_iteration(p1_zeros, test_mol, g0, p0, atom_pert, coord_pert, nelec,
                  'False')
f1 = get_f1(pi0, p0, hcore1, pi1, p1)
e1 = get_e1(p0, p1, hcore0, hcore1, pi0, pi1, e0)
print("\nFirst order quantities:\n")
print("Density matrix:\n", p1)
print("Fock matrix:\n", f1)
print("First order energy:\n", e1)
