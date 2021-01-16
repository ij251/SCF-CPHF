import numpy as np
from pyscf import gto, scf, grad
from CPHFfunctions import get_p0, get_hcore0, get_pi0, get_f0, get_hcore1,\
get_pi1, get_f1, get_p1_ortho, p1_iteration, get_e1_elec, get_e0_elec, get_x,\
get_e0_nuc, get_e1_nuc, get_s


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
print("\n###################################################################")
print("Zeroth order quantities:")
print("###################################################################\n")

x0 = get_x0(test_mol)
x0_inv = np.linalg.inv(x0)
# g0 = m.mo_coeff
# g0_ortho = np.dot(np.linalg.inv(x0), g0)
# g0_zeros = np.fill_diagonal(np.dot(g0_ortho.T.conj(),
#                                    g0_ortho),
#                             0)
# p0 = m.make_rdm1()

# g0 = np.array([[-0.5275465, 0, -1.5678230, 0],
#                [-0.5275465, 0, +1.5678230, 0],
#                [0, -0.5275465, 0, -1.5678230],
#                [0, -0.5275465, 0, +1.5678230]])
g0 = np.array([[-0.5846498, 0, -0.9647349, 0],
               [-0.5846498, 0, +0.9647349, 0],
               [0, -0.5846498, 0, -0.9647349],
               [0, -0.5846498, 0, +0.9647349]])
p0 = np.dot(g0[:,0:2], g0[:,0:2].T.conj())
g0_ortho = np.dot(np.linalg.inv(x0), g0)
print(g0_ortho)
p0_ortho = np.linalg.multi_dot([x0_inv, p0, x0_inv.T.conj()])

print("prtho p0:\n", p0_ortho)
hcore0 = get_hcore0(test_mol)
pi0 = get_pi0(test_mol)
f0 = get_f0(get_hcore0(test_mol), get_pi0(test_mol), p0)
f0_ortho = np.linalg.multi_dot([x0.T.conj(), f0, x0])
e0_nuc = get_e0_nuc(test_mol)
e0_elec = get_e0_elec(test_mol, p0)
#print("Canonical orthogonalisation matrix x0:\n", x0)
print("Coefficient matrix\n", g0)
print("Density matrix\n", p0)
print("Fock matrix\n", f0)
print("Zeroth order nuclear repulsion energy:\n", e0_nuc)
print("Zeroth order electronic energy:\n", e0_elec)
print("Total zeroth order energy:\n", e0_elec+e0_nuc)

"""First order quantities"""
print("\n###################################################################")
print("First order quantities:")
print("###################################################################\n")

atom_pert =1
coord_pert = 2
nelec = 2
hcore1 = get_hcore1(test_mol, atom_pert, coord_pert)
pi1 = get_pi1(test_mol, atom_pert, coord_pert)
p1_zeros = np.zeros((4,4))
p1_guess = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]) * 100
p1 = p1_iteration(p1_guess, test_mol, g0, p0, atom_pert, coord_pert, nelec,
                  False)
f1 = get_f1(pi0, p0, hcore1, pi1, p1)
e1_nuc = get_e1_nuc(test_mol, atom_pert, coord_pert)
e1_elec = get_e1_elec(test_mol, p0, p1, atom_pert, coord_pert)
print("Density matrix:\n", p1)
print("Fock matrix:\n", f1)
print("First order nuclear repulsion energy:\n", e1_nuc)
print("First order electronic energy:\n", e1_elec)
print("Total first order energy:\n", e1_elec+e1_nuc)

"""Troubleshooting"""
print("\n###################################################################")
print("Troubleshooting printing:")
print("###################################################################\n")

print("Full overlap matrix:\n", get_s(test_mol, atom_pert, coord_pert))
# print("zeroth order core hamiltonian:\n", hcore0)
# print("First order core hamiltonian:\n", hcore1)
# print("first order pi tensor:\n", pi1)
# print("hcore1 + pi1 . p0:\n", hcore1 + np.einsum("ijkl,lj->ik", pi1, p0))
# print("pi0 . p1:\n", np.einsum("ijkl,lj->ik", pi0, p1))

# f1_1 = get_f1(pi0, p0, hcore1, pi1, p1_guess)
# p1_ortho_1 = get_p1_ortho(g0, f0, f1_1, nelec, False, test_mol)
# p1_1 = np.linalg.multi_dot([x, p1_ortho_1, x.T.conj()])
# print("first f1 iteration:\n", f1_1)
# print("first p1 iteration:\n", p1_1)

# p = p0 + p1
# f = f0 + f1

# p_ortho = np.linalg.multi_dot([x_inv, p, np.linalg.inv(x.T.conj())])
# f_ortho = np.linalg.multi_dot([x.T.conj(), f, x])
# print(np.dot(p_ortho, f_ortho)-np.dot(f_ortho,p_ortho))
