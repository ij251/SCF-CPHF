import numpy as np
import pyscf
from pyscf import gto, scf, grad
from ghf_CPHFfunctions import get_p0, get_p1, get_hcore0, get_pi0,\
get_f0, get_s1, get_hcore1, get_pi1, get_f1, get_g1_x, g1_iteration,\
get_e0_elec, get_e0_nuc, get_e1_elec, get_e1_nuc, make_ghf
from without_p1 import get_pe0, get_e1_without_p1

mol = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

mol2 = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1.05;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)

m = scf.RHF(mol)
scf.hf.energy_elec(mol)[0]
scf.hf.energy_tot(mol)
nelec = 2
atom = 0
coord = 2
if coord == 0:
    pert = "x"
elif coord == 1:
    pert = "y"
elif coord == 2:
    pert = "z"

# mol = gto.M(
#         atom = (
#             f"H 0 0.3745046 -1.9337695;"
#             f"H 0 -0.7492090 0;"
#             f"H 0 0.3745046 1.9337695;"
#         ),
#         basis = '6-31g',
#         unit = 'Bohr',
#         charge = 1,
#         spin = 0)
# m = scf.RHF(mol)
# nelec = 2
# atom = 1
# coord = 1

print("\nAny transformed quantities using the x matrix are denoted by _x")
"""Zeroth order quantities"""
print("\n###################################################################")
print("Zeroth order quantities:")
print("###################################################################\n")

m.kernel()

g0_rhf = m.mo_coeff
g0 = make_ghf(g0_rhf, 2)

p0 = get_p0(g0, False, 2)
x = g0
x_inv = np.linalg.inv(x)
g0_x = np.dot(x_inv, g0)

hcore0 = get_hcore0(mol)
pi0 = get_pi0(mol)
f0 = get_f0(hcore0, pi0, p0)
f0_x = np.linalg.multi_dot([x.T.conj(), f0, x])
eta0 = np.linalg.eig(f0_x)[0]
index = np.argsort(eta0)
eta0 = eta0[index]

f0_prime = hcore0 + 0.5 * np.einsum("ijkl,jl->ik", pi0, p0)

print("\ng0:\n", g0)
print("g0_x:\n", g0_x)
print("e0_elec:\n", np.trace(np.dot(f0_prime, p0)))

"""First order quantities"""
print("\n###################################################################")
print("First order quantities:")
print("###################################################################\n")

print("Atom", atom, "at coordinates", mol.atom_coord(atom),
      "is perturbed in the", pert, "direction\n")

g1_x_zeros = np.zeros_like(g0)
g1_x_guess_1 = np.array([[1, 1, 0, 0],
                         [1, 1, 4, 0],
                         [0, 456, 1, 1],
                         [0, 0, 1, 1]]) * 100

s1 = get_s1(mol, atom, coord)
s1_x = np.linalg.multi_dot([x.T.conj(), s1, x])

g1 = g1_iteration(False, mol, g0, x, atom, coord, nelec, g1_x_zeros)
p1 = get_p1(g0, g1, False, 2)
g1_x = np.dot(x_inv, g1)

hcore1 = get_hcore1(mol, atom, coord)

e1_elec = get_e1_elec(mol, g0, g1,atom, coord, False, nelec)
e1_elec_1e = get_e1_elec(mol, g0, g1,atom, coord, False, nelec)[1]
e1_elec_2e = get_e1_elec(mol, g0, g1,atom, coord, False, nelec)[2]
e1_elec = get_e1_elec(mol, g0, g1,atom, coord, False, nelec)[0]
e1_nuc = get_e1_nuc(mol, atom, coord)
e1 = e1_elec + e1_nuc

e1_without_p1 = get_e1_without_p1(g0_rhf, mol, atom, coord, nelec)

print("########################")
print("1 electron e1_elec:\n", e1_elec_1e)
print("2 electron e1_elec:\n", e1_elec_2e)
print("total e1_elec:\n", e1_elec)
print("Nuclear repulsion e1_nuc:\n", get_e1_nuc(mol, atom, coord))
print("Total e1:\n", get_e1_nuc(mol, atom, coord) + e1_elec)
print("\ne1_elec without p1 from new dimension:\n", e1_without_p1)
print("########################")
def compare_pyscf_energy(mol_a, mol_b):

    print("for 2 geometries where one coordinate has been perturbed by\
          0.05 bohr:")
    print("Total energy difference:\n", scf.hf.energy_tot(mol_b)
                                        - scf.hf.energy_tot(mol_a))

compare_pyscf_energy(mol, mol2)
print("\ng1:\n", g1)
print("p1:\n", p1)

# print("\ne1_elec Tr(F'(1)P(0) + F'(0)P(1)):\n", e1_elec)
# print("e1_nuc:\n", e1_nuc)
# print("e1 total:\n", e1_elec + e1_nuc)
# print("e1_qchem:\n-0.3650435")
# print("\nHcore1:\n", hcore1)


print("\nMine * 0.05:\n",
      "1 electron: 0.0879\n",
      "2 electron: 0.004816\n",
      "Nuclear:", e1_nuc*0.05, "\n",
      "Total:", 0.05*(e1_elec+e1_nuc), "\n",
      "Qchem difference:\n",
      "1 electron: 0.03631\n",
      "2 electron: 0.004849\n",
      "Nuclear: 0.04762\n"
      "Total: 0.01616")


g = grad.rhf.Gradients(m)
print("\n###################################################################")
print("From PySCF rhf gradient source code:")
print("###################################################################\n")
print("In directory:\n", pyscf.__file__)
g.kernel()
print("###################################################################\n")
