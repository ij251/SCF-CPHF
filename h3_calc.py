import numpy as np
from pyscf import gto, scf, grad
from gerratt_CPHFfunctions import get_p0, get_p1, get_hcore0, get_pi0,\
get_f0, get_s1, get_hcore1, get_pi1, get_f1, get_g1_x, g1_iteration,\
get_e0_elec, get_e0_nuc, get_e1_elec, get_e1_nuc, make_ghf

mol = gto.M(
        atom = (
            f"H 0 0.3745046 -1.9337695;"
            f"H 0 -0.7492090 0;"
            f"H 0 0.3745046 1.9337695;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 1,
        spin = 0)
m = scf.RHF(mol)
m.kernel()
nelec = 2
atom = 1
coord = 1
g0_rhf = m.mo_coeff
g0 = make_ghf(g0_rhf, nelec)

x = g0
g1_guess = np.zeros_like(g0)
g1 = g1_iteration(False, mol, g0, x, atom, coord, nelec, g1_guess)
print(g1)

e1_elec = get_e1_elec(mol,g0, g1, atom, coord, False, nelec)
print(e1_elec)
print(get_e1_nuc(mol, atom, coord))

print("geometry 1:\n")
print("1 electron energy:\n-2.18234905626298"
print("2 electron energy:\n0.5310456899")
print("Nuclear repulsion energy:\n1.152793145205")
print("total energy:\n0.5310456899")

