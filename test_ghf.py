import numpy as np
import pyscf
from pyscf import gto, scf, grad
from ghf_CPHFfunctions import make_ghf, get_p1
from rhf_CPHFfunctions import get_hcore0, get_pi0, get_f0, get_p0,\
get_x_lowdin

h2 = gto.M(
        atom = (
            f"H 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)
nelec_h2 = 2

oh = gto.M(
        atom = (
            f"O 0 0 0;"
            f"H 0 0 1;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = -1,)
nelec_oh = 10

h3 = gto.M(
        atom = (
            f"H 0 0.3745046 -1.9337695;"
            f"H 0 -0.7492090 0;"
            f"H 0 0.3745046 1.9337695;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 1,
        spin = 0)
nelec_h3 = 2

h2o = gto.M(
        atom = (
            f"O 0 0.1088584 0;"
            f"H 0 -0.8636449 1.2990232;"
            f"H 0 -0.8636449 -1.2990232;"
        ),
        basis = 'sto-3g',
        unit = 'Bohr',
        charge = 0,
        spin = 0)
nelec_h2o = 10

nelec = nelec_oh
mol = oh
m = scf.RHF(mol)
m.kernel()

g0_rhf = m.mo_coeff
g0_ghf = make_ghf(g0_rhf, nelec)
print("g0_rhf:\n", g0_rhf)
print("g0_ghf:\n", g0_ghf)

x = get_x_lowdin(mol)
f0 = get_f0(get_hcore0(mol), get_pi0(mol), get_p0(g0_rhf, False, int(nelec/2)))
f0_x = np.linalg.multi_dot([x.T.conj(), f0, x])
eta0, g0_rhf_x = np.linalg.eig(f0_x)
print("eigenvalues before sorting:\n", eta0)
index = np.argsort(eta0)
eta0 = eta0[index]
print("eigenvalues after sorting:\n", eta0)

g0_rhf_x = g0_rhf_x[:, index]
g0_rhf = np.dot(x, g0_rhf_x)

g0_ghf = make_ghf(g0_rhf, nelec)
print("g0_rhf:\n", g0_rhf)
print("g0_ghf:\n", g0_ghf)
