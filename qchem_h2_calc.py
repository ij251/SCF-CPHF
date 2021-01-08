import numpy as np
from pyscf import gto, scf, grad
from CPHFfunctions import get_p0, get_hcore0, get_pi0, get_f0, get_hcore1,\
get_pi1, get_f1, get_p1, p1_iteration, get_e1


h2 = gto.M(atom=("H 0 0 0;"
                     "H 0 0 1;"),
                basis="sto-3g",
                unit="Bohr")

h2_g0_rhf = np.array([
    [-0.5780280, 0.9965031],
    [-0.5780280, -0.9965031]
    ])

h2_g0_ghf = np.block([
    [h2_g0_rhf, np.zeros((2,2))],
    [np.zeros((2,2)), h2_g0_rhf],
    ])

h2_g0_ghf[:,[1,2]] = h2_g0_ghf[:,[2,1]]
print(h2_g0_ghf)


p1_zeros = np.zeros((4,4))

p1_guess2 = np.array([[0.1, 0.1, 0.1, 0.1],
                      [0.02, 0.2, 0.3, 0.04],
                      [0.05, 0.06, 0.2, 0.03],
                      [0, 0.0456, 0.01, 0]])

p1_guess3 = np.array([[434,4,7,9],
                      [0,20,89,456],
                      [4,999,345,56],
                      [3,77,1,0]])

p1_actual = np.array([[ 0.000000E0, -8.943927E-1,  0.000000E0, -5.408334E0],
                      [-8.943927E-1, -1.788785E0, -7.763035E-3, -5.416097E0],
                      [ 0.000000E0, -7.763035E-3,  0.000000E0,  4.521705E0],
                      [-5.408334E0, -5.416097E0,  4.521705E0,  9.043410E0]])

p0_h2 = get_p0(h2_g0_ghf, "False")
p1_h2 = p1_iteration(p1_zeros, h2, h2_g0_ghf, 0, 2, 2, "False")
f0_h2 = get_f0(get_hcore0(h2), get_pi0(h2), p0_h2)
f1_h2 = get_f1(get_pi0(h2), p0_h2, get_hcore1(h2, 0, 2), get_pi1(h2, 0, 2),
               p1_h2)

e0 = get_e1(p0_h2, p1_h2, f0_h2, f1_h2)[1]
print(e0)
print("\n")
#print(get_pi0(h2))
#print(p0_h2)
#print("\n")
#print(h2_g0_ghf)
#print("\n")
#print(p1_h2)
#print("\n")
#print(get_pi0(h2))
#print("\n")
#print(f0_h2)
#print("\n")

