
from pyscf import gto, scf, grad

h2 = gto.M(atom = ("H 0 0 0;"
                   "H 0 0 1;"),
            basis = 'sto3g',
            unit = 'Angstrom')

def get_grad_Hcore(molecule, atom):

    mf = scf.RHF(molecule)
    g = grad.rhf.Gradients(mf)

    grad_Hcore = g.hcore_generator(molecule)(atom)

    return grad_Hcore

print(get_grad_Hcore(h2, 0))




