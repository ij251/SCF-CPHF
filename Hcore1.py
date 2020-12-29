from pyscf import gto, scf, grad

h2 = gto.M(atom = ("H 0 0 0;"
                   "H 0 0 1;"),
            basis = 'sto3g',
            unit = 'Angstrom')


def get_hcore1(molecule, atom):

    '''function to generate first order core hamiltonian,
    requires which atom is being perturbed as an input'''

    mf = scf.RHF(molecule)
    g = grad.rhf.Gradients(mf)

    hcore1 = g.hcore_generator(molecule)(atom)

    return hcore1


print(get_hcore1(h2, 0))




