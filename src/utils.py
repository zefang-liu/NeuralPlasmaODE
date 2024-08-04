"""
Utilities
"""

import scipy.constants as sc
import torch
from scipy.constants import e, m_e, gas_constant

from src import config

m_p = sc.value('proton mass')
m_d = sc.value('deuteron mass')
m_t = sc.value('triton mass')
m_a = sc.value('alpha particle mass')
m_he = sc.value('helion mass')
k = sc.value('Boltzmann constant')
R_gas = gas_constant

# Mass numbers:
A_p = 1
A_d = 2
A_t = 3
A_a = 4
A_e = m_e / m_p

# Atomic numbers:
Z_p = Z_d = Z_t = Z_e = 1
Z_a = 2
Z_be = 4
Z_ar = 18
Z_imp = {'be': Z_be, 'ar': Z_ar}


def eV2J(T_eV):
    """
    Convert a temperature from eV to J
    :param T_eV:
    :return:
    """
    T_J = T_eV * e
    return T_J


def J2eV(T_J):
    """
    Convert a temperature from J to eV
    :param T_J:
    :return:
    """
    T_eV = T_J / e
    return T_eV


def keV2J(T_keV):
    """
    Convert a temperature from keV to J
    :param T_keV:
    :return:
    """
    T_J = T_keV * 1E3 * e
    return T_J


def J2keV(T_J):
    """
    Convert a temperature from J to keV
    :param T_J:
    :return:
    """
    T_keV = T_J / (1E3 * e)
    return T_keV


def var2num(var):
    """
    Count the number of variables

    :param var: variables 'n', 'T', or 'nT'
    :return: number of variables
    """
    assert var in ['n', 'T', 'nT']

    if var == 'n':
        var_num = 1
    elif var == 'T':
        var_num = 2
    elif var == 'nT':
        var_num = 3
    else:
        var_num = 0

    return var_num


def array2tensor(arr):
    """
    Convert an array to a tensor

    :param arr: input array
    :return: output tensor
    """
    return torch.from_numpy(arr).double().to(config.device)


def arrays2tensors(t: tuple):
    """
    Convert a tuple of arrays to a tuple of tensors

    :param t: input tuple of arrays
    :return: output tuple of tensors
    """
    return tuple(map(array2tensor, t))


def tensor2array(x: torch.Tensor):
    """
    Convert a tensor to an array

    :param x: input tensor
    :return: output array
    """
    return x.detach().cpu().numpy()


def tensors2arrays(t: tuple):
    """
    Convert a tuple of tensors to a tuple of arrays

    :param t: input tuple of tensors
    :return: output tuple of arrays
    """
    return tuple(map(tensor2array, t))
