"""
Particle and Power Sources
"""
import numpy as np

from src import config


class Heater(object):
    """
    Particle and power fractions
    """

    def __init__(self):
        """
        Initialize the Heater
        """
        rho_core = config.rho_core
        rho_edge = config.rho_edge

        pc_nbi = np.array([-20.44, 45.92, -35.89, 10.42, 0.])
        p_nbi = np.poly1d(pc_nbi)
        pp_nbi = np.append(p_nbi, 0)
        pp_nbi_int = np.poly1d(np.polyint(pp_nbi))

        lambda_ch = 6
        p_ch = lambda x: np.exp(-lambda_ch * x) - np.exp(-lambda_ch)
        pp_ch_int = lambda x: - 0.5 * np.exp(-lambda_ch) * x ** 2 \
                              + (1 - np.exp(-lambda_ch * x) * (1 + lambda_ch * x)) / lambda_ch ** 2

        lambda_gas = 20
        p_gas = lambda x: np.power(10.0, -lambda_gas * (1 - x))
        pp_gas_int = lambda x: np.power(10.0, -lambda_gas) * (np.power(10.0, lambda_gas * x) - 1) \
                               / (lambda_gas * np.log(10.0))

        pc_ecr = np.array([-1.333, 3.314, -2.335, 0.118, 0.238])
        p_ecr = np.poly1d(pc_ecr)
        pp_ecr = np.append(p_ecr, 0)
        pp_ecr_int = np.poly1d(np.polyint(pp_ecr))

        self.f_nbi_core = self.get_fraction(pp_nbi_int, 0, rho_core)
        self.f_nbi_edge = self.get_fraction(pp_nbi_int, rho_core, rho_edge)

        self.f_ch_core = self.get_fraction(pp_ch_int, 0, rho_core)
        self.f_ch_edge = self.get_fraction(pp_ch_int, rho_core, rho_edge)

        self.f_gas_core = self.get_fraction(pp_gas_int, 0, rho_core)
        self.f_gas_edge = self.get_fraction(pp_gas_int, rho_core, rho_edge)

        self.f_ecr_core = self.get_fraction(pp_ecr_int, 0, rho_core)
        self.f_ecr_edge = self.get_fraction(pp_ecr_int, rho_core, rho_edge)

    def get_fraction(self, pp_int, rho0, rho1):
        """
        Nodal fraction

        :param pp_int: integration of p(rho) * p
        :param rho0: start normalized radius
        :param rho1: end normalized radius
        :return: f_nbi_node
        """
        f_nbi_node = 1 / (rho1 ** 2 - rho0 ** 2) * (pp_int(rho1) - pp_int(rho0)) / (pp_int(1) - pp_int(0))
        return f_nbi_node


if __name__ == '__main__':
    pass
    # heater = Heater()
    # print(heater.f_nbi_core, heater.f_nbi_edge,
    #       heater.f_ch_core, heater.f_ch_edge,
    #       heater.f_gas_core, heater.f_gas_edge,
    #       heater.f_ecr_core, heater.f_ecr_edge)
