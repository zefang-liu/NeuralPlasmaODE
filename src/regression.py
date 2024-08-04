"""
Regression for ITER
"""

import numpy as np
from sklearn.linear_model import Ridge

from src import config


class Regression(object):
    """
    Regression for ITER
    """

    def __init__(self):
        """
        Initialize the Regression
        """
        self.rho_core = config.rho_core
        self.rho_edge = config.rho_edge
        self.rho_sol = config.rho_sol
        self.drho_edge_sol = (self.rho_sol - self.rho_core) / 2

        self.R = 6.2  # [m]
        self.a = 2.0  # [m]
        self.Bt = 5.3  # [T]
        self.Ip = 15.0  # [MA]
        self.A = self.R / self.a
        self.M = 2.5
        self.kappa95 = 1.7
        self.q95 = 3.0

        self.alpha = \
            {'Bt': -3.5, 'n': 0.9, 'T': 1.0, 'grad_T': 1.2, 'q': 3.0, 'kappa': -2.9, 'M': -0.6, 'R': 0.7, 'a': -0.2}

    def get_tau_h98(self, Ip, Bt, ne, P, R, kappa, A, M):
        """
        Thermal energy confinement time with the ITER H-98 P(y,2) scaling

        :param Ip: plasma current [MA]
        :param Bt: toroidal magnetic field [T]
        :param ne: line averaged electron density [10^19 m^-3]
        :param P: heating power [MW]
        :param R: major radius [m]
        :param kappa: elongation []
        :param A: aspect ratio []
        :param M: hydrogenic atomic mass number []
        :return: tau_h98 [s]
        """
        tau_h98 = 0.0562 * Ip ** 0.93 * Bt ** 0.15 * ne ** 0.41 * P ** (-0.69) * R ** 1.97 * kappa ** 0.78 \
                  * A ** (-0.58) * M ** 0.19
        return tau_h98

    def get_chi_e_edge(self, ne_core, ne_edge, Te_core, Te_edge, tau_h98):
        """
        Electron thermal diffusivity for the edge node

        :param ne_core: core electron density [10^19 m^-3]
        :param ne_edge: edge electron density [10^19 m^-3]
        :param Te_core: core electron temperature [keV]
        :param Te_edge: edge electron temperature [keV]
        :param tau_h98: energy confinement time [s]
        :return: chi_e_edge [m^2/s]
        """
        r_core = self.rho_core * self.a
        r_edge = self.rho_edge * self.a
        dr_edge_sol = self.drho_edge_sol * self.a
        chi_e_edge = dr_edge_sol / (2 * r_edge) * (r_core ** 2 * ne_core * Te_core + r_edge ** 2 * ne_edge * Te_edge) \
                     / (ne_edge * Te_edge * tau_h98)
        return chi_e_edge

    def get_average(self, ne_core, ne_edge):
        """
        Averaged density or temperature

        :param ne_core: core electron density [10^19 m^-3] or temperature [keV]
        :param ne_edge: edge electron density [10^19 m^-3] or temperature [keV]
        :return: ne [10^19 m^-3] or Te [eV]
        """
        ratio = self.rho_core ** 2 / self.rho_edge ** 2
        ne = ratio * ne_core + (1 - ratio) * ne_edge
        return ne

    def get_data(self):
        """
        Data for regression

        :return: X, y
        """
        X, y = [], []
        dr_edge_sol = self.drho_edge_sol * self.a

        for ne_core in np.linspace(5, 15, 11):  # [10^19 m^-3]
            for ne_edge in ne_core / np.linspace(1, 2, 5):  # [10^19 m^-3]
                for Te_core in np.logspace(np.log10(5), np.log10(50), 21):  # [keV]
                    for Te_edge in Te_core / np.linspace(1, 5, 5):  # [keV]
                        for P in np.linspace(50, 250, 21):  # [MW]
                            grad_Te_edge = np.abs(Te_edge) / dr_edge_sol
                            x = np.log([ne_edge, Te_edge, grad_Te_edge])
                            X.append(x)

                            ne = self.get_average(ne_core, ne_edge)
                            tau_h98 = self.get_tau_h98(Ip=self.Ip, Bt=self.Bt, ne=ne, P=P, R=self.R,
                                                       kappa=self.kappa95, A=self.A, M=self.M)
                            chi_e_edge = self.get_chi_e_edge(ne_core, ne_edge, Te_core, Te_edge, tau_h98)
                            y.append(np.log(chi_e_edge))

        X = np.array(X)
        y = np.array(y) - self.alpha['Bt'] * np.log(self.Bt) - self.alpha['q'] * np.log(self.q95) \
            - self.alpha['kappa'] * np.log(self.kappa95) - self.alpha['M'] * np.log(self.M) \
            - self.alpha['R'] * np.log(self.R) - self.alpha['a'] * np.log(self.a)

        return X, y

    def fit_data(self):
        """
        Fit data

        :return: alpha_H, alphas = [alpha_n, alpha_T, alpha_grad_T]
        """
        X, y = self.get_data()
        reg = Ridge(alpha=0.1).fit(X, y)
        alpha_H = np.exp(reg.intercept_)
        alphas = reg.coef_
        print(np.round(alpha_H, 4), np.round(alphas, 4))
        return alpha_H, alphas


if __name__ == '__main__':
    regression = Regression()
    regression.fit_data()
