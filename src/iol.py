"""
Ion Orbit Loss
"""
from typing import Tuple

import numpy as np
import torch
from scipy.constants import pi, mu_0

from src import config
from src.utils import *


class IOL(object):
    """
    Ion Orbit Loss
    """

    def __init__(self, R0: float, r0: float, mi: float, B0: float, IP: float, reactor_type: str = 'd3d'):
        """
        Initialize the IOL class
        
        :param R0: major radius [m]
        :param r0: minor radius [m]
        :param mi: ion mass [kg]
        :param B0: toroidal field at magnetic axis [T]
        :param IP: plasma current [A]
        :param reactor_type: reactor type
        """
        self.rho = (config.rho_core + config.rho_edge) / 2
        self.rho_s = 1
        self.zeta0 = 1
        self.f_phi0 = 1
        self.f_phis = 1

        self.R0 = R0
        self.r0 = r0
        self.mi = mi
        self.B_phi0 = B0
        self.IP = IP
        self.reactor_type = reactor_type

        self.V0 = np.min(self.get_V0(self.rho, np.linspace(-pi, pi, 180), self.zeta0))
        self.E_min = self.get_E_min(self.V0)

    def get_phi(self, rho):
        """
        Electric potential fitted for DIII-D

        :param rho: normalized minor radius
        :return: phi [V]
        """
        if self.reactor_type == 'd3d':
            pc_phi_core = [6.75, 18, -33, 0, 6.12]
            pc_phi_edge = [-78025, 336437, -578276, 495331, -211459, 35992]
            rho_edge_phi = 0.85
        else:  # self.reactor_type == 'iter'
            pc_phi_core = [17.75, 15.6667, -59.5, 10., 14.32308421]
            pc_phi_edge = [388., -721., 333]
            rho_edge_phi = 0.925

        p_phi_core = np.poly1d(pc_phi_core)
        p_phi_edge = np.poly1d(pc_phi_edge)
        phi = np.where(rho < rho_edge_phi, p_phi_core(rho), p_phi_edge(rho)) * self.r0 * 1E3
        return phi

    def get_h(self, rho, theta):
        """
        Geometry h

        :param rho: normalized minor radius
        :param theta: poloidal angle
        :return: h [m]
        """
        h = 1 + rho * self.r0 / self.R0 * np.cos(theta)
        return h

    def get_R(self, rho, theta):
        """
        Geometry R

        :param rho: normalized minor radius
        :param theta: poloidal angle
        :return: R [m]
        """
        R = self.R0 * self.get_h(rho, theta)
        return R

    def get_B_phi(self, rho, theta):
        """
        Toroidal field

        :param rho: normalized minor radius
        :param theta: poloidal angle
        :return: B_phi [T]
        """
        B_phi = self.B_phi0 / self.get_h(rho, theta)
        return B_phi

    def get_psi(self, rho):
        """
        Magnetic flux

        :param rho: normalized minor radius
        :return: psi [Wb]
        """
        psi = 0.5 * (mu_0 * self.IP) / (2 * pi) * self.R0 * rho ** 2
        return psi

    def get_V0(self, rho, theta, zeta0):
        """
        Initial ion velocity

        :param rho: normalized minor radius
        :param theta: poloidal angle
        :param zeta0: direction cosine
        :return: V0 [m/s]
        """
        R = self.get_R(rho, theta)
        psi_0 = self.get_psi(rho)
        psi_s = self.get_psi(self.rho_s)
        phi_0 = self.get_phi(rho)
        phi_s = self.get_phi(self.rho_s)
        B_0 = self.get_B_phi(rho, theta)
        B_s = self.get_B_phi(self.rho_s, theta)

        a = (np.abs(B_s / B_0) * (self.f_phi0 / self.f_phis) * zeta0) ** 2 - 1 \
            + (1 - zeta0 ** 2) * abs(B_s / B_0)
        b = (2 * e * (psi_0 - psi_s)) / (R * self.mi * self.f_phis) \
            * (abs(B_s / B_0) * (self.f_phi0 / self.f_phis) * zeta0)
        c = ((e * (psi_0 - psi_s)) / (R * self.mi * self.f_phis)) ** 2 \
            - (2 * e * (phi_0 - phi_s)) / self.mi

        delta = b ** 2 - 4 * a * c
        V0_neg = (-b - np.sqrt(delta)) / (2 * a)
        return V0_neg

    def get_E_min(self, V0):
        """
        Minimum energy

        :param V0: initial ion velocity [m/s]
        :return: E_min [eV]
        """
        E_min = J2eV(self.mi * V0 ** 2 / 2)
        return E_min

    def get_epsilon_min(self, E_min: torch.Tensor, Ti: torch.Tensor) -> torch.Tensor:
        """
        Reduced minimum energy

        :param E_min: minimum energy [eV]
        :param Ti: ion temperature [eV]
        :return: epsilon_min []
        """
        epsilon_min = E_min / Ti
        return epsilon_min

    def get_F_orb(self, epsilon_min: torch.Tensor) -> torch.Tensor:
        """
        Particle loss fraction

        :param epsilon_min: reduced minimum energy []
        :return: F_orb []
        """
        F_orb = torch.igammac(torch.ones_like(epsilon_min, dtype=torch.float) * 3 / 2, epsilon_min)
        return F_orb

    def get_E_orb(self, epsilon_min: torch.Tensor) -> torch.Tensor:
        """
        Energy loss fraction

        :param epsilon_min: reduced minimum energy []
        :return: E_orb []
        """
        E_orb = torch.igammac(torch.ones_like(epsilon_min, dtype=torch.float) * 5 / 2, epsilon_min)
        return E_orb

    def get_loss_fractions(self, Ti: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loss fractions

        :param Ti: ion temperature [eV]
        :return: F_orb, E_orb
        """
        E_min = torch.ones_like(Ti, dtype=torch.float) * self.E_min
        epsilon_min = self.get_epsilon_min(E_min, Ti)
        F_orb = self.get_F_orb(epsilon_min)
        E_orb = self.get_E_orb(epsilon_min)
        return F_orb, E_orb
