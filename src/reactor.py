"""
Fusion Reactors
"""
from typing import Tuple, Iterable

import numpy as np
import torch.nn as nn
from scipy.constants import pi, epsilon_0, torr, N_A
from torch import sqrt, log, arctan, exp, erf
from torchinterp1d import Interp1d

from src.heater import Heater
from src.iol import IOL
from src.preprocessor import Preprocessor, PreprocessorITER
from src.reaction import Reaction
from src.utils import *


class Reactor0D(object):
    """
    Zero-Dimension Fusion Reactor
    """

    def __init__(self, shot_num: int = None, dim: int = 0, tuned: bool = False, net: nn.Module = None):
        """
        Initialize the 0D reactor

        :param shot_num: shot number
        :param dim: dimension of the reactor
        :param tuned: true for tuned confinement times
        :param net: the neural network for tuned confinement times
        """
        if tuned:
            assert net is not None

        self.dim = dim
        self.num_vars = config.num_vars[self.dim]
        self.num_outputs = config.num_outputs[self.dim]
        self.shot_num = shot_num
        self.interp1d = Interp1d()
        self.device = config.device
        self.impurity_charge = config.impurity_charge
        self.C_gas = config.C_gas
        self.system_count = 0

        if self.shot_num is not None:
            preprocessor = Preprocessor()
            self.shot = preprocessor.preprocess(self.shot_num)
            self.time = self.shot['time']
        else:
            self.shot = None
            self.time = None

        self.reaction = Reaction()
        self.tuned = tuned
        self.net = net

    def get_shot(self) -> dict:
        """
        Shot

        :return: shot
        """
        return self.shot

    def get_time(self) -> torch.Tensor:
        """
        Time slices

        :return: t [ms]
        """
        return self.shot['time']

    def get_ne(self, t: torch.Tensor) -> torch.Tensor:
        """
        Electron density

        :param t: time [s]
        :return: ne [m^-3]
        """
        return self.interp1d(self.time, self.shot['ne'], torch.reshape(t, (-1,))).squeeze()

    def get_ni(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ion density

        :param t: time [s]
        :return: ni [m^-3]
        """
        return self.get_ne(t) - self.impurity_charge * self.get_nc(t)

    def get_nc(self, t: torch.Tensor) -> torch.Tensor:
        """
        Impurity density

        :param t: time [s]
        :return: nc [m^-3]
        """
        return self.interp1d(self.time, self.shot['nc'], torch.reshape(t, (-1,))).squeeze()

    def get_n0(self, t: torch.Tensor) -> torch.Tensor:
        """
        Neutral hydrogen density (1E15 m^-3)

        :param t: time [s]
        :return: n0 [m^-3]
        """
        return torch.ones_like(t) * 1E15

    def get_Te(self, t: torch.Tensor) -> torch.Tensor:
        """
        Electron temperature

        :param t: time [s]
        :return: Te [eV]
        """
        return self.interp1d(self.time, self.shot['te'], torch.reshape(t, (-1,))).squeeze()

    def get_Ti(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ion temperature

        :param t: time [s]
        :return: Ti [eV]
        """
        return self.interp1d(self.time, self.shot['ti'], torch.reshape(t, (-1,))).squeeze()

    def get_Tw(self, t: torch.Tensor) -> torch.Tensor:
        """
        Wall temperature (350 K)

        :param t: time [s]
        :return: Tw [eV]
        """
        return torch.ones_like(t) * J2eV(k * 350)

    def get_a(self, t: torch.Tensor) -> torch.Tensor:
        """
        Plasma minor radius

        :param t: time [s]
        :return: a [m]
        """
        return self.interp1d(self.time, self.shot['aminor'], torch.reshape(t, (-1,))).squeeze()

    def get_R0(self, t: torch.Tensor) -> torch.Tensor:
        """
        Plasma major radius

        :param t: time [s]
        :return: R0 [m]
        """
        return self.interp1d(self.time, self.shot['r0'], torch.reshape(t, (-1,))).squeeze()

    def get_kappa0(self, t: torch.Tensor) -> torch.Tensor:
        """
        Elongation at the magnetic axis

        :param t: time [s]
        :return: kappa0 []
        """
        return self.interp1d(self.time, self.shot['kappa0'], torch.reshape(t, (-1,))).squeeze()

    def get_kappa(self, t: torch.Tensor) -> torch.Tensor:
        """
        Elongation at the plasma boundary

        :param t: time [s]
        :return: kappa0 []
        """
        return self.interp1d(self.time, self.shot['kappa'], torch.reshape(t, (-1,))).squeeze()

    def get_kappa95(self, t: torch.Tensor) -> torch.Tensor:
        """
        Elongation at the 95% flux surface

        :param t: time [s]
        :return: kappa95 []
        """
        return self.interp1d(self.time, self.shot['kappa95'], torch.reshape(t, (-1,))).squeeze()

    def get_vol(self, t: torch.Tensor) -> torch.Tensor:
        """
        Plasma volume

        :param t: time [s]
        :return: volume [m^3]
        """
        return self.interp1d(self.time, self.shot['volume'], torch.reshape(t, (-1,))).squeeze()

    def get_B0(self, t: torch.Tensor) -> torch.Tensor:
        """
        Toroidal field at magnetic axis

        :param t: time [s]
        :return: BT0 [T]
        """
        return self.interp1d(self.time, self.shot['bt0'], torch.reshape(t, (-1,))).squeeze()

    def get_IP(self, t: torch.Tensor) -> torch.Tensor:
        """
        Plasma current

        :param t: time [s]
        :return: IP [A]
        """
        return self.interp1d(self.time, self.shot['ip'], torch.reshape(t, (-1,))).squeeze()

    def get_q0(self, t: torch.Tensor) -> torch.Tensor:
        """
        Safety factor at magnetic axis

        :param t: time [s]
        :return: q0 []
        """
        return self.interp1d(self.time, self.shot['q0'], torch.reshape(t, (-1,))).squeeze()

    def get_q95(self, t: torch.Tensor) -> torch.Tensor:
        """
        Safety factor at 95% flux surface

        :param t: time [s]
        :return: q95 []
        """
        return self.interp1d(self.time, self.shot['q95'], torch.reshape(t, (-1,))).squeeze()

    def get_gas(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calibrated gas flow

        :param t: time [s]
        :return: GAS  [Torr L s^-1 m^-3]
        """
        return self.interp1d(self.time, self.shot['gas_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_Si_gas(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ion gas puffing

        :param t: time [s]
        :return: Si_gas [m^-3 s^-1]
        """
        gas = self.get_gas(t)
        C_gas = self.C_gas  # tuned parameters
        Si_gas = (gas * torr * 1E-3) * N_A / (R_gas * C_gas)
        return Si_gas

    def get_Si_ext(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ion external particle source

        :param t: time [s]
        :return: Si [m^-3 s^-1]
        """
        E_b0 = self.get_E_b0()
        P_nbi = self.get_P_nbi(t)
        # C_rec = 0.904
        # tau_pi = self.get_taup(ni, ne, t)

        Si_nbi = P_nbi / eV2J(E_b0)
        Si_gas = self.get_Si_gas(t)
        # S_i_rec = C_rec * ni / tau_pi

        Si = Si_nbi + Si_gas
        return Si

    def get_Sa(self):
        """
        Alpha particle source, assumed to be 0

        :return: Sa [m^-3 s^-1]
        """
        Sa = 0
        return Sa

    def get_Sz(self):
        """
        Impurity particle source, assumed to be 0

        :return: Sz [m^-3 s^-1]
        """
        Sz = 0
        return Sz

    def get_sigma_v_fus(self, Ti):
        """
        Fusion reactivity

        For D-D fusion, this will be an average of D + D -> T + p and D + D -> He + n.

        :param Ti: ion temperature [eV]
        :return: sigma_v_fus [m^3/s]
        """
        sigma_v1 = self.reaction.get_fusion_coefficient(Ti, 'ddpt')
        sigma_v2 = self.reaction.get_fusion_coefficient(Ti, 'ddnh')
        sigma_v_fus = (sigma_v1 + sigma_v2) / 2
        return sigma_v_fus

    def get_U_fe(self, Te):
        """
        Fusion energy deposition to electrons

        :param Te: electron temperature [eV]
        :return: U_fe [J]
        """
        U1 = self.reaction.get_fusion_energy('ddpt')
        nbi2i_1 = self.get_nbi2i(Te, m_d, m_t, U1)
        U2 = self.reaction.get_fusion_energy('ddnh')
        nbi2i_2 = self.get_nbi2i(Te, m_d, m_he, U2)
        return eV2J(U1 * (1 - nbi2i_1) + U2 * (1 - nbi2i_2)) / 2

    def get_U_fi(self, Te):
        """
        Fusion energy deposition to ions

        :param Te: electron temperature
        :return: U_fi [J]
        """
        U1 = self.reaction.get_fusion_energy('ddpt')
        nbi2i_1 = self.get_nbi2i(Te, m_d, m_t, U1)
        U2 = self.reaction.get_fusion_energy('ddnh')
        nbi2i_2 = self.get_nbi2i(Te, m_d, m_he, U2)
        return eV2J(U1 * nbi2i_1 + U2 * nbi2i_2) / 2

    def get_Si_fus(self, ni, Ti):
        """
        Ion fusion particle source

        :param ni: ion density [m^-3]
        :param Ti: ion temperature [eV]
        :return: Si_fus [m^-3 s^-1]
        """
        sigma_v_fus = self.get_sigma_v_fus(Ti)
        return 2 * ni ** 2 * sigma_v_fus

    def get_x(self, ne, t: torch.Tensor):
        """
        Input for the confinement time model

        :param ne: electron density [m^-3]
        :param t: time [s]
        :return: x = [IP, B0, n, P_tot, R0, kappa, A, M]
        """
        # I (MA) is the plasma current
        # B (T) is the toroidal magnetic field
        # n (10^19 m^-3) is the central line averaged electron density
        # P (MW) is the absorbed power
        # R (m) is the major radius
        # kappa is the elongation
        # epsilon is the inverse aspect ratio (R0/a)^-1
        # Scr is the cross sectional area
        # M is the hydrogen isotope mass (the atomic mass in amu)

        IP = self.get_IP(t) / 1E6
        B0 = self.get_B0(t)
        n = ne / 1E19
        P_tot = self.get_P_tot(t) / 1E6
        R0 = self.get_R0(t)
        kappa = self.get_kappa0(t)
        a = self.get_a(t)
        A = R0 / a
        M = 2 * torch.ones(t.shape, dtype=torch.double, device=self.device)  # for D

        x = torch.stack([IP, B0, n, P_tot, R0, kappa, A, M], dim=t.dim())
        return x

    def get_tau_e98(self, x: torch.Tensor) -> torch.Tensor:
        """
        ITER-98 energy confinement time

        :param x: input for the confinement time model
        :return: tau_e98 [s]
        """
        # ITER H-98 P(y, 2) Scaling
        # Cx10^3 | I    | B    | n    | P     | R    | kappa | epsilon | S | M
        # 56.2   | 0.93 | 0.15 | 0.41 | -0.69 | 1.97 | 0.78  | 0.58    | - | 0.19
        IP, B0, n, P_tot, R0, kappa, A, M = x.split(1, dim=-1)
        tau_e98 = 0.0562 * IP ** 0.93 * B0 ** 0.15 * n ** 0.41 \
                  * P_tot ** (-0.69) * R0 ** 1.97 * kappa ** 0.78 * A ** (-0.58) * M ** 0.19
        return tau_e98.squeeze()

    def get_tau(self, x: torch.Tensor) -> torch.Tensor:
        """
        Confinement times

        :param x: input for the confinement time model
        :return: [tau_pi, tau_ei, tau_ee] [s]
        """
        if self.tuned:
            taus = torch.exp(self.net(torch.log(x)))
        else:
            tau_e98 = self.get_tau_e98(x)
            taus = torch.stack([tau_e98] * self.num_outputs, dim=tau_e98.dim())

        return taus

    def get_P_tot(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total input power, including NB, Ohmic, ECH, and ICH

        :param t: time [s]
        :return: P_tot [W]
        """
        return self.interp1d(self.time, self.shot['ptot'], torch.reshape(t, (-1,))).squeeze()

    def get_P_oh(self, t: torch.Tensor) -> torch.Tensor:
        """
        Ohmic power

        :param t: time [s]
        :return: P_oh [W/m^3]
        """
        return self.interp1d(self.time, self.shot['poh_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_P_nbi(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total injected neutral beam power

        :param t: time [s]
        :return: P_nbi [W/m^3]
        """
        return self.interp1d(self.time, self.shot['pnbi_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_P_ech(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total ECH power

        :param t: time [s]
        :return: P_ech [W/m^3]
        """
        if 'echpwrc' not in self.shot:
            return torch.zeros_like(t)
        return self.interp1d(self.time, self.shot['echpwrc_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_P_ich(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total ICH power

        :param t: time [s]
        :return: P_ich [W/m^3]
        """
        if 'ichpwrc' not in self.shot:
            return torch.zeros_like(t)
        return self.interp1d(self.time, self.shot['ichpwrc_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_P_lh(self, t: torch.Tensor) -> torch.Tensor:
        """
        Total LH power

        :param t: time [s]
        :return: P_lh [W/m^3]
        """
        if 'plh' not in self.shot:
            return torch.zeros_like(t)
        return self.interp1d(self.time, self.shot['plh_vol'], torch.reshape(t, (-1,))).squeeze()

    def get_E_b0(self):
        """
        Initial beam energy, assumed to be 80keV

        :return: E_b0 [eV]
        """
        E_b0 = 80E3
        return E_b0

    def get_nbi2i(self, Te, m_i, m_b, E_b0):
        """
        Fraction of NBI power to ion

        :param Te: electron temperature [eV]
        :param m_i: ion mass [kg]
        :param m_b: beam particle mass [kg]
        :param E_b0: initial beam energy [eV]
        :return: fraction to ion
        """
        E_c = (3 * pi ** 0.5 / 4) ** (2 / 3) * (m_i / m_e) ** (1 / 3) * (m_b / m_i) * Te  # [eV]
        phi = lambda x: 1 / x * (1 / 3 * log((1 - x ** 0.5 + x) / (1 + x ** 0.5) ** 2)
                                 + 2 / 3 ** 0.5 * (arctan((2 * x ** 0.5 - 1) / 3 ** 0.5) + pi / 6))
        nbi2i = phi(E_b0 / E_c)
        return nbi2i

    def get_Pi_aux(self, Te, t):
        """
        Auxiliary ion heating power

        :param Te: electron temperature [eV]
        :param t: time [s]
        :return: Pi_aux [W/m^3]
        """
        E_b0 = self.get_E_b0()
        nbi2i = self.get_nbi2i(Te, m_d, m_d, E_b0)
        P_nbi = self.get_P_nbi(t)
        P_ich = self.get_P_ich(t)
        Pi_aux = nbi2i * P_nbi + P_ich
        return Pi_aux

    def get_Pe_aux(self, Te, t):
        """
        Auxiliary electron heating power

        :param Te: electron temperature [eV]
        :param t: time [s]
        :return: Pe_aux [W/m^3]
        """
        E_b0 = self.get_E_b0()
        nbi2i = self.get_nbi2i(Te, m_d, m_d, E_b0)
        P_nbi = self.get_P_nbi(t)
        P_ech = self.get_P_ech(t)
        Pe_aux = (1 - nbi2i) * P_nbi + P_ech
        return Pe_aux

    def get_coulomb_log(self, n1, n2, T1, T2, z1: int = 1, z2: int = 1):
        """
        Coulomb logarithm

        :param n1: density [m^-3]
        :param n2: density [m^-3]
        :param T1: temperature [eV]
        :param T2: temperature [eV]
        :param z1: atomic number []
        :param z2: atomic number []
        :return: ln(Lambda) []
        """
        T = (n1 * T1 + n2 * T2) / (n1 + n2)
        q1 = z1 * e
        q2 = z2 * e
        coulomb_log = log(12 * pi * sqrt((epsilon_0 * eV2J(T)) ** 3 / (n2 * q2 ** 4 * q1 ** 2)))
        return coulomb_log

    def get_Q_ie(self, ni, ne, Ti, Te, mi=m_d, me=m_e, zi: int = 1, ze: int = 1):
        """
        Ion-electron collisional energy transfer

        :param ni: ion density [m^-3]
        :param ne: electron density [m^-3]
        :param Ti: ion temperature [eV]
        :param Te: electron temperature [eV]
        :param mi: ion mass [kg]
        :param me: electron mass [kg]
        :param zi: atomic number []
        :param ze: atomic number []
        :return: Q_ie [W/m^3]
        """
        qi = zi * e
        qe = ze * e
        coulomb_log = self.get_coulomb_log(ni, ne, Ti, Te)
        Q_ie = (ni * ne * qi ** 2 * qe ** 2 * me * coulomb_log * (1 - Ti / Te)) / (
                2 * pi * epsilon_0 ** 2 * sqrt(2 * pi * me * eV2J(Te)) * mi
                * (1 + 4 * np.sqrt(pi) / 3 * ((3 * me * Ti) / (2 * mi * Te)) ** 1.5)
        )
        return Q_ie

    def get_K(self, alpha_n, alpha_T, beta_T):
        """
        K in ECR formula

        :param alpha_n: alpha_n in n_e = n_{e_0} * (1 - rho^2)^{alpha_n}
        :param alpha_T: alpha_T in (T_{e_0} - T_{e_a}) * (1 - rho^{beta_T})^{alpha_T} + T_{e_a}
        :param beta_T: beta_T in (T_{e_0} - T_{e_a}) * (1 - rho^{beta_T})^{alpha_T} + T_{e_a}
        :return: K
        """
        K = (alpha_n + 3.87 * alpha_T + 1.46) ** (-0.79) * (1.98 + alpha_T) ** 1.36 * beta_T ** 2.14 \
            * (beta_T ** 1.53 + 1.87 * alpha_T - 0.16) ** (-1.33)
        return K

    def get_G(self, A):
        """
        G in ECR formula

        :param A: aspect ratio
        :return: G
        """
        G = 0.93 * (1 + 0.85 * exp(-0.82 * A))
        return G

    def get_P_ecr(self, ne0, Te0, a, R, Bt, kappa, V, **kwargs):
        """
        Electron Cyclotron Radiation (ECR)

        :param ne0: central electron density [m^-3]
        :param Te0: central temperature [eV]
        :param a: minor radius [m]
        :param R: major radius [m]
        :param Bt: magnetic field [T]
        :param kappa: elongation []
        :param V: volume [m^3]
        :key alpha_n: alpha_n in the electron density profile []
        :key alpha_T: alpha_T in the electron temperature profile []
        :key beta_T: beta_T in the electron temperature profile []
        :key r: reflection coefficient []
        :return: P_ecr [W/m^3]
        """
        # ITER parameters
        if kwargs:
            alpha_n = kwargs['alpha_ne']
            alpha_T = kwargs['alpha_te']
            beta_T = kwargs['beta_te']
            r = kwargs['r']
        else:
            alpha_n = 0.5
            alpha_T = 8.0
            beta_T = 5.0
            r = 0.8

        A = R / a
        K = self.get_K(alpha_n, alpha_T, beta_T)
        G = self.get_G(A)

        ne0_20 = ne0 / 1E20
        Te0_keV = Te0 / 1E3
        pa0 = 6.04E3 * a * ne0_20 / Bt

        P_ecr = 3.84E-2 * (1 - r) ** 0.5 * R * a ** 1.38 * kappa ** 0.79 * Bt ** 2.62 * ne0_20 ** 0.38 * Te0_keV \
                * (16 + Te0_keV) ** 2.61 * (1 + 0.12 * Te0_keV / pa0 ** 0.41) ** (-1.51) * K * G
        return P_ecr / V

    def get_P_brem(self, ni, ne, Te, Z_eff):
        """
        Bremsstrahlung radiation

        :param ni: ion density [m^-3]
        :param ne: electron density [m^-3]
        :param Te: electron temperature [eV]
        :param Z_eff: effective atomic number []
        :return: P_brem [W/m^3]
        """
        P_brem = 1.7E-38 * Z_eff ** 2 * ni * ne * sqrt(Te / 1E3)
        return P_brem

    def get_P_imp(self, ne, nz, Te, Zz):
        """
        Impurity radiation

        :param ne: electron density [m^-3]
        :param nz: impurity density [m^-3]
        :param Te: electron temperature [eV]
        :param Zz: impurity charge []
        :return: P_imp [W/m^3]
        """
        if isinstance(Zz, Iterable):
            P_imp = torch.zeros_like(Te)
            for _nz, _Zz in zip(nz, Zz):
                P_imp += (1 + 0.3 * (Te / 1E3)) * 1E-37 * ne * _nz * _Zz ** (3.7 - 0.33 * log(Te / 1E3))
        else:
            P_imp = (1 + 0.3 * (Te / 1E3)) * 1E-37 * ne * nz * Zz ** (3.7 - 0.33 * log(Te / 1E3))
        return P_imp

    def get_P_rad(self, ni, ne, nz, Te, a, R0, B0, kappa, vol, Zz=None, Z_eff=None, ecr: bool = True):
        """
        Radiative Power Loss

        :param ni: ion density [m^-3]
        :param ne: electron density [m^-3]
        :param Te: electron temperature [eV]
        :param nz: impurity density [m^-3]
        :param a: minor radius [m]
        :param R0: major radius [m]
        :param B0: magnetic field [T]
        :param kappa: elongation []
        :param vol: volume [m^3]
        :param Zz: impurity charge []
        :param Z_eff: effective atomic number []
        :param ecr: true to include ECR
        :return: P_rad [W/m^3]
        """
        if Zz is None:
            Zz = self.impurity_charge
        if Z_eff is None:
            Z_eff = sqrt((ni + nz * Zz ** 2) / ni)

        P_brem = self.get_P_brem(ni, ne, Te, Z_eff)
        P_imp = self.get_P_imp(ne, nz, Te, Zz)
        if ecr:
            P_ecr = self.get_P_ecr(ne, Te, a, R0, B0, kappa, vol)
        else:
            P_ecr = 0
        P_rad = P_brem + P_imp + P_ecr

        return P_rad

    def get_Pi_fus(self, ni, Ti, Te):
        """
        Fusion power to ions

        :param ni: ion density [m^-3]
        :param Ti: ion temperature [eV]
        :param Te: electron temperature [eV]
        :return: Pi_fus [W/m^3]
        """
        sigma_v = self.get_sigma_v_fus(Ti)
        U_fi = self.get_U_fi(Te)
        return ni ** 2 * sigma_v * U_fi

    def get_Pe_fus(self, ni, Ti, Te):
        """
        Fusion power to electrons

        :param ni: ion density [m^-3]
        :param Ti: ion temperature [eV]
        :param Te: electron temperature [eV]
        :return: Pe_fus [W/m^3]
        """
        sigma_v = self.get_sigma_v_fus(Ti)
        U_fe = self.get_U_fe(Te)
        return ni ** 2 * sigma_v * U_fe

    def get_sources(self, ni: torch.Tensor, Ui: torch.Tensor, Ue: torch.Tensor, t: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Particle and energy source terms

        :param ni: ion density [m^-3]
        :param Ui: ion energy [J]
        :param Ue: electron energy [J]
        :param t: time [s]
        :return: Si, Pi, Pe
        """
        nc = self.get_nc(t)
        ne = ni + nc * self.impurity_charge
        Ti = J2eV(Ui / ni * 2 / 3)
        Te = J2eV(Ue / ne * 2 / 3)

        x = self.get_x(ne, t)
        taus = self.get_tau(x)
        tau_pi, tau_ei, tau_ee = tuple(map(torch.squeeze, taus.split(1, dim=-1)))
        a = self.get_a(t)
        R0 = self.get_R0(t)
        B0 = self.get_B0(t)
        kappa = self.get_kappa0(t)
        vol = self.get_vol(t)

        Si_ext = self.get_Si_ext(t)
        Si_fus = self.get_Si_fus(ni, Ti)
        Si_dif = ni / tau_pi
        Si = (Si_ext, Si_fus, Si_dif)

        Pi_aux = self.get_Pi_aux(Te, t)
        Pi_fus = self.get_Pi_fus(ni, Ti, Te)
        Q_ie = self.get_Q_ie(ni, ne, Ti, Te)
        Pi_dif = 3 / 2 * ni * eV2J(Ti) / tau_ei
        Pi = (Pi_aux, Pi_fus, Q_ie, Pi_dif)

        P_oh = self.get_P_oh(t)
        Pe_aux = self.get_Pe_aux(Te, t)
        Pe_fus = self.get_Pe_fus(ni, Ti, Te)
        P_rad = self.get_P_rad(ni, ne, nc, Te, a, R0, B0, kappa, vol)
        Pe_dif = 3 / 2 * ne * eV2J(Te) / tau_ee
        Pe = (P_oh, Pe_aux, Pe_fus, P_rad, Pe_dif)

        return Si, Pi, Pe

    def system(self, t, y):
        """
        Dynamical system for the single-node model

        :param t: time [s]
        :param y: ni [10^19 m^-3], Ui, Ue [10^19 keV/m^3]
        :return: dy/dt
        """
        self.system_count += 1

        y = torch.abs(y)
        ni = y[0] * 1E19  # [10^19 m^-3] -> [m^-3]
        Ui = keV2J(y[1] * 1E19)  # [10^19 keV] -> [J]
        Ue = keV2J(y[2] * 1E19)  # [10^19 keV] -> [J]

        Si, Pi, Pe = self.get_sources(ni, Ui, Ue, t)
        Si_ext, Si_fus, Si_dif = Si
        Pi_aux, Pi_fus, Q_ie, Pi_dif = Pi
        P_oh, Pe_aux, Pe_fus, P_rad, Pe_dif = Pe

        Si = Si_ext - Si_fus - Si_dif
        Pi = Pi_aux + Pi_fus + Q_ie - Pi_dif
        Pe = P_oh + Pe_aux + Pe_fus - Q_ie - P_rad - Pe_dif

        dy_dt = torch.zeros_like(y)
        dy_dt[0] = Si / 1E19  # [m^-3] -> [10^19 m^-3]
        dy_dt[1] = J2keV(Pi) / 1E19  # [J] -> [10^19 keV]
        dy_dt[2] = J2keV(Pe) / 1E19  # [J] -> [10^19 keV]

        return dy_dt


class Reactor1D(Reactor0D):
    """
    One-Dimension Fusion Reactor
    """

    def __init__(self, shot_num: int = None, tuned: bool = False, net: nn.Module = None):
        """
        Initialize the 1D reactor

        :param shot_num: shot number
        :param tuned: true for tuned transport times
        :param net: the neural network for the diffusivity model
        """
        self.dim = 1
        super().__init__(shot_num=shot_num, dim=self.dim, tuned=tuned, net=net)

        self.nodes = config.nodes
        self.rho_core = config.rho_core
        self.rho_edge = config.rho_edge
        self.rho_sol = config.rho_sol
        self.rhos = config.rhos
        self.drho_core_edge = (self.rho_edge - 0) / 2
        self.drho_edge_sol = (self.rho_sol - self.rho_core) / 2
        self.drho_sol_div = (self.rho_sol - self.rho_edge) / 2

        if self.time is not None:
            self.iol = IOL(R0=self.get_average(self.get_R0(self.time)), r0=self.get_average(self.get_a(self.time)),
                           mi=m_d, B0=self.get_average(self.get_B0(self.time)),
                           IP=self.get_average(self.get_IP(self.time)))
        else:
            self.iol = None
        self.heater = Heater()

        self.M = A_d
        self.epsilon_coefficient = None
        self.n0_sol = 1E15  # [m^-3]
        self.T0c_sol = 1  # [eV]

    def get_average(self, x: torch.Tensor) -> float:
        """
        Average

        :param x: tensor
        :return: scalar
        """
        return x.mean().detach().cpu().numpy().item()

    def get_ne_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Electron density in one node

        :param t: time [s]
        :param node: node name
        :return: ne_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['ne_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_nd_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Deuteron density in one node

        :param t: time [s]
        :param node: node name
        :return: nd_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.get_ne_node(t, node=node) - self.impurity_charge * self.get_nc_node(t, node=node)

    def get_nc_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Impurity density in one node

        :param t: time [s]
        :param node: node name
        :return: nc_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['nc_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_n0_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Neutral hydrogen density in one node

        :param t: time [s]
        :param node: node name
        :return: n0_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.get_n0(t)

    def get_Te_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Electron temperature in one node

        :param t: time [s]
        :param node: node name
        :return: Te_node [eV]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['te_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_Td_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Deuteron temperature in one node

        :param t: time [s]
        :param node: node name
        :return: Td_node [eV]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['ti_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_vol_node(self, vol: torch.Tensor, node: str) -> torch.Tensor:
        """
        Plasma volume in one node

        :param vol: volume [m^3]
        :param node: node name
        :return: V_node [m^3]
        """
        node = node.lower()
        assert node in self.nodes
        vol_core = vol * (self.rho_core / self.rho_edge) ** 2

        if node == 'core':
            return vol_core
        elif node == 'edge':
            return vol - vol_core
        else:  # node == 'sol'
            vol_sol = vol * ((self.rho_sol ** 2 - self.rho_edge ** 2) / (self.rho_edge ** 2))
            return vol_sol

    def get_Sd_ext_node(self, t: torch.Tensor):
        """
        Deuteron external particle source for nodes

        :param t: time [s]
        :return: Sd_ext_core, Sd_ext_edge [m^-3 s^-1]
        """
        E_b0 = self.get_E_b0()
        P_nbi = self.get_P_nbi(t)

        Si_nbi = P_nbi / eV2J(E_b0)
        Si_gas = self.get_Si_gas(t)

        Sd_ext_core = self.heater.f_nbi_core * Si_nbi + self.heater.f_gas_core * Si_gas
        Sd_ext_edge = self.heater.f_nbi_edge * Si_nbi + self.heater.f_gas_edge * Si_gas
        return Sd_ext_core, Sd_ext_edge

    def get_Sd_fus_node(self, nd_node, sigma_v_fus_node):
        """
        Deuteron fusion particle source in one node

        :param nd_node: deuteron density [m^-3]
        :param sigma_v_fus_node: fusion reaction rate coefficient [m^3/s]
        :return: Sd_fus_node [m^-3 s^-1]
        """
        Sd_fus_node = - 2 * nd_node ** 2 * sigma_v_fus_node
        return Sd_fus_node

    def get_Sd_ion_node(self, n0_node, ne_node, sigma_v_ion_node):
        """
        Deuteron ionization particle source in one node

        :param n0_node: neutral particle density [m^-3]
        :param ne_node: electron particle density [m^-3]
        :param sigma_v_ion_node: ionization reaction rate coefficient [m^3/s]
        :return: Sd_ion_node [m^-3 s^-1]
        """
        Sd_ion_node = n0_node * ne_node * sigma_v_ion_node
        return Sd_ion_node

    def get_Sd_rec_node(self, nd_node, ne_node, sigma_v_rec_node):
        """
        Deuteron recombination particle source in one node

        :param nd_node: deuteron density [m^-3]
        :param ne_node: electron particle density [m^-3]
        :param sigma_v_rec_node: recombination reaction rate coefficient [m^3/s]
        :return: Sd_rec_node [m^-3 s^-1]
        """
        Sd_rec_node = - nd_node * ne_node * sigma_v_rec_node
        return Sd_rec_node

    def get_P_oh_node(self, t: torch.Tensor):
        """
        Ohmic power in one node

        :param t: time [s]
        :return: P_oh_node [W/m^3]
        """
        return self.get_P_oh(t)

    def get_Ps_aux_node(self, Te_core, Te_edge, t):
        """
        Auxiliary heating power for nodes

        :param Te_core: core electron temperature [eV]
        :param Te_edge: edge electron temperature [eV]
        :param t: time [s]
        :return: Pd_aux_core, Pd_aux_edge, Pe_aux_core, Pe_aux_edge [W/m^3]
        """
        E_b0 = self.get_E_b0()
        nbi2i_core = self.get_nbi2i(Te_core, m_d, m_d, E_b0)
        nbi2i_edge = self.get_nbi2i(Te_edge, m_d, m_d, E_b0)
        nbi2e_core = 1 - nbi2i_core
        nbi2e_edge = 1 - nbi2i_edge

        P_nbi = self.get_P_nbi(t)
        P_ich = self.get_P_ich(t)
        P_ech = self.get_P_ech(t)

        Pi_nbi_core = nbi2i_core * self.heater.f_nbi_core * P_nbi
        Pi_nbi_edge = nbi2i_edge * self.heater.f_nbi_edge * P_nbi
        Pe_nbi_core = nbi2e_core * self.heater.f_nbi_core * P_nbi
        Pe_nbi_edge = nbi2e_edge * self.heater.f_nbi_edge * P_nbi

        Pi_ich_core = self.heater.f_ch_core * P_ich
        Pi_ich_edge = self.heater.f_ch_edge * P_ich
        Pe_ech_core = self.heater.f_ch_core * P_ech
        Pe_ech_edge = self.heater.f_ch_edge * P_ech

        Pd_aux_core = Pi_nbi_core + Pi_ich_core
        Pd_aux_edge = Pi_nbi_edge + Pi_ich_edge
        Pe_aux_core = Pe_nbi_core + Pe_ech_core
        Pe_aux_edge = Pe_nbi_edge + Pe_ech_edge

        return Pd_aux_core, Pd_aux_edge, Pe_aux_core, Pe_aux_edge

    def get_Ps_fus_node(self, nd_node, Te_node, sigma_v_fus_node):
        """
        Fusion power in one node

        :param nd_node: deuteron density [m^-3]
        :param Te_node: electron temperature [eV]
        :param sigma_v_fus_node: fusion reaction rate coefficient [m^3/s]
        :return: Pd_fus_node, Pe_fus_node [W/m^3]
        """
        U1 = self.reaction.get_fusion_energy('ddpt')
        nbi2i_1 = self.get_nbi2i(Te_node, m_d, m_t, U1)
        U2 = self.reaction.get_fusion_energy('ddnh')
        nbi2i_2 = self.get_nbi2i(Te_node, m_d, m_he, U2)

        U_fd = eV2J(U1 * nbi2i_1 + U2 * nbi2i_2) / 2
        U_fe = eV2J(U1 * (1 - nbi2i_1) + U2 * (1 - nbi2i_2)) / 2
        Pd_fus_node = nd_node ** 2 * sigma_v_fus_node * U_fd
        Pe_fus_node = nd_node ** 2 * sigma_v_fus_node * U_fe

        return Pd_fus_node, Pe_fus_node

    def get_Pe_ion_node(self, n0_node, ne_node, sigma_v_ion_node):
        """
        Ionization power in one node

        :param n0_node: neutral particle density [m^-3]
        :param ne_node: electron particle density [m^-3]
        :param sigma_v_ion_node: ionization reaction rate coefficient [m^3/s]
        :return: Pe_ion_node [W/m^3]
        """
        E_ion_node = self.reaction.get_ionization_energy()
        Pe_ion_node = - eV2J(E_ion_node) * n0_node * ne_node * sigma_v_ion_node
        return Pe_ion_node

    def get_Pe_rec_node(self, nd_node, ne_node, sigma_v_rec_node):
        """
        Recombination power in one node

        :param nd_node: deuteron density [m^-3]
        :param ne_node: electron particle density [m^-3]
        :param sigma_v_rec_node: ionization reaction rate coefficient [m^3/s]
        :return: Pe_rec_node [W/m^3]
        """
        E_rec_node = self.reaction.get_recombination_energy()
        Pe_rec_node = eV2J(E_rec_node) * nd_node * ne_node * sigma_v_rec_node
        return Pe_rec_node

    def get_RN(self, Ti):
        """
        Particle reflection coefficient

        :param Ti: ion temperature [eV]
        :return: RN []
        """
        E0 = 1.5 * Ti / 1E3  # [eV] -> [keV]

        if self.epsilon_coefficient is None:
            Z1, Z2 = 1, 6
            M1, M2 = 2, 12
            mu = M2 / M1
            self.epsilon_coefficient = 32.55 * (mu / (1 + mu)) \
                                       * (1 / (Z1 * Z2 * (Z1 ** (2 / 3) + Z2 ** (2 / 3)) ** 0.5))

        epsilon = self.epsilon_coefficient * E0
        A1, A2, A3, A4, A5, A6 = 0.5173, 2.549, 5.325, 0.5719, 1.094, 1.933
        RN = A1 * log(A2 * epsilon + np.e) / (1 + A3 * epsilon ** A4 + A5 * epsilon ** A6)
        return RN

    def get_Pd_at_node(self, nd_node, Td_node, sigma_v_cx_node, sigma_v_el_node):
        """
        Atomic process power in one node for the charge exchange and elastic scattering

        :param nd_node: deuteron density [m^-3]
        :param Td_node: deuteron temperature [eV]
        :param sigma_v_cx_node: charge exchange reaction rate coefficient [m^3/s]
        :param sigma_v_el_node: elastic scattering reaction rate coefficient [m^3/s]
        :return: Pd_at_node [W/m^3]
        """
        RN_node = self.get_RN(Td_node)
        n0c_node = RN_node * self.n0_sol
        T0c_node = self.T0c_sol
        Pd_at_node = - 1.5 * eV2J(Td_node - T0c_node) * nd_node * n0c_node * (sigma_v_cx_node + sigma_v_el_node)
        return Pd_at_node

    def get_Q_de_node(self, nd_node, ne_node, Td_node, Te_node):
        """
        Deuteron-electron collisional energy transfer in one node

        :param nd_node: deuteron density [m^-3]
        :param ne_node: electron density [m^-3]
        :param Td_node: deuteron temperature [eV]
        :param Te_node: electron temperature [eV]
        :return: Q_de_node [W/m^3]
        """
        return self.get_Q_ie(nd_node, ne_node, Td_node, Te_node)

    def get_P_rad_node(self, nd_node, ne_node, nz_node, Te_node, a, R0, B0, kappa, vol, Zz=None, Z_eff=None,
                       ecr: bool = True):
        """
        Radiative Power Loss in one node

        :param nd_node: deuteron density [m^-3]
        :param ne_node: electron density [m^-3]
        :param Te_node: electron temperature [eV]
        :param nz_node: impurity density [m^-3]
        :param a: minor radius [m]
        :param R0: major radius [m]
        :param B0: magnetic field [T]
        :param kappa: elongation []
        :param vol: volume [m^3]
        :param Zz: impurity charge []
        :param Z_eff: effective atomic number []
        :param ecr: true to include ECR
        :return: P_rad_node [W/m^3]
        """
        return self.get_P_rad(nd_node, ne_node, nz_node, Te_node, a, R0, B0, kappa, vol, Zz, Z_eff, ecr)

    def get_x_node(self, ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t: torch.Tensor,
                   node: str):
        """
        Input for the diffusivity model
        [Bt, ne, Te, grad_Te, q, kappa, M, R, a]

        :param ne_core: core electron density [m^-3]
        :param ne_edge: edge electron density [m^-3]
        :param ne_sol: SOL electron density [m^-3]
        :param Te_core: core electron temperature [eV]
        :param Te_edge: edge electron temperature [eV]
        :param Te_sol: SOL electron temperature [eV]
        :param a: minor radius [m]
        :param R0: major radius [m]
        :param kappa: elongation []
        :param B0: magnetic field [T]
        :param t: time [s]
        :param node: node name
        :return: x_node
        """
        assert node.lower() in self.nodes

        R = R0  # [m]
        Bt = B0  # [T]
        q = self.get_q95(t)
        M = torch.squeeze(self.M * torch.ones_like(t))

        if node == 'core':
            ne = ne_core  # [m^-3]
            Te = Te_core  # [eV]
            dr_core_edge = self.drho_core_edge * a
            grad_Te = torch.abs((Te_core - Te_edge) / dr_core_edge)  # [eV/m]
        elif node == 'edge':
            ne = ne_edge  # [m^-3]
            Te = Te_edge  # [eV]
            dr_edge_sol = self.drho_edge_sol * a
            grad_Te = torch.abs(Te_edge / dr_edge_sol)  # [eV/m]
        else:  # node == 'sol'
            ne = ne_sol  # [m^-3]
            Te = Te_sol  # [eV]
            dr_sol_div = self.drho_sol_div * a
            grad_Te = torch.abs(Te_sol / dr_sol_div)  # [eV/m]

        x_node = torch.stack([Bt, ne / 1E19, Te / 1E3, grad_Te / 1E3, q, kappa, M, R, a], dim=t.squeeze().dim())
        return x_node

    def get_chi_H98(self, x_node: torch.Tensor):
        """
        Heat diffusivity (H98)

        :param x_node: input for the diffusivity model
        :return: chi_H98 [m^2/s]
        """
        alpha_H = 0.123
        Bt, ne, Te, grad_Te, q, kappa, M, R, a = x_node.split(1, dim=-1)
        chi_H98 = alpha_H * Bt ** (-3.5) * ne ** 0.9 * Te * grad_Te ** 1.2 * q ** 3.0 \
                  * kappa ** (-2.9) * M ** (-0.6) * R ** 0.7 * a ** (-0.2)
        return chi_H98.squeeze()

    def get_chi_node(self, x_node: torch.Tensor, node: str, **kwargs):
        """
        Nodal diffusivities
        [D_d_node, chi_d_node, chi_e_node]

        :param x_node: input for the diffusivity model
        :param node: node name
        :key alpha_particle: true for alpha particle diffusivities
        :return: chi_node [m^2/s]
        """
        if self.tuned:
            chi_node = torch.exp(self.net(torch.log(x_node), node=node))
        else:
            chi = self.get_chi_H98(x_node)
            if 'alpha_particle' in kwargs and kwargs['alpha_particle']:
                chi_node = torch.stack([0.6 * chi, 0.6 * chi, chi, chi, chi], dim=chi.dim())
            else:
                chi_node = torch.stack([0.6 * chi, chi, chi], dim=chi.dim())
        return chi_node

    def get_tau_node(self, chi_node: torch.Tensor, a: torch.Tensor, node: str):
        """
        Inter-nodal transport times

        :param chi_node: inter-nodal diffusivity vector [m^2/s]
        :param a: minor radius [m]
        :param node: node name
        :return: tau_node = [tau_pd_node_node', tau_ed_node_node', tau_ee_node_node'] [s]
        """
        if node == 'core':
            r_core = self.rho_core * a
            dr_core_edge = self.drho_core_edge * a
            geom_ratio = ((r_core ** 2) / (2 * r_core) * dr_core_edge).unsqueeze(-1)
        elif node == 'edge':
            r_core = self.rho_core * a
            r_edge = self.rho_edge * a
            dr_edge_sol = self.drho_edge_sol * a
            geom_ratio = ((r_edge ** 2 - r_core ** 2) / (2 * r_edge) * dr_edge_sol).unsqueeze(-1)
        else:  # node == 'sol'
            r_edge = self.rho_edge * a
            r_sol = self.rho_sol * a
            dr_sol_div = self.drho_sol_div * a
            geom_ratio = ((r_sol ** 2 - r_edge ** 2) / (2 * r_sol) * dr_sol_div).unsqueeze(-1)
        tau_node = geom_ratio / chi_node
        return tau_node

    def get_Ps_tran_node(self, Us_core, Us_edge, Us_sol, tau_es_core_edge, tau_es_edge_sol, tau_es_sol_div, vol,
                         nodes: list = config.nodes):
        """
        Nodal energy transport terms

        :param Us_core: core energy density [J/m^3]
        :param Us_edge: edge energy density [J/m^3]
        :param Us_sol: SOL energy density [J/m^3]
        :param tau_es_core_edge: energy transport time from the core to the edge [s]
        :param tau_es_edge_sol: energy transport time from the edge to the SOL [s]
        :param tau_es_sol_div: energy transport time from the SOL to the divertor [s]
        :param vol: volume [m^3]
        :param nodes: nodes
        :return: Ps_tran = [Ps_tran_core, Ps_tran_edge, Ps_tran_sol] [W/m^3]
        """
        V_core = self.get_vol_node(vol, node='core')
        V_edge = self.get_vol_node(vol, node='edge')

        Ps_tran = []
        if 'core' in nodes:
            Ps_tran_core = - (Us_core - Us_edge) / tau_es_core_edge
            Ps_tran.append(Ps_tran_core)
        if 'edge' in nodes:
            Ps_tran_edge = V_core / V_edge * (Us_core - Us_edge) / tau_es_core_edge - Us_edge / tau_es_edge_sol
            Ps_tran.append(Ps_tran_edge)
        if 'sol' in nodes:
            V_sol = self.get_vol_node(vol, node='sol')
            Ps_tran_sol = V_edge / V_sol * Us_edge / tau_es_edge_sol - Us_sol / tau_es_sol_div
            Ps_tran.append(Ps_tran_sol)

        if len(Ps_tran) == 1:
            return Ps_tran[0]
        else:
            return Ps_tran

    def get_Ps_iol_node(self, Us_node, Es_orb_node, tau_es_iol_node):
        """
        Nodal IOL term

        :param Us_node: nodal energy density [J/m^3]
        :param Es_orb_node: nodal energy loss fraction []
        :param tau_es_iol_node: energy timescale of IOL
        :return: Ps_iol_node [W/m^3]
        """
        Ps_iol_node = - Es_orb_node * Us_node / tau_es_iol_node
        return Ps_iol_node

    def get_Ps_iol_sol(self, Ps_iol_edge):
        """
        IOL term for the SOL node

        :param Ps_iol_edge: edge IOL term [W/m^3]
        :return: Ps_iol_sol [W/m^3]
        """
        V_edge_V_sol = (self.rho_edge ** 2 - self.rho_core ** 2) / (self.rho_sol ** 2 - self.rho_edge ** 2)
        Ps_iol_sol = - V_edge_V_sol * Ps_iol_edge
        return Ps_iol_sol

    def get_sources(self, nd: Tuple[torch.Tensor, ...], Ud: Tuple[torch.Tensor, ...], Ue: Tuple[torch.Tensor, ...],
                    t: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Particle and energy source terms

        :param nd: deuteron densities [m^-3]
        :param Ud: deuteron energies [J/m^3]
        :param Ue: electron energies [J/m^3]
        :param t: time [s]
        :return: Sd [m^-3 s^-1], Pd, Pe [W/m^3]
        """
        nd_core, nd_edge, nd_sol = nd
        Ud_core, Ud_edge, Ud_sol = Ud
        Ue_core, Ue_edge, Ue_sol = Ue

        nc_core = self.get_nc_node(t, node='core')
        nc_edge = self.get_nc_node(t, node='edge')
        nc_sol = self.get_nc_node(t, node='sol')
        n0_sol = self.get_n0_node(t, node='sol')
        ne_core = nd_core + nc_core * self.impurity_charge
        ne_edge = nd_edge + nc_edge * self.impurity_charge
        ne_sol = nd_sol + nc_sol * self.impurity_charge

        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Td_sol = J2eV(Ud_sol / nd_sol * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)
        Te_sol = J2eV(Ue_sol / ne_sol * 2 / 3)

        a = self.get_a(t)
        R0 = self.get_R0(t)
        B0 = self.get_B0(t)
        kappa = self.get_kappa0(t)
        vol = self.get_vol(t)

        sigma_v_ion_sol = self.reaction.get_ionization_coefficient(Te_sol)
        sigma_v_rec_sol = self.reaction.get_recombination_coefficient(Te_sol)
        sigma_v_cx_sol = self.reaction.get_charge_exchange_coefficient(Td_sol)
        sigma_v_el_sol = self.reaction.get_elastic_scattering_coefficient(Td_sol)

        x_core = self.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t, node='core')
        x_edge = self.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t, node='edge')
        x_sol = self.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t, node='sol')
        chi_core = self.get_chi_node(x_core, node='core')
        chi_edge = self.get_chi_node(x_edge, node='edge')
        chi_sol = self.get_chi_node(x_sol, node='sol')
        tau_core = self.get_tau_node(chi_core, a, node='core')
        tau_edge = self.get_tau_node(chi_edge, a, node='edge')
        tau_sol = self.get_tau_node(chi_sol, a, node='sol')
        tau_pd_core_edge, tau_ed_core_edge, tau_ee_core_edge = tuple(map(torch.squeeze, tau_core.split(1, dim=-1)))
        tau_pd_edge_sol, tau_ed_edge_sol, tau_ee_edge_sol = tuple(map(torch.squeeze, tau_edge.split(1, dim=-1)))
        tau_pd_sol_div, tau_ed_sol_div, tau_ee_sol_div = tuple(map(torch.squeeze, tau_sol.split(1, dim=-1)))
        Fd_orb_edge, Ed_orb_edge = self.iol.get_loss_fractions(Td_edge)

        Sd_ext_core, Sd_ext_edge = self.get_Sd_ext_node(t)
        Sd_ion_sol = self.get_Sd_ion_node(n0_sol, ne_sol, sigma_v_ion_sol)
        Sd_rec_sol = self.get_Sd_rec_node(nd_sol, ne_sol, sigma_v_rec_sol)
        Sd_tran_core, Sd_tran_edge, Sd_tran_sol = \
            self.get_Ps_tran_node(nd_core, nd_edge, nd_sol, tau_pd_core_edge, tau_pd_edge_sol, tau_pd_sol_div, vol)
        Sd_iol_edge = self.get_Ps_iol_node(nd_edge, Fd_orb_edge, tau_pd_edge_sol)
        Sd_iol_sol = self.get_Ps_iol_sol(Sd_iol_edge)

        Pd_aux_core, Pd_aux_edge, Pe_aux_core, Pe_aux_edge = self.get_Ps_aux_node(Te_core, Te_edge, t)
        P_oh_core = self.get_P_oh_node(t)
        P_oh_edge = self.get_P_oh_node(t)
        P_rad_core = self.get_P_rad_node(nd_core, ne_core, nc_core, Te_core, a, R0, B0, kappa, vol)
        P_rad_edge = self.get_P_rad_node(nd_edge, ne_edge, nc_edge, Te_edge, a, R0, B0, kappa, vol)
        P_rad_sol = self.get_P_rad_node(nd_sol, ne_sol, nc_sol, Te_sol, a, R0, B0, kappa, vol, ecr=False)
        Q_de_core = self.get_Q_de_node(nd_core, ne_core, Td_core, Te_core)
        Q_de_edge = self.get_Q_de_node(nd_edge, ne_edge, Td_edge, Te_edge)
        Q_de_sol = self.get_Q_de_node(nd_sol, ne_sol, Td_sol, Te_sol)
        Pe_ion_sol = self.get_Pe_ion_node(n0_sol, ne_sol, sigma_v_ion_sol)
        Pe_rec_sol = self.get_Pe_rec_node(nd_sol, ne_sol, sigma_v_rec_sol)
        Pd_at_sol = self.get_Pd_at_node(nd_sol, Td_sol, sigma_v_cx_sol, sigma_v_el_sol)
        Pd_tran_core, Pd_tran_edge, Pd_tran_sol = \
            self.get_Ps_tran_node(Ud_core, Ud_edge, Ud_sol, tau_ed_core_edge, tau_ed_edge_sol, tau_ed_sol_div, vol)
        Pe_tran_core, Pe_tran_edge, Pe_tran_sol = \
            self.get_Ps_tran_node(Ue_core, Ue_edge, Ue_sol, tau_ee_core_edge, tau_ee_edge_sol, tau_ee_sol_div, vol)
        Pd_iol_edge = self.get_Ps_iol_node(Ud_edge, Ed_orb_edge, tau_ed_edge_sol)
        Pd_iol_sol = self.get_Ps_iol_sol(Pd_iol_edge)

        Sd = (Sd_ext_core, Sd_ext_edge, Sd_ion_sol, Sd_rec_sol, Sd_tran_core, Sd_tran_edge, Sd_tran_sol, Sd_iol_edge,
              Sd_iol_sol)
        Pd = (Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Q_de_sol, Pd_at_sol,
              Pd_tran_core, Pd_tran_edge, Pd_tran_sol, Pd_iol_edge, Pd_iol_sol)
        Pe = (Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, P_rad_sol,
              Pe_ion_sol, Pe_rec_sol, Pe_tran_core, Pe_tran_edge, Pe_tran_sol)

        return Sd, Pd, Pe

    def get_sources_core_edge(self, nd: Tuple[torch.Tensor, ...],
                              Ud: Tuple[torch.Tensor, ...], Ue: Tuple[torch.Tensor, ...],
                              t: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Particle and energy source terms for the core and edge nodes

        :param nd: deuteron densities [m^-3]
        :param Ud: deuteron energies [J/m^3]
        :param Ue: electron energies [J/m^3]
        :param t: time [s]
        :return: Sd [m^-3 s^-1], Pd, Pe [W/m^3]
        """
        nodes = ['core', 'edge']
        nd_core, nd_edge = nd
        Ud_core, Ud_edge = Ud
        Ue_core, Ue_edge = Ue

        nc_core = self.get_nc_node(t, node='core')
        nc_edge = self.get_nc_node(t, node='edge')
        ne_core = nd_core + nc_core * self.impurity_charge
        ne_edge = nd_edge + nc_edge * self.impurity_charge

        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)

        a = self.get_a(t)
        R0 = self.get_R0(t)
        B0 = self.get_B0(t)
        kappa = self.get_kappa0(t)
        vol = self.get_vol(t)

        x_core = self.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='core')
        x_edge = self.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='edge')
        chi_core = self.get_chi_node(x_core, node='core')
        chi_edge = self.get_chi_node(x_edge, node='edge')
        tau_core = self.get_tau_node(chi_core, a, node='core')
        tau_edge = self.get_tau_node(chi_edge, a, node='edge')
        tau_pd_core_edge, tau_ed_core_edge, tau_ee_core_edge = tuple(map(torch.squeeze, tau_core.split(1, dim=-1)))
        tau_pd_edge_sol, tau_ed_edge_sol, tau_ee_edge_sol = tuple(map(torch.squeeze, tau_edge.split(1, dim=-1)))
        Fd_orb_edge, Ed_orb_edge = self.iol.get_loss_fractions(Td_edge)

        Sd_ext_core, Sd_ext_edge = self.get_Sd_ext_node(t)
        Sd_tran_core, Sd_tran_edge = \
            self.get_Ps_tran_node(nd_core, nd_edge, None, tau_pd_core_edge, tau_pd_edge_sol, None, vol, nodes=nodes)
        Sd_iol_edge = self.get_Ps_iol_node(nd_edge, Fd_orb_edge, tau_pd_edge_sol)

        Pd_aux_core, Pd_aux_edge, Pe_aux_core, Pe_aux_edge = self.get_Ps_aux_node(Te_core, Te_edge, t)
        P_oh_core = self.get_P_oh_node(t)
        P_oh_edge = self.get_P_oh_node(t)
        P_rad_core = self.get_P_rad_node(nd_core, ne_core, nc_core, Te_core, a, R0, B0, kappa, vol)
        P_rad_edge = self.get_P_rad_node(nd_edge, ne_edge, nc_edge, Te_edge, a, R0, B0, kappa, vol)
        Q_de_core = self.get_Q_de_node(nd_core, ne_core, Td_core, Te_core)
        Q_de_edge = self.get_Q_de_node(nd_edge, ne_edge, Td_edge, Te_edge)
        Pd_tran_core, Pd_tran_edge = \
            self.get_Ps_tran_node(Ud_core, Ud_edge, None, tau_ed_core_edge, tau_ed_edge_sol, None, vol, nodes=nodes)
        Pe_tran_core, Pe_tran_edge = \
            self.get_Ps_tran_node(Ue_core, Ue_edge, None, tau_ee_core_edge, tau_ee_edge_sol, None, vol, nodes=nodes)
        Pd_iol_edge = self.get_Ps_iol_node(Ud_edge, Ed_orb_edge, tau_ed_edge_sol)

        Sd = (Sd_ext_core, Sd_ext_edge, Sd_tran_core, Sd_tran_edge, Sd_iol_edge)
        Pd = (Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Pd_tran_core, Pd_tran_edge, Pd_iol_edge)
        Pe = (Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, Pe_tran_core, Pe_tran_edge)

        return Sd, Pd, Pe

    def get_sources_sol(self, nd_sol: torch.Tensor, Ud_sol: torch.Tensor, Ue_sol: torch.Tensor, t: torch.Tensor) \
            -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Particle and energy source terms for the SOL node

        :param nd_sol: SOL deuteron densities [m^-3]
        :param Ud_sol: SOL deuteron energies [J]
        :param Ue_sol: SOL electron energies [J]
        :param t: time [s]
        :return: Sd [m^-3 s^-1], Pd, Pe [W/m^3]
        """
        nodes = ['sol']
        nd_edge = self.get_nd_node(t, node='edge')
        Td_edge = self.get_Td_node(t, node='edge')
        Te_edge = self.get_Te_node(t, node='edge')
        nc_edge = self.get_nc_node(t, node='edge')
        nc_sol = self.get_nc_node(t, node='sol')
        n0_sol = self.get_n0_node(t, node='sol')
        ne_edge = nd_edge + nc_edge * self.impurity_charge
        ne_sol = nd_sol + nc_sol * self.impurity_charge

        Td_sol = J2eV(Ud_sol / nd_sol * 2 / 3)
        Te_sol = J2eV(Ue_sol / ne_sol * 2 / 3)
        Ud_edge = 3 / 2 * nd_edge * eV2J(Td_edge)
        Ue_edge = 3 / 2 * ne_edge * eV2J(Te_edge)

        a = self.get_a(t)
        R0 = self.get_R0(t)
        B0 = self.get_B0(t)
        kappa = self.get_kappa0(t)
        vol = self.get_vol(t)

        sigma_v_ion_sol = self.reaction.get_ionization_coefficient(Te_sol)
        sigma_v_rec_sol = self.reaction.get_recombination_coefficient(Te_sol)
        sigma_v_cx_sol = self.reaction.get_charge_exchange_coefficient(Td_sol)
        sigma_v_el_sol = self.reaction.get_elastic_scattering_coefficient(Td_sol)

        x_edge = self.get_x_node(None, ne_edge, ne_sol, None, Te_edge, Te_sol, a, R0, kappa, B0, t, node='edge')
        x_sol = self.get_x_node(None, ne_edge, ne_sol, None, Te_edge, Te_sol, a, R0, kappa, B0, t, node='sol')
        chi_edge = self.get_chi_node(x_edge, node='edge')
        chi_sol = self.get_chi_node(x_sol, node='sol')
        tau_edge = self.get_tau_node(chi_edge, a, node='edge')
        tau_sol = self.get_tau_node(chi_sol, a, node='sol')
        tau_pd_edge_sol, tau_ed_edge_sol, tau_ee_edge_sol = tuple(map(torch.squeeze, tau_edge.split(1, dim=-1)))
        tau_pd_sol_div, tau_ed_sol_div, tau_ee_sol_div = tuple(map(torch.squeeze, tau_sol.split(1, dim=-1)))
        Fd_orb_edge, Ed_orb_edge = self.iol.get_loss_fractions(Td_edge)

        Sd_ion_sol = self.get_Sd_ion_node(n0_sol, ne_sol, sigma_v_ion_sol)
        Sd_rec_sol = self.get_Sd_rec_node(nd_sol, ne_sol, sigma_v_rec_sol)
        Sd_tran_sol = \
            self.get_Ps_tran_node(None, nd_edge, nd_sol, None, tau_pd_edge_sol, tau_pd_sol_div, vol, nodes=nodes)
        Sd_iol_edge = self.get_Ps_iol_node(nd_edge, Fd_orb_edge, tau_pd_edge_sol)
        Sd_iol_sol = self.get_Ps_iol_sol(Sd_iol_edge)

        P_rad_sol = self.get_P_rad_node(nd_sol, ne_sol, nc_sol, Te_sol, a, R0, B0, kappa, vol, ecr=False)
        Q_de_sol = self.get_Q_de_node(nd_sol, ne_sol, Td_sol, Te_sol)
        Pe_ion_sol = self.get_Pe_ion_node(n0_sol, ne_sol, sigma_v_ion_sol)
        Pe_rec_sol = self.get_Pe_rec_node(nd_sol, ne_sol, sigma_v_rec_sol)
        Pd_at_sol = self.get_Pd_at_node(nd_sol, Td_sol, sigma_v_cx_sol, sigma_v_el_sol)
        Pd_tran_sol = \
            self.get_Ps_tran_node(None, Ud_edge, Ud_sol, None, tau_ed_edge_sol, tau_ed_sol_div, vol, nodes=nodes)
        Pe_tran_sol = \
            self.get_Ps_tran_node(None, Ue_edge, Ue_sol, None, tau_ee_edge_sol, tau_ee_sol_div, vol, nodes=nodes)
        Pd_iol_edge = self.get_Ps_iol_node(Ud_edge, Ed_orb_edge, tau_ed_edge_sol)
        Pd_iol_sol = self.get_Ps_iol_sol(Pd_iol_edge)

        Sd = (Sd_ion_sol, Sd_rec_sol, Sd_tran_sol, Sd_iol_sol)
        Pd = (Q_de_sol, Pd_at_sol, Pd_tran_sol, Pd_iol_sol)
        Pe = (P_rad_sol, Pe_ion_sol, Pe_rec_sol, Pe_tran_sol)

        return Sd, Pd, Pe

    def print_sources(self, Sd, Pd, Pe):
        """
        Print source terms

        :param Sd: deuteron particle sources [m^-3 s^-1]
        :param Pd: deuteron energy sources [W/m^3]
        :param Pe: electron energy sources [W/m^3]
        :return: None
        """
        Sd_ext_core, Sd_ext_edge, Sd_ion_sol, Sd_rec_sol, \
        Sd_tran_core, Sd_tran_edge, Sd_tran_sol, Sd_iol_edge, Sd_iol_sol = Sd
        Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Q_de_sol, Pd_at_sol, \
        Pd_tran_core, Pd_tran_edge, Pd_tran_sol, Pd_iol_edge, Pd_iol_sol = Pd
        Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, P_rad_sol, \
        Pe_ion_sol, Pe_rec_sol, Pe_tran_core, Pe_tran_edge, Pe_tran_sol = Pe

        change_density_unit = lambda Ss: torch.tensor(list(map(lambda x: x / 1E19, Ss)))
        change_energy_unit = lambda Ps: torch.tensor(list(map(lambda x: J2keV(x) / 1E19, Ps)))

        print('Sd_core', change_density_unit((Sd_ext_core, Sd_tran_core)))
        print('Sd_edge', change_density_unit((Sd_ext_edge, Sd_tran_edge, Sd_iol_edge)))
        print('Sd_sol', change_density_unit((Sd_ion_sol, Sd_rec_sol, Sd_tran_sol, Sd_iol_sol)))
        print('Pd_core', change_energy_unit((Pd_aux_core, Q_de_core, Pd_tran_core)))
        print('Pd_edge', change_energy_unit((Pd_aux_edge, Q_de_edge, Pd_tran_edge, Pd_iol_edge)))
        print('Pd_sol', change_energy_unit((Pd_at_sol, Q_de_sol, Pd_tran_sol, Pd_iol_sol)))
        print('Pe_core', change_energy_unit((P_oh_core, Pe_aux_core, -P_rad_core, -Q_de_core, Pe_tran_core)))
        print('Pe_edge', change_energy_unit((P_oh_edge, Pe_aux_edge, -P_rad_edge, -Q_de_edge, Pe_tran_edge)))
        print('Pe_sol', change_energy_unit((Pe_ion_sol, Pe_rec_sol, -P_rad_sol, -Q_de_sol, Pe_tran_sol)))

    def system_core_edge(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Dynamical system for the core and edge nodes in the multi-nodal model

        :param t: time [s]
        :param y: nd_core, nd_edge [10^19 m^-3], Ud_core, Ud_edge, Ue_core, Ue_edge [10^19 keV/m^3]
        :return: dy/dt = [Sd_core, Sd_edge, Pd_core, Pd_edge, Pe_core, Pe_edge]
        """
        self.system_count += 1

        y = torch.abs(y)
        nd_core = y[0] * 1E19  # [10^19 m^-3] -> [m^-3]
        nd_edge = y[1] * 1E19
        Ud_core = keV2J(y[2] * 1E19)  # [10^19 keV] -> [J]
        Ud_edge = keV2J(y[3] * 1E19)
        Ue_core = keV2J(y[4] * 1E19)
        Ue_edge = keV2J(y[5] * 1E19)

        Sd, Pd, Pe = self.get_sources_core_edge((nd_core, nd_edge), (Ud_core, Ud_edge),
                                                (Ue_core, Ue_edge), t)
        Sd_ext_core, Sd_ext_edge, Sd_tran_core, Sd_tran_edge, Sd_iol_edge = Sd
        Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Pd_tran_core, Pd_tran_edge, Pd_iol_edge = Pd
        Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, Pe_tran_core, Pe_tran_edge = Pe

        Sd_core = Sd_ext_core + Sd_tran_core
        Sd_edge = Sd_ext_edge + Sd_tran_edge + Sd_iol_edge
        Pd_core = Pd_aux_core + Q_de_core + Pd_tran_core
        Pd_edge = Pd_aux_edge + Q_de_edge + Pd_tran_edge + Pd_iol_edge
        Pe_core = P_oh_core + Pe_aux_core - P_rad_core - Q_de_core + Pe_tran_core
        Pe_edge = P_oh_edge + Pe_aux_edge - P_rad_edge - Q_de_edge + Pe_tran_edge

        dy_dt = torch.zeros_like(y)
        dy_dt[0] = Sd_core / 1E19  # [m^-3] -> [10^19 m^-3]
        dy_dt[1] = Sd_edge / 1E19
        dy_dt[2] = J2keV(Pd_core) / 1E19  # [J] -> [10^19 keV]
        dy_dt[3] = J2keV(Pd_edge) / 1E19
        dy_dt[4] = J2keV(Pe_core) / 1E19
        dy_dt[5] = J2keV(Pe_edge) / 1E19

        return dy_dt

    def system_sol(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Dynamical system for the SOL node in the multi-nodal model

        :param t: time [s]
        :param y: nd_sol [10^19 m^-3], Ud_sol, Ue_sol [10^19 keV/m^3]
        :return: dy/dt [Sd_sol, Pd_sol, Pe_sol]
        """
        self.system_count += 1

        y = torch.abs(y)
        nd_sol = y[0] * 1E19  # [10^19 m^-3] -> [m^-3]
        Ud_sol = keV2J(y[1] * 1E19)  # [10^19 keV] -> [J]
        Ue_sol = keV2J(y[2] * 1E19)  # [10^19 keV] -> [J]

        Sd, Pd, Pe = self.get_sources_sol(nd_sol, Ud_sol, Ue_sol, t)
        Sd_ion_sol, Sd_rec_sol, Sd_tran_sol, Sd_iol_sol = Sd
        Q_de_sol, Pd_at_sol, Pd_tran_sol, Pd_iol_sol = Pd
        P_rad_sol, Pe_ion_sol, Pe_rec_sol, Pe_tran_sol = Pe

        Sd_sol = Sd_ion_sol + Sd_rec_sol + Sd_tran_sol + Sd_iol_sol
        Pd_sol = Pd_at_sol + Q_de_sol + Pd_tran_sol + Pd_iol_sol
        Pe_sol = Pe_ion_sol + Pe_rec_sol - P_rad_sol - Q_de_sol + Pe_tran_sol

        dy_dt = torch.zeros_like(y)
        dy_dt[0] = Sd_sol / 1E19  # [m^-3] -> [10^19 m^-3]
        dy_dt[1] = J2keV(Pd_sol) / 1E19  # [J] -> [10^19 keV]
        dy_dt[2] = J2keV(Pe_sol) / 1E19  # [J] -> [10^19 keV]

        return dy_dt

    def system(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Dynamical system for the multi-nodal model

        :param t: time [s]
        :param y: nd_core, nd_edge, nd_sol [10^19 m^-3],
                  Ud_core, Ud_edge, Ud_sol, Ue_core, Ue_edge, Ue_sol [10^19 keV/m^3]
        :return: dy/dt = [Sd_core, Sd_edge, Sd_sol, Pd_core, Pd_edge, Pd_sol, Pe_core, Pe_edge, Pe_sol]
        """
        self.system_count += 1

        y = torch.abs(y)
        nd_core = y[0] * 1E19  # [10^19 m^-3] -> [m^-3]
        nd_edge = y[1] * 1E19
        nd_sol = y[2] * 1E19
        Ud_core = keV2J(y[3] * 1E19)  # [10^19 keV] -> [J]
        Ud_edge = keV2J(y[4] * 1E19)
        Ud_sol = keV2J(y[5] * 1E19)
        Ue_core = keV2J(y[6] * 1E19)
        Ue_edge = keV2J(y[7] * 1E19)
        Ue_sol = keV2J(y[8] * 1E19)

        Sd, Pd, Pe = self.get_sources((nd_core, nd_edge, nd_sol), (Ud_core, Ud_edge, Ud_sol),
                                      (Ue_core, Ue_edge, Ue_sol), t)
        Sd_ext_core, Sd_ext_edge, Sd_ion_sol, Sd_rec_sol, Sd_tran_core, Sd_tran_edge, Sd_tran_sol, \
        Sd_iol_edge, Sd_iol_sol = Sd
        Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Q_de_sol, Pd_at_sol, \
        Pd_tran_core, Pd_tran_edge, Pd_tran_sol, Pd_iol_edge, Pd_iol_sol = Pd
        Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, P_rad_sol, \
        Pe_ion_sol, Pe_rec_sol, Pe_tran_core, Pe_tran_edge, Pe_tran_sol = Pe

        Sd_core = Sd_ext_core + Sd_tran_core
        Sd_edge = Sd_ext_edge + Sd_tran_edge + Sd_iol_edge
        Sd_sol = Sd_ion_sol + Sd_rec_sol + Sd_tran_sol + Sd_iol_sol
        Pd_core = Pd_aux_core + Q_de_core + Pd_tran_core
        Pd_edge = Pd_aux_edge + Q_de_edge + Pd_tran_edge + Pd_iol_edge
        Pd_sol = Pd_at_sol + Q_de_sol + Pd_tran_sol + Pd_iol_sol
        Pe_core = P_oh_core + Pe_aux_core - P_rad_core - Q_de_core + Pe_tran_core
        Pe_edge = P_oh_edge + Pe_aux_edge - P_rad_edge - Q_de_edge + Pe_tran_edge
        Pe_sol = Pe_ion_sol + Pe_rec_sol - P_rad_sol - Q_de_sol + Pe_tran_sol

        dy_dt = torch.zeros_like(y)
        dy_dt[0] = Sd_core / 1E19  # [m^-3] -> [10^19 m^-3]
        dy_dt[1] = Sd_edge / 1E19
        dy_dt[2] = Sd_sol / 1E19
        dy_dt[3] = J2keV(Pd_core) / 1E19  # [J] -> [10^19 keV]
        dy_dt[4] = J2keV(Pd_edge) / 1E19
        dy_dt[5] = J2keV(Pd_sol) / 1E19
        dy_dt[6] = J2keV(Pe_core) / 1E19
        dy_dt[7] = J2keV(Pe_edge) / 1E19
        dy_dt[8] = J2keV(Pe_sol) / 1E19

        return dy_dt


class ReactorITER(Reactor1D):
    """
    One-Dimension Fusion Reactor for ITER
    """

    def __init__(self, scenario: int, tuned: bool = False, net: nn.Module = None, delayed_fusion: bool = True):
        """
        Initialize the 1D reactor for ITER

        :param scenario: scenario number
        :param tuned: true for tuned transport times
        :param net: the neural network for the diffusivity model
        :param delayed_fusion: delayed fusion heating
        """
        super().__init__(shot_num=None, tuned=tuned, net=net)
        self.scenario = scenario
        self.shot_num = scenario
        self.nodes = config.nodes_iter
        self.num_vars = config.num_vars_iter[self.dim]
        self.M = (A_d + A_t) / 2
        self.sensitivity_analysis = False
        self.delayed_fusion = delayed_fusion

        preprocessor = PreprocessorITER()
        self.shot = preprocessor.preprocess(self.scenario)
        self.time = self.shot['time']
        self.ecr_parameters = self.shot['ecr_parameters']
        self.iol_parameters = self.shot['iol_parameters']
        self.impurity_fractions = self.shot['impurity_fractions']
        self.Zz = self.get_Zz()

        self.iol_d = IOL(R0=self.get_average(self.get_R0(self.time)), r0=self.get_average(self.get_a(self.time)),
                         mi=m_d, B0=self.get_average(self.get_B0(self.time)),
                         IP=self.get_average(self.get_IP(self.time)), reactor_type='iter')
        self.iol_t = IOL(R0=self.get_average(self.get_R0(self.time)), r0=self.get_average(self.get_a(self.time)),
                         mi=m_t, B0=self.get_average(self.get_B0(self.time)),
                         IP=self.get_average(self.get_IP(self.time)), reactor_type='iter')
        self.iol_a = IOL(R0=self.get_average(self.get_R0(self.time)), r0=self.get_average(self.get_a(self.time)),
                         mi=m_a, B0=self.get_average(self.get_B0(self.time)),
                         IP=self.get_average(self.get_IP(self.time)), reactor_type='iter')

    def get_Zz(self):
        """
        Atomic numbers of impurities

        :return: Zz
        """
        Zz = []
        for impurity in self.impurity_fractions.keys():
            Zz.append(Z_imp[impurity])
        return tuple(Zz)

    def get_rhos_node(self, node: str) -> torch.Tensor:
        """
        Normalized radii in one node

        :param node: node name
        :return: rhos_node
        """
        assert node.lower() in self.nodes
        return self.shot['rhos_' + node.lower()]

    def get_nd_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Deuteron density in one node

        :param t: time [s]
        :param node: node name
        :return: nd_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['nd_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_nt_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Triton density in one node

        :param t: time [s]
        :param node: node name
        :return: nt_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.get_nd_node(t, node=node)

    def get_na_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Alpha particle density in one node

        :param t: time [s]
        :param node: node name
        :return: na_node [m^-3]
        """
        assert node.lower() in self.nodes
        return self.interp1d(self.time, self.shot['na_' + node.lower()], torch.reshape(t, (-1,))).squeeze()

    def get_ne_nz_node(self, nd_node, nt_node, na_node):
        """
        Electron and impurity densities in one node by the charge neutrality

        :param nd_node: deuteron density [m^-3]
        :param nt_node: triton density [m^-3]
        :param na_node: alpha particle density [m^-3]
        :return: ne_node, nz_node [m^-3]
        """
        divisor = 1
        for impurity, impurity_fraction in self.impurity_fractions.items():
            divisor -= Z_imp[impurity] * impurity_fraction
        ne_node = (nd_node + nt_node + Z_a * na_node) / divisor
        nz_node = []
        for impurity_fraction in self.impurity_fractions.values():
            nz_node.append(impurity_fraction * ne_node)
        return ne_node, nz_node

    def get_Tt_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Triton temperature in one node

        :param t: time [s]
        :param node: node name
        :return: Tt_node [eV]
        """
        assert node.lower() in self.nodes
        return self.get_Td_node(t, node=node)

    def get_Ta_node(self, t: torch.Tensor, node: str) -> torch.Tensor:
        """
        Alpha particle temperature in one node

        :param t: time [s]
        :param node: node name
        :return: Ta_node [eV]
        """
        assert node.lower() in self.nodes
        return self.get_Td_node(t, node=node)

    def get_Ti_node(self, nd_node, nt_node, Td_node, Tt_node, na_node=None, Ta_node=None):
        """
        Averaged ion temperature of deuterons and tritons

        :param nd_node: deuteron density [m^-3]
        :param nt_node: triton density [m^-3]
        :param Td_node: deuteron temperature [eV]
        :param Tt_node: triton temperature [eV]
        :param na_node: alpha particle density [m^-3]
        :param Ta_node: alpha particle temperature [eV]
        :return: Ti_node [eV]
        """
        if na_node is not None and Ta_node is not None:
            Ti_node = (nd_node * Td_node + nt_node * Tt_node + na_node * Ta_node) / (nd_node + nt_node + na_node)
        else:
            Ti_node = (nd_node * Td_node + nt_node * Tt_node) / (nd_node + nt_node)
        return Ti_node

    def get_sigma_v_fus(self, Ti):
        """
        Fusion reactivity with only the D-T reaction

        :param Ti: ion temperature [eV]
        :return: sigma_v_fus [m^3/s]
        """
        sigma_v_fus = self.reaction.get_fusion_coefficient(Ti, 'tdna')
        return sigma_v_fus

    def get_Ss_fus_node(self, nd_node, nt_node, Td_node, Tt_node):
        """
        Deuterium-Tritium (D-T) fusion particle sources in one node

        :param nd_node: deuteron density [m^-3]
        :param nt_node: triton density [m^-3]
        :param Td_node: deuteron temperature [eV]
        :param Tt_node: triton temperature [eV]
        :return: Sd_fus_node, St_fus_node, Sa_fus_node [m^-3 s^-1]
        """
        Ti_node = self.get_Ti_node(nd_node=nd_node, nt_node=nt_node, Td_node=Td_node, Tt_node=Tt_node)
        sigma_v_fus_node = self.get_sigma_v_fus(Ti_node)
        Sa_fus_node = nd_node * nt_node * sigma_v_fus_node
        Sd_fus_node = - Sa_fus_node
        St_fus_node = - Sa_fus_node
        return Sd_fus_node, St_fus_node, Sa_fus_node

    def get_Ps_fus_node_iter(self, nd_node, nt_node, na_node, Te_node, Sa_fus_node):
        """
        Deuterium-Tritium (D-T) fusion power in one node

        :param nd_node: deuteron density [m^-3]
        :param nt_node: triton density [m^-3]
        :param na_node: alpha particle density [m^-3]
        :param Te_node: electron temperature [eV]
        :param Sa_fus_node: fusion particle source [m^-3 s^-1]
        :return: Pd_fus_node, Pt_fus_node, Pa_fus_node, Pe_fus_node [W/m^3]
        """
        Uf = self.reaction.get_fusion_energy('tdna')
        m_i = (nd_node * m_d + nt_node * m_t + na_node * m_a) / (nd_node + nt_node + na_node)
        nbi2i = self.get_nbi2i(Te=Te_node, m_i=m_i, m_b=m_a, E_b0=Uf)

        nbi2d = nbi2i * nd_node / (nd_node + nt_node + na_node)
        nbi2t = nbi2i * nt_node / (nd_node + nt_node + na_node)
        nbi2a = nbi2i * na_node / (nd_node + nt_node + na_node)
        nbi2e = 1 - nbi2i

        Pd_fus_node = Sa_fus_node * eV2J(Uf) * nbi2d
        Pt_fus_node = Sa_fus_node * eV2J(Uf) * nbi2t
        Pa_fus_node = Sa_fus_node * eV2J(Uf) * nbi2a
        Pe_fus_node = Sa_fus_node * eV2J(Uf) * nbi2e

        return Pd_fus_node, Pt_fus_node, Pa_fus_node, Pe_fus_node

    def get_coulomb_log_ss(self, ns_Ts: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor,
                           A1: int, A2: int, Z1: int, Z2: int) -> torch.Tensor:
        """
        Classical Coulomb logarithm between two species with different species

        :param ns_Ts: n_star / T_star [m^-3/eV]
        :param T1: temperature [eV]
        :param T2: temperature [eV]
        :param A1: mass number []
        :param A2: mass number []
        :param Z1: atomic number []
        :param Z2: atomic number []
        :return: coulomb_log_ss []
        """
        coulomb_log_12 = 30.37 - np.log(Z1 * Z2) - log((A1 + A2) / (A2 * T1 + A1 * T2) * sqrt(ns_Ts))
        return coulomb_log_12

    def get_ns_Ts(self, ns: Tuple[torch.Tensor, ...], Ts: Tuple[torch.Tensor, ...],
                  zs: Tuple[int, ...]) -> torch.Tensor:
        """
        n_star / T_star in the classical Coulomb logarithm formula

        :param ns: densities [m^-3]
        :param Ts: temperatures [eV]
        :param zs: atomic numbers []
        :return: ns_Ts [m^-3/eV]
        """
        assert len(ns) == len(Ts) and len(Ts) == len(zs) and len(ns) > 0
        ns_Ts = torch.zeros_like(ns[0])
        for n, T, z in zip(ns, Ts, zs):
            ns_Ts += z ** 2 * n / T
        return ns_Ts

    def get_nu_ii(self, n2, T1, m1, m2, z1: int, z2: int, coulomb_log_12):
        """
        Characteristic collision frequency between two species

        :param n2: density [m^-3]
        :param T1: temperature [eV]
        :param m1: mass [kg]
        :param m2: mass [kg]
        :param z1: atomic number []
        :param z2: atomic number []
        :param coulomb_log_12: Coulomb logarithm []
        :return: nu_ii [Hz]
        """
        q1 = z1 * e
        q2 = z2 * e
        nu_12 = ((2 * np.sqrt(2)) / (3 * np.sqrt(pi))) * ((q1 * q2) / (4 * pi * epsilon_0)) ** 2 \
                * ((4 * pi * n2) / eV2J(T1) ** 1.5) * ((m1 * m2) ** 0.5 / (m1 + m2) ** 1.5) * coulomb_log_12
        return nu_12

    def get_Q_ii_node(self, n1, n2, T1, T2, m1, m2, z1: int, z2: int, coulomb_log_12):
        """
        Collisional energy transfer between two ions

        :param n1: density [m^-3]
        :param n2: density [m^-3]
        :param T1: temperature [eV]
        :param T2: temperature [eV]
        :param m1: mass [kg]
        :param m2: mass [kg]
        :param z1: atomic number []
        :param z2: atomic number []
        :param coulomb_log_12: Coulomb logarithm []
        :return: Q_ii_node [W/m^3]
        """
        nu_12 = self.get_nu_ii(n2, T1, m1, m2, z1, z2, coulomb_log_12)
        Q_12 = 1.5 * n1 * eV2J(T2 - T1) * nu_12
        return Q_12

    def get_Q_ss_node(self, n1, n2, T1, T2, m1, m2, z1: int, z2: int, coulomb_log_12):
        """
        Collisional energy transfer from one energetic ion to another species

        :param n1: density [m^-3]
        :param n2: density [m^-3]
        :param T1: temperature [eV]
        :param T2: temperature [eV]
        :param m1: mass [kg]
        :param m2: mass [kg]
        :param z1: atomic number []
        :param z2: atomic number []
        :param coulomb_log_12: Coulomb logarithm []
        :return: Q_ss_node [W/m^3]
        """
        q1 = z1 * e
        q2 = z2 * e
        V1 = sqrt(2 * eV2J(T1) / m1)
        V2 = sqrt(2 * eV2J(T2) / m2)

        gamma_12 = (n2 * (q1 * q2) ** 2 * coulomb_log_12) / (4 * pi * epsilon_0 ** 2 * m1 ** 2)
        dV12_dt = (gamma_12 / V1) * (1 + m1 / m2) \
                  * ((2 / np.sqrt(pi)) * exp(- V1 ** 2 / V2 ** 2) / V2 - erf(V1 / V2) / V1)
        Q_12 = n1 * m1 * V1 * dV12_dt
        return Q_12

    def get_Q_ie_node(self, ni, ne, Ti, Te, mi, me, zi: int, ze: int, coulomb_log_ie):
        """
        Collisional energy transfer from ions to electrons

        :param ni: ion density [m^-3]
        :param ne: electron density [m^-3]
        :param Ti: ion temperature [eV]
        :param Te: electron temperature [eV]
        :param coulomb_log_ie: Coulomb logarithm []
        :param mi: ion mass [kg]
        :param me: electron mass [kg]
        :param zi: atomic number []
        :param ze: atomic number []
        :return: Q_ie_node [W/m^3]
        """
        qi = zi * e
        qe = ze * e
        Q_ie = (ni * ne * qi ** 2 * qe ** 2 * me * coulomb_log_ie * (1 - Ti / Te)) / (
                2 * pi * epsilon_0 ** 2 * sqrt(2 * pi * me * eV2J(Te)) * mi
                * (1 + 4 * np.sqrt(pi) / 3 * ((3 * me * Ti) / (2 * mi * Te)) ** 1.5)
        )
        return Q_ie

    def get_Qs_node(self, nd_node, nt_node, na_node, ne_node, Td_node, Tt_node, Ta_node, Te_node):
        """
        Collisional energy transfer in one node

        :param nd_node: deuteron density [m^-3]
        :param nt_node: triton density [m^-3]
        :param na_node: alpha particle density [m^-3]
        :param ne_node: electron density [m^-3]
        :param Td_node: deuteron temperature [eV]
        :param Tt_node: triton temperature [eV]
        :param Ta_node: alpha particle temperature [eV]
        :param Te_node: electron temperature [eV]
        :return: Qd_node, Qt_node, Qa_node, Qe_node, (Qs_node) [W/m^3]
        """
        ns_Ts = self.get_ns_Ts((nd_node, nt_node, na_node, ne_node), (Td_node, Tt_node, Ta_node, Te_node), (1, 1, 2, 1))

        coulomb_log_ad = self.get_coulomb_log_ss(ns_Ts, Ta_node, Td_node, A1=A_a, A2=A_d, Z1=Z_a, Z2=Z_d)
        coulomb_log_at = self.get_coulomb_log_ss(ns_Ts, Ta_node, Tt_node, A1=A_a, A2=A_t, Z1=Z_a, Z2=Z_t)
        coulomb_log_ae = self.get_coulomb_log_ss(ns_Ts, Ta_node, Te_node, A1=A_a, A2=A_e, Z1=Z_a, Z2=Z_e)
        coulomb_log_de = self.get_coulomb_log_ss(ns_Ts, Td_node, Te_node, A1=A_d, A2=A_e, Z1=Z_d, Z2=Z_e)
        coulomb_log_te = self.get_coulomb_log_ss(ns_Ts, Tt_node, Te_node, A1=A_t, A2=A_e, Z1=Z_t, Z2=Z_e)
        coulomb_log_dt = self.get_coulomb_log_ss(ns_Ts, Td_node, Tt_node, A1=A_d, A2=A_t, Z1=Z_d, Z2=Z_t)

        Q_ad_node = self.get_Q_ii_node(na_node, nd_node, Ta_node, Td_node, m_a, m_d, z1=Z_a, z2=Z_d,
                                       coulomb_log_12=coulomb_log_ad)
        Q_at_node = self.get_Q_ii_node(na_node, nt_node, Ta_node, Tt_node, m_a, m_t, z1=Z_a, z2=Z_t,
                                       coulomb_log_12=coulomb_log_at)
        Q_dt_node = self.get_Q_ii_node(nd_node, nt_node, Td_node, Tt_node, m_d, m_t, z1=Z_d, z2=Z_t,
                                       coulomb_log_12=coulomb_log_dt)

        Q_de_node = self.get_Q_ie_node(nd_node, ne_node, Td_node, Te_node, m_d, m_e, zi=Z_d, ze=Z_e,
                                       coulomb_log_ie=coulomb_log_de)
        Q_te_node = self.get_Q_ie_node(nt_node, ne_node, Tt_node, Te_node, m_t, m_e, zi=Z_t, ze=Z_e,
                                       coulomb_log_ie=coulomb_log_te)
        Q_ae_node = self.get_Q_ie_node(na_node, ne_node, Ta_node, Te_node, m_a, m_e, zi=Z_a, ze=Z_e,
                                       coulomb_log_ie=coulomb_log_ae)

        Qd_node = -Q_ad_node + Q_dt_node + Q_de_node
        Qt_node = -Q_at_node - Q_dt_node + Q_te_node
        Qa_node = Q_ad_node + Q_at_node + Q_ae_node
        Qe_node = -Q_ae_node - Q_de_node - Q_te_node
        Qs_node = (Q_ad_node, Q_at_node, Q_dt_node, Q_de_node, Q_te_node, Q_ae_node)

        return Qd_node, Qt_node, Qa_node, Qe_node, Qs_node

    def get_Ss_ext_node(self, t: torch.Tensor):
        """
        External particle source for nodes

        :param t: time [s]
        :return: Sd_ext_core, Sd_ext_edge, St_ext_core, St_ext_edge, Sa_ext_core, Sa_ext_edge [m^-3 s^-1]
        """
        E_b0 = self.get_E_b0()
        P_nbi = self.get_P_nbi(t)

        Si_nbi = P_nbi / eV2J(E_b0)
        Si_gas = self.get_Si_gas(t)

        Sd_ext_core = self.heater.f_nbi_core * Si_nbi + self.heater.f_gas_core * Si_gas
        Sd_ext_edge = self.heater.f_nbi_edge * Si_nbi + self.heater.f_gas_edge * Si_gas
        St_ext_core = Sd_ext_core
        St_ext_edge = Sd_ext_edge
        Sa_ext_core = torch.zeros_like(t)
        Sa_ext_edge = torch.zeros_like(t)

        return Sd_ext_core, Sd_ext_edge, St_ext_core, St_ext_edge, Sa_ext_core, Sa_ext_edge

    def get_E_b0(self):
        """
        Initial beam energy

        :return: E_b0 [eV]
        """
        E_b0 = 1E6
        return E_b0

    def get_tau_se(self, ne, Te, m_b, Z_b):
        """
        Electron slowing-down time

        :param ne: electron density [m^-3]
        :param Te: electron temperature [eV]
        :param m_b: beam particle mass [kg]
        :param Z_b: beam atomic number []
        :return: tau_se [s]
        """
        coulomb_log = log(12 * pi * sqrt((epsilon_0 * eV2J(Te)) ** 3 / (ne * e ** 6 * Z_e ** 4 * Z_b ** 2)))
        A_D = (ne * e ** 4 * Z_b ** 2 * Z_e ** 2 * coulomb_log) / (2 * pi * epsilon_0 ** 2 * m_b ** 2)
        tau_se = (3 * (2 * pi) ** (1 / 2) * eV2J(Te) ** (3 / 2)) / (m_e ** (1 / 2) * m_b * A_D)
        return tau_se

    def get_Ts_node_delay(self, Ts_node, dTs_node_dt, tau_se_node):
        """
        Delayed nodal temperature

        :param Ts_node: nodal temperature [eV]
        :param dTs_node_dt: nodal temperature derivative [ev/s]
        :param tau_se_node: electron slowing-down time [s]
        :return: Ts_node_delay [eV]
        """
        Ts_node_delay = Ts_node - dTs_node_dt * tau_se_node
        return Ts_node_delay

    def get_P_oh_node(self, t: torch.Tensor, **kwargs):
        """
        Ohmic power in one node

        :param t: time [s]
        :key Z_eff: effective atomic number []
        :key IP: plasma current [A]
        :key a: minor radius [m]
        :key Te_node: electron temperature [eV]
        :return: P_oh_node [W/m^3]
        """
        Z_eff = kwargs['Z_eff']
        IP = kwargs['IP']
        a = kwargs['a']
        Te_node = kwargs['Te_node'] / 1E3  # [eV] -> [keV]
        P_oh_node = 2.8E-9 * Z_eff * IP ** 2 / (a ** 4 * Te_node ** 1.5)
        return P_oh_node

    def get_Ps_aux_node(self, Te_core, Te_edge, t, **kwargs):
        """
        Auxiliary heating power for nodes

        :param Te_core: core electron temperature [eV]
        :param Te_edge: edge electron temperature [eV]
        :param t: time [s]
        :key nd_core: core deuteron density [m^-3]
        :key nt_core: core triton density [m^-3]
        :key nd_edge: edge deuteron density [m^-3]
        :key nt_edge: edge triton density [m^-3]
        :return: Pd_aux_core, Pd_aux_edge, Pt_aux_core, Pt_aux_edge,
        Pa_aux_core, Pa_aux_edge, Pe_aux_core, Pe_aux_edge [W/m^3]
        """
        fd_core = kwargs['nd_core'] / (kwargs['nd_core'] + kwargs['nt_core'])
        fd_edge = kwargs['nd_edge'] / (kwargs['nd_edge'] + kwargs['nt_edge'])
        ft_core = 1 - fd_core
        ft_edge = 1 - fd_edge

        E_b0 = self.get_E_b0()
        nbi2i_core = self.get_nbi2i(Te_core, m_d, m_d, E_b0)
        nbi2i_edge = self.get_nbi2i(Te_edge, m_d, m_d, E_b0)
        nbi2e_core = 1 - nbi2i_core
        nbi2e_edge = 1 - nbi2i_edge

        P_nbi = self.get_P_nbi(t)
        P_ich = self.get_P_ich(t)
        P_ech = self.get_P_ech(t)
        P_lh = self.get_P_lh(t)

        Pi_nbi_core = nbi2i_core * self.heater.f_nbi_core * P_nbi
        Pi_nbi_edge = nbi2i_edge * self.heater.f_nbi_edge * P_nbi
        Pe_nbi_core = nbi2e_core * self.heater.f_nbi_core * P_nbi
        Pe_nbi_edge = nbi2e_edge * self.heater.f_nbi_edge * P_nbi

        Pi_ich_core = self.heater.f_ch_core * P_ich
        Pi_ich_edge = self.heater.f_ch_edge * P_ich
        Pe_ech_core = self.heater.f_ch_core * P_ech
        Pe_ech_edge = self.heater.f_ch_edge * P_ech
        Pe_lh_core = self.heater.f_ch_core * P_lh
        Pe_lh_edge = self.heater.f_ch_edge * P_lh

        Pd_aux_core = fd_core * (Pi_nbi_core + Pi_ich_core)
        Pd_aux_edge = fd_edge * (Pi_nbi_edge + Pi_ich_edge)
        Pt_aux_core = ft_core * (Pi_nbi_core + Pi_ich_core)
        Pt_aux_edge = ft_edge * (Pi_nbi_edge + Pi_ich_edge)
        Pa_aux_core = torch.zeros_like(t)
        Pa_aux_edge = torch.zeros_like(t)
        Pe_aux_core = Pe_nbi_core + Pe_ech_core + Pe_lh_core
        Pe_aux_edge = Pe_nbi_edge + Pe_ech_edge + Pe_lh_edge

        return Pd_aux_core, Pd_aux_edge, Pt_aux_core, Pt_aux_edge, Pa_aux_core, Pa_aux_edge, Pe_aux_core, Pe_aux_edge

    def get_P_ecr_node(self, ne_core, ne_edge, Te_core, Te_edge, a, R0, B0, kappa, vol):
        """
        ECR power in nodes

        :param ne_core: core electron density [m^-3]
        :param ne_edge: edge electron density [m^-3]
        :param Te_core: core electron temperature [eV]
        :param Te_edge: edge electron temperature [eV]
        :param a: minor radius [m]
        :param R0: major radius [m]
        :param B0: magnetic field [T]
        :param kappa: elongation []
        :param vol: volume [m^3]
        :return: P_ecr_core, P_ecr_edge [W/m^3]
        """
        vol_core = self.get_vol_node(vol=vol, node='core')
        vol_edge = self.get_vol_node(vol=vol, node='edge')
        ne0 = (1 + self.ecr_parameters['alpha_ne']) * (vol_core * ne_core + vol_edge * ne_edge) / vol
        Te0 = self.ecr_parameters['core2e0'] * Te_core + self.ecr_parameters['edge2e0'] * Te_edge

        P_ecr = self.get_P_ecr(ne0, Te0, a, R0, B0, kappa, vol, **self.ecr_parameters)
        P_ecr_core = self.heater.f_ecr_core * P_ecr
        P_ecr_edge = self.heater.f_ecr_edge * P_ecr
        return P_ecr_core, P_ecr_edge

    def get_P_rad_node_iter(self, ndt_node, nz_node, ne_node, Te_node, P_ecr_node, Zz_node, Z_eff_node):
        """
        Radiative Power Loss in one node

        :param ndt_node: deuteron and triton density [m^-3]
        :param nz_node: impurity density [m^-3]
        :param ne_node: electron density [m^-3]
        :param Te_node: electron temperature [eV]
        :param P_ecr_node: ECR power [W/m^3]
        :param Zz_node: impurity charge []
        :param Z_eff_node: effective atomic number []
        :return: P_rad_node, (Ps_rad_node) [W/m^3]
        """
        P_brem_node = self.get_P_brem(ndt_node, ne_node, Te_node, Z_eff_node)
        P_imp_node = self.get_P_imp(ne_node, nz_node, Te_node, Zz_node)
        P_rad_node = P_ecr_node + P_brem_node + P_imp_node
        return P_rad_node, (P_ecr_node, P_brem_node, P_imp_node)

    def get_Z_eff(self, ni, nz, Zz):
        """
        Effective atomic number

        :param ni: ion density [m^-3]
        :param ni: ion density [m^-3]
        :param nz: impurity density [m^-3]
        :param Zz: impurity atomic number []
        :return: Z_eff []
        """
        Z_eff_squared = torch.ones_like(ni)
        if isinstance(Zz, Iterable):
            for _nz, _Zz in zip(nz, Zz):
                Z_eff_squared += _nz * _Zz ** 2 / ni
        else:
            Z_eff_squared += nz * Zz ** 2 / ni
        Z_eff = sqrt(Z_eff_squared)
        return Z_eff

    def get_sources_iter(self, nd: Tuple[torch.Tensor, ...], nt: Tuple[torch.Tensor, ...], na: Tuple[torch.Tensor, ...],
                         Ud: Tuple[torch.Tensor, ...], Ut: Tuple[torch.Tensor, ...], Ua: Tuple[torch.Tensor, ...],
                         Ue: Tuple[torch.Tensor, ...], t: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Particle and energy source terms for ITER

        :param nd: deuteron densities [m^-3]
        :param nt: triton densities [m^-3]
        :param na: alpha particle densities [m^-3]
        :param Ud: deuteron energies [J/m^3]
        :param Ut: triton energies [J/m^3]
        :param Ua: alpha particle energies [J/m^3]
        :param Ue: electron energies [J/m^3]
        :param t: time [s]
        :return: Sd, St, Sa [m^-3 s^-1], Pd, Pt, Pa, Pe, (Ps) [W/m^3]
        """
        nodes = self.nodes
        nd_core, nd_edge = nd
        nt_core, nt_edge = nt
        na_core, na_edge = na
        Ud_core, Ud_edge = Ud
        Ut_core, Ut_edge = Ut
        Ua_core, Ua_edge = Ua
        Ue_core, Ue_edge = Ue

        ne_core, nz_core = self.get_ne_nz_node(nd_core, nt_core, na_core)
        ne_edge, nz_edge = self.get_ne_nz_node(nd_edge, nt_edge, na_edge)
        naz_core = (na_core, *nz_core)
        naz_edge = (na_edge, *nz_edge)
        Zaz = (Z_a, *self.Zz)

        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Tt_core = J2eV(Ut_core / nt_core * 2 / 3)
        Tt_edge = J2eV(Ut_edge / nt_edge * 2 / 3)
        Ta_core = J2eV(Ua_core / na_core * 2 / 3)
        Ta_edge = J2eV(Ua_edge / na_edge * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)

        # Tokamak parameters:
        a = self.get_a(t)
        R0 = self.get_R0(t)
        B0 = self.get_B0(t)
        IP = self.get_IP(t)
        kappa = self.get_kappa95(t)
        vol = self.get_vol(t)
        Z_eff_core = self.get_Z_eff(nd_core + nt_core, naz_core, Zaz)
        Z_eff_edge = self.get_Z_eff(nd_edge + nt_edge, naz_edge, Zaz)

        # Diffusivities:
        x_core = self.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='core')
        x_edge = self.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='edge')
        chi_core = self.get_chi_node(x_core, node='core', alpha_particle=True)
        chi_edge = self.get_chi_node(x_edge, node='edge', alpha_particle=True)

        # Transport times:
        tau_core = self.get_tau_node(chi_core, a, node='core')
        tau_edge = self.get_tau_node(chi_edge, a, node='edge')
        tau_pd_core_edge, tau_pa_core_edge, tau_ed_core_edge, tau_ea_core_edge, tau_ee_core_edge = \
            tuple(map(torch.squeeze, tau_core.split(1, dim=-1)))
        tau_pd_edge_sol, tau_pa_edge_sol, tau_ed_edge_sol, tau_ea_edge_sol, tau_ee_edge_sol = \
            tuple(map(torch.squeeze, tau_edge.split(1, dim=-1)))
        tau_pt_core_edge, tau_pt_edge_sol = tau_pd_core_edge, tau_pd_edge_sol
        tau_et_core_edge, tau_et_edge_sol = tau_ed_core_edge, tau_ed_edge_sol

        # Particle transport:
        Sd_tran_core, Sd_tran_edge = \
            self.get_Ps_tran_node(nd_core, nd_edge, None, tau_pd_core_edge, tau_pd_edge_sol, None, vol, nodes=nodes)
        St_tran_core, St_tran_edge = \
            self.get_Ps_tran_node(nt_core, nt_edge, None, tau_pt_core_edge, tau_pt_edge_sol, None, vol, nodes=nodes)
        Sa_tran_core, Sa_tran_edge = \
            self.get_Ps_tran_node(na_core, na_edge, None, tau_pa_core_edge, tau_pa_edge_sol, None, vol, nodes=nodes)

        # Energy transport:
        Pd_tran_core, Pd_tran_edge = \
            self.get_Ps_tran_node(Ud_core, Ud_edge, None, tau_ed_core_edge, tau_ed_edge_sol, None, vol, nodes=nodes)
        Pt_tran_core, Pt_tran_edge = \
            self.get_Ps_tran_node(Ut_core, Ut_edge, None, tau_et_core_edge, tau_et_edge_sol, None, vol, nodes=nodes)
        Pa_tran_core, Pa_tran_edge = \
            self.get_Ps_tran_node(Ua_core, Ua_edge, None, tau_ea_core_edge, tau_ea_edge_sol, None, vol, nodes=nodes)
        Pe_tran_core, Pe_tran_edge = \
            self.get_Ps_tran_node(Ue_core, Ue_edge, None, tau_ee_core_edge, tau_ee_edge_sol, None, vol, nodes=nodes)

        # External particle source:
        Sd_ext_core, Sd_ext_edge, St_ext_core, St_ext_edge, Sa_ext_core, Sa_ext_edge = self.get_Ss_ext_node(t)

        # Auxiliary heating:
        Pd_aux_core, Pd_aux_edge, Pt_aux_core, Pt_aux_edge, Pa_aux_core, Pa_aux_edge, Pe_aux_core, Pe_aux_edge \
            = self.get_Ps_aux_node(Te_core, Te_edge, t,
                                   nd_core=nd_core, nt_core=nt_core, nd_edge=nd_edge, nt_edge=nt_edge)

        # Ohmic heating:
        P_oh_core = self.get_P_oh_node(t, Te_node=Te_core, a=a, IP=IP, Z_eff=Z_eff_core)
        P_oh_edge = self.get_P_oh_node(t, Te_node=Te_edge, a=a, IP=IP, Z_eff=Z_eff_edge)

        # Fusion:
        Sd_fus_core, St_fus_core, Sa_fus_core = self.get_Ss_fus_node(nd_core, nt_core, Td_core, Tt_core)
        Sd_fus_edge, St_fus_edge, Sa_fus_edge = self.get_Ss_fus_node(nd_edge, nt_edge, Td_edge, Tt_edge)
        Pd_fus_core, Pt_fus_core, Pa_fus_core, Pe_fus_core = \
            self.get_Ps_fus_node_iter(nd_core, nt_core, na_core, Te_core, Sa_fus_core)
        Pd_fus_edge, Pt_fus_edge, Pa_fus_edge, Pe_fus_edge = \
            self.get_Ps_fus_node_iter(nd_edge, nt_edge, na_edge, Te_edge, Sa_fus_edge)

        # Collisional energy transfer:
        Qd_core, Qt_core, Qa_core, Qe_core, Qs_core = self.get_Qs_node(
            nd_core, nt_core, na_core, ne_core, Td_core, Tt_core, Ta_core, Te_core)
        Qd_edge, Qt_edge, Qa_edge, Qe_edge, Qs_edge = self.get_Qs_node(
            nd_edge, nt_edge, na_edge, ne_edge, Td_edge, Tt_edge, Ta_edge, Te_edge)

        # Radiation:
        P_ecr_core, P_ecr_edge = self.get_P_ecr_node(
            ne_core, ne_edge, Te_core, Te_edge, a, R0, B0, kappa, vol)
        P_rad_core, Ps_rad_core = self.get_P_rad_node_iter(
            nd_core + nt_core, naz_core, ne_core, Te_core, P_ecr_core, Zaz, Z_eff_core)
        P_rad_edge, Ps_rad_edge = self.get_P_rad_node_iter(
            nd_edge + nt_edge, naz_edge, ne_edge, Te_edge, P_ecr_edge, Zaz, Z_eff_edge)

        # IOL fractions:
        Fd_orb_edge, Ed_orb_edge = self.iol_d.get_loss_fractions(Td_edge)
        Ft_orb_edge, Et_orb_edge = self.iol_t.get_loss_fractions(Tt_edge)
        Fa_orb_edge, Ea_orb_edge = self.iol_a.get_loss_fractions(Ta_edge)

        # IOL terms:
        Sd_iol_edge = self.get_Ps_iol_node(
            nd_edge, Fd_orb_edge, self.iol_parameters['tau_ps_iol_edge'] * tau_pd_edge_sol)
        St_iol_edge = self.get_Ps_iol_node(
            nt_edge, Ft_orb_edge, self.iol_parameters['tau_ps_iol_edge'] * tau_pt_edge_sol)
        Sa_iol_edge = self.get_Ps_iol_node(
            na_edge, Fa_orb_edge, self.iol_parameters['tau_ps_iol_edge'] * tau_pa_edge_sol)
        Pd_iol_edge = self.get_Ps_iol_node(
            Ud_edge, Ed_orb_edge, self.iol_parameters['tau_es_iol_edge'] * tau_ed_edge_sol)
        Pt_iol_edge = self.get_Ps_iol_node(
            Ut_edge, Et_orb_edge, self.iol_parameters['tau_es_iol_edge'] * tau_et_edge_sol)
        Pa_iol_edge = self.get_Ps_iol_node(
            Ua_edge, Ea_orb_edge, self.iol_parameters['tau_es_iol_edge'] * tau_ea_edge_sol)

        # Delayed fusion:
        if self.delayed_fusion:
            Sd_core = Sd_ext_core + Sd_fus_core + Sd_tran_core
            St_core = St_ext_core + St_fus_core + St_tran_core
            Pd_core = Pd_aux_core + Pd_fus_core + Qd_core + Pd_tran_core
            Pt_core = Pt_aux_core + Pt_fus_core + Qt_core + Pt_tran_core
            dTd_core_dt = J2eV((Pd_core * 2 / 3 - eV2J(Td_core) * Sd_core) / nd_core)
            dTt_core_dt = J2eV((Pt_core * 2 / 3 - eV2J(Tt_core) * St_core) / nt_core)

            tau_se_core = self.get_tau_se(ne=ne_core, Te=Te_core, m_b=m_a, Z_b=Z_a)
            Td_core_delay = self.get_Ts_node_delay(Td_core, dTd_core_dt, tau_se_core)
            Tt_core_delay = self.get_Ts_node_delay(Tt_core, dTt_core_dt, tau_se_core)
            _, _, Sa_fus_core_delay = \
                self.get_Ss_fus_node(nd_core, nt_core, Td_core_delay, Tt_core_delay)
            Pd_fus_core, Pt_fus_core, Pa_fus_core, _ = \
                self.get_Ps_fus_node_iter(nd_core, nt_core, na_core, Te_core, Sa_fus_core_delay)

        # Nodal terms:
        Sd = (Sd_ext_core, Sd_ext_edge, Sd_fus_core, Sd_fus_edge, Sd_tran_core, Sd_tran_edge, Sd_iol_edge)
        St = (St_ext_core, St_ext_edge, St_fus_core, St_fus_edge, St_tran_core, St_tran_edge, St_iol_edge)
        Sa = (Sa_ext_core, Sa_ext_edge, Sa_fus_core, Sa_fus_edge, Sa_tran_core, Sa_tran_edge, Sa_iol_edge)

        Pd = (Pd_aux_core, Pd_aux_edge, Pd_fus_core, Pd_fus_edge, Qd_core, Qd_edge, Pd_tran_core, Pd_tran_edge,
              Pd_iol_edge)
        Pt = (Pt_aux_core, Pt_aux_edge, Pt_fus_core, Pt_fus_edge, Qt_core, Qt_edge, Pt_tran_core, Pt_tran_edge,
              Pt_iol_edge)
        Pa = (Pa_aux_core, Pa_aux_edge, Pa_fus_core, Pa_fus_edge, Qa_core, Qa_edge, Pa_tran_core, Pa_tran_edge,
              Pa_iol_edge)
        Pe = (Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, Pe_fus_core, Pe_fus_edge, P_rad_core, P_rad_edge,
              Qe_core, Qe_edge, Pe_tran_core, Pe_tran_edge)
        Ps = (*Qs_core, *Qs_edge, *Ps_rad_core, *Ps_rad_edge)

        return Sd, St, Sa, Pd, Pt, Pa, Pe, Ps

    def system(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Dynamical system for the multi-nodal model in ITER

        :param t: time [s]
        :param y: nd_core, nt_core, na_core, nd_edge, nt_edge, na_edge [10^19 m^-3],
                  Ud_core, Ut_core, Ua_core, Ue_core, Ud_edge, Ut_edge, Ua_edge, Ue_edge [10^19 keV/m^3]
        :return: dy/dt = [Sd_core, St_core, Sa_core, Sd_edge, St_edge, Sa_edge,
                          Pd_core, Pt_core, Pa_core, Pe_core, Pd_edge, Pt_edge, Pa_edge, Pe_edge]
        """
        self.system_count += 1

        y = torch.abs(y)
        nd_core = y[0] * 1E19  # [10^19 m^-3] -> [m^-3]
        nt_core = y[1] * 1E19
        na_core = y[2] * 1E19
        nd_edge = y[3] * 1E19
        nt_edge = y[4] * 1E19
        na_edge = y[5] * 1E19
        Ud_core = keV2J(y[6] * 1E19)  # [10^19 keV] -> [J]
        Ut_core = keV2J(y[7] * 1E19)
        Ua_core = keV2J(y[8] * 1E19)
        Ue_core = keV2J(y[9] * 1E19)
        Ud_edge = keV2J(y[10] * 1E19)
        Ut_edge = keV2J(y[11] * 1E19)
        Ua_edge = keV2J(y[12] * 1E19)
        Ue_edge = keV2J(y[13] * 1E19)

        Sd, St, Sa, Pd, Pt, Pa, Pe, _ = \
            self.get_sources_iter((nd_core, nd_edge), (nt_core, nt_edge), (na_core, na_edge),
                                  (Ud_core, Ud_edge), (Ut_core, Ut_edge), (Ua_core, Ua_edge), (Ue_core, Ue_edge), t)

        # Particle source terms:
        Sd_ext_core, Sd_ext_edge, Sd_fus_core, Sd_fus_edge, Sd_tran_core, Sd_tran_edge, Sd_iol_edge = Sd
        St_ext_core, St_ext_edge, St_fus_core, St_fus_edge, St_tran_core, St_tran_edge, St_iol_edge = St
        Sa_ext_core, Sa_ext_edge, Sa_fus_core, Sa_fus_edge, Sa_tran_core, Sa_tran_edge, Sa_iol_edge = Sa

        # Energy source terms:
        Pd_aux_core, Pd_aux_edge, Pd_fus_core, Pd_fus_edge, Qd_core, Qd_edge, Pd_tran_core, Pd_tran_edge, Pd_iol_edge \
            = Pd
        Pt_aux_core, Pt_aux_edge, Pt_fus_core, Pt_fus_edge, Qt_core, Qt_edge, Pt_tran_core, Pt_tran_edge, Pt_iol_edge \
            = Pt
        Pa_aux_core, Pa_aux_edge, Pa_fus_core, Pa_fus_edge, Qa_core, Qa_edge, Pa_tran_core, Pa_tran_edge, Pa_iol_edge \
            = Pa
        Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, Pe_fus_core, Pe_fus_edge, P_rad_core, P_rad_edge, \
        Qe_core, Qe_edge, Pe_tran_core, Pe_tran_edge = Pe

        # Particle balance equations:
        Sd_core = Sd_ext_core + Sd_fus_core + Sd_tran_core
        St_core = St_ext_core + St_fus_core + St_tran_core
        Sa_core = Sa_ext_core + Sa_fus_core + Sa_tran_core
        Sd_edge = Sd_ext_edge + Sd_fus_edge + Sd_tran_edge + Sd_iol_edge
        St_edge = St_ext_edge + St_fus_edge + St_tran_edge + St_iol_edge
        Sa_edge = Sa_ext_edge + Sa_fus_edge + Sa_tran_edge + Sa_iol_edge

        # Energy balance equations:
        Pd_core = Pd_aux_core + Pd_fus_core + Qd_core + Pd_tran_core
        Pt_core = Pt_aux_core + Pt_fus_core + Qt_core + Pt_tran_core
        Pa_core = Pa_aux_core + Pa_fus_core + Qa_core + Pa_tran_core
        Pe_core = P_oh_core + Pe_aux_core + Pe_fus_core - P_rad_core + Qe_core + Pe_tran_core
        Pd_edge = Pd_aux_edge + Pd_fus_edge + Qd_edge + Pd_tran_edge + Pd_iol_edge
        Pt_edge = Pt_aux_edge + Pt_fus_edge + Qt_edge + Pt_tran_edge + Pt_iol_edge
        Pa_edge = Pa_aux_edge + Pa_fus_edge + Qa_edge + Pa_tran_edge + Pa_iol_edge
        Pe_edge = P_oh_edge + Pe_aux_edge + Pe_fus_edge - P_rad_edge + Qe_edge + Pe_tran_edge

        dy_dt = torch.zeros_like(y)
        dy_dt[0] = Sd_core / 1E19  # [m^-3] -> [10^19 m^-3]
        dy_dt[1] = St_core / 1E19
        dy_dt[2] = Sa_core / 1E19
        dy_dt[3] = Sd_edge / 1E19
        dy_dt[4] = St_edge / 1E19
        dy_dt[5] = Sa_edge / 1E19
        dy_dt[6] = J2keV(Pd_core) / 1E19  # [J] -> [10^19 keV]
        dy_dt[7] = J2keV(Pt_core) / 1E19
        dy_dt[8] = J2keV(Pa_core) / 1E19
        dy_dt[9] = J2keV(Pe_core) / 1E19
        dy_dt[10] = J2keV(Pd_edge) / 1E19
        dy_dt[11] = J2keV(Pt_edge) / 1E19
        dy_dt[12] = J2keV(Pa_edge) / 1E19
        dy_dt[13] = J2keV(Pe_edge) / 1E19

        return dy_dt


if __name__ == '__main__':
    pass
    # reactor = ReactorITER(2, delayed_fusion=True)
    # t = reactor.time[0]
    #
    # nd_core = reactor.get_nd_node(t, node='core')
    # nd_edge = reactor.get_nd_node(t, node='edge')
    # nt_core = reactor.get_nt_node(t, node='core')
    # nt_edge = reactor.get_nt_node(t, node='edge')
    # na_core = reactor.get_na_node(t, node='core')
    # na_edge = reactor.get_na_node(t, node='edge')
    # ne_core = reactor.get_ne_node(t, node='core')
    # ne_edge = reactor.get_ne_node(t, node='edge')
    #
    # Td_core = reactor.get_Td_node(t, node='core')
    # Td_edge = reactor.get_Td_node(t, node='edge')
    # Tt_core = reactor.get_Tt_node(t, node='core')
    # Tt_edge = reactor.get_Tt_node(t, node='edge')
    # Te_core = reactor.get_Te_node(t, node='core')
    # Te_edge = reactor.get_Te_node(t, node='edge')
    # Ta_core = reactor.get_Ta_node(t, node='core')
    # Ta_edge = reactor.get_Ta_node(t, node='edge')
    #
    # Ud_core = 3 / 2 * nd_core * eV2J(Td_core)
    # Ud_edge = 3 / 2 * nd_edge * eV2J(Td_edge)
    # Ut_core = 3 / 2 * nt_core * eV2J(Tt_core)
    # Ut_edge = 3 / 2 * nt_edge * eV2J(Tt_edge)
    # Ua_core = 3 / 2 * na_core * eV2J(Ta_core)
    # Ua_edge = 3 / 2 * na_edge * eV2J(Ta_edge)
    # Ue_core = 3 / 2 * ne_core * eV2J(Te_core)
    # Ue_edge = 3 / 2 * ne_edge * eV2J(Te_edge)
    #
    # sources = reactor.get_sources_iter((nd_core, nd_edge), (nt_core, nt_edge), (na_core, na_edge),
    #                                    (Ud_core, Ud_edge), (Ut_core, Ut_edge), (Ua_core, Ua_edge),
    #                                    (Ue_core, Ue_edge), t)
    # print(sources)
