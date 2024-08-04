""""
Reactions
"""
import scipy.constants as sc
import torch
from torch import sqrt, exp, log
from torchinterp1d import Interp1d

Ry = sc.value('Rydberg constant times hc in eV')


class Reaction(object):
    """
    Reactions
    """

    def __init__(self):
        self.interp1d = Interp1d()

    def get_fusion_coefficient(self, Ti: torch.Tensor, reaction_type='tdna') -> torch.Tensor:
        """
        Fusion reaction rate coefficient from Bosch-Hale model

        :param Ti: ion temperature [eV]
        :param reaction_type: fusion reaction type in {'tdna', 'hdpa', 'ddpt', 'ddnh'}
        :return: sigma_v_fus [m^3/s]
        """
        T = Ti * 1E-3  # [keV]

        if reaction_type == 'tdna':
            B_G = 34.3827
            mrc2 = 1124656

            C1 = 1.17302E-9
            C2 = 1.51361E-2
            C3 = 7.51886E-2
            C4 = 4.60643E-3
            C5 = 1.35000E-2
            C6 = -1.06750E-4
            C7 = 1.36600E-5
        elif reaction_type == 'hdpa':
            B_G = 68.7508
            mrc2 = 1124572

            C1 = 5.51036E-10
            C2 = 6.41918E-3
            C3 = -2.02896E-3
            C4 = -1.91080E-5
            C5 = 1.35776E-4
            C6 = 0
            C7 = 0
        elif reaction_type == 'ddpt':
            B_G = 31.3970
            mrc2 = 937814

            C1 = 5.65718E-12
            C2 = 3.41267E-3
            C3 = 1.99167E-3
            C4 = 0
            C5 = 1.05060E-5
            C6 = 0
            C7 = 0
        elif reaction_type == 'ddnh':
            B_G = 31.3970
            mrc2 = 937814

            C1 = 5.43360E-12
            C2 = 5.85778E-3
            C3 = 7.68222E-3
            C4 = 0
            C5 = -2.96400E-6
            C6 = 0
            C7 = 0
        else:
            raise KeyError('fusion reaction type not in {tdna, hdpa, ddpt, ddnh}')

        theta = T / (1 - (T * (C2 + T * (C4 + T * C6))) /
                     (1 + T * (C3 + T * (C5 + T * C7))))
        xi = (B_G ** 2 / (4 * theta)) ** (1 / 3)
        sigma_v_fus = 1e-6 * C1 * theta * sqrt(xi / (mrc2 * T ** 3)) * exp(-3 * xi)
        sigma_v_fus = torch.nan_to_num(sigma_v_fus, nan=0)

        return sigma_v_fus

    def get_fusion_energy(self, reaction_type='tdna'):
        """
        Thermonuclear energy deposition to plasmas

        :param reaction_type: reaction name in {'tdna', 'hdpa', 'ddpt', 'ddnh'}
        :return: E_fus [eV]
        """
        if reaction_type == 'tdna':
            E_fus = (17.6 - 14.1) * 1E6
        elif reaction_type == 'ddpt':
            E_fus = 4.0E6
        elif reaction_type == 'ddnh':
            E_fus = (3.25 - 2.5) * 1E6
        elif reaction_type == 'hdpa':
            E_fus = 18.2E6
        else:
            E_fus = 0
        return E_fus

    def polyval(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a polynomial at specific values

        :param p: 1D array of polynomial coefficients (including coefficients equal to zero)
        from highest degree to the constant term
        :param x: A number, an array of numbers, or an instance of poly1d, at which to evaluate p
        :return: values
        """
        assert p.ndim == 1
        values = p[0] * torch.ones_like(x)
        for i in range(1, p.shape[0]):
            values = values * x + p[i]
        return values

    def get_excitation_coefficient(self, Te: torch.Tensor) -> torch.Tensor:
        """
        Excitation reaction rate coefficient

        Reaction: e + H (1s) -> e + H* (2p)

        :param Te: electron temperature [eV]
        :return: sigma_v_exc [m^3/s]
        """
        b0 = -2.814949375869e+01
        b1 = 1.009828023274e+01
        b2 = -4.771961915818e+00
        b3 = 1.467805963618e+00
        b4 = -2.979799374553e-01
        b5 = 3.861631407174e-02
        b6 = -3.051685780771e-03
        b7 = 1.335472720988e-04
        b8 = -2.476088392502e-06

        p = torch.tensor([b8, b7, b6, b5, b4, b3, b2, b1, b0])
        sigma_v_exc = exp(self.polyval(p, log(Te))) * 1e-6  # [cm^3/s] -> [m^3/s]
        return sigma_v_exc

    def get_excitation_energy(self):
        """
        Excitation energy

        Reaction: e + H (1s) -> e + H* (2p)

        :return: E_exc [eV]
        """
        E_exc = 10.2
        return E_exc

    def get_ionization_coefficient(self, Te: torch.Tensor) -> torch.Tensor:
        """
        Ionization reaction rate coefficient

        Reaction: e + H (1s) -> e + H+ + e

        :param Te: electron temperature [eV]
        :return: sigma_v_ion [m^3/s]
        """
        b0 = -3.271396786375e+01
        b1 = 1.353655609057e+01
        b2 = -5.739328757388e+00
        b3 = 1.563154982022e+00
        b4 = -2.877056004391e-01
        b5 = 3.482559773737e-02
        b6 = -2.631976175590e-03
        b7 = 1.119543953861e-04
        b8 = -2.039149852002e-06

        p = torch.tensor([b8, b7, b6, b5, b4, b3, b2, b1, b0])
        sigma_v_ion = exp(self.polyval(p, log(Te))) * 1e-6  # [cm^3/s] -> [m^3/s]
        return sigma_v_ion

    def get_ionization_energy(self):
        """
        Ionization energy

        Reaction: e + H (1s) -> e + H+ + e

        :return: E_ion [eV]
        """
        E_ion = 13.6
        return E_ion

    def get_recombination_coefficient(self, Te: torch.Tensor, nl: str = '1s') -> torch.Tensor:
        """
        Recombination reaction rate coefficient

        Reaction: e + H+ -> H(n) + hv

        :param Te: electron temperature [eV]
        :param nl: energy level
        :return: sigma_v_rec [m^3/s]
        """
        n = int(nl[0])
        E_ion_n = Ry / n ** 2
        beta_n = E_ion_n / Te

        if nl in ['1s', '2s', '2p']:
            if nl == '1s':
                A_nl = 3.92
                chi_nl = 0.35
            elif nl == '2s':
                A_nl = 2.47
                chi_nl = 0.12
            else:  # nl == '2p'
                A_nl = 6.22
                chi_nl = 0.61
            sigma_v_rec = A_nl * 1e-14 * (E_ion_n / Ry) ** 0.5 * beta_n ** 1.5 / (beta_n + chi_nl)  # [cm^3/s]
        elif n >= 3:
            gamma = 1.78
            exp_beta_E1_beta = lambda beta: log(1 + (1 / (gamma * beta)) * (1 + 1.4 * gamma * beta) / (1 + 1.4 * beta))
            sigma_v_rec = 5.201e-14 * beta_n ** 1.5 * exp_beta_E1_beta(beta_n)  # [cm^3/s]
        else:
            raise KeyError('nl not in [1s, 2s, 2p] nor n >= 3')

        sigma_v_rec *= 1e-6  # [cm^3/s] -> [m^3/s]
        return sigma_v_rec

    def get_recombination_energy(self, nl: str = '1s'):
        """
        Recombination energy

        Reaction: e + H+ -> H(n) + hv

        :param nl: energy level
        :return: E_rec [eV]
        """
        n = int(nl[0])
        E_rec = Ry / n ** 2
        return E_rec

    def get_charge_exchange_coefficient(self, Ti: torch.Tensor) -> torch.Tensor:
        """
        Charge exchange reaction rate coefficient (Ta = 1 eV)

        Reaction: D+ + D0 -> D0 + D+

        :param Ti: ion temperature [eV]
        :return: sigma_v_cx [m^3/s]
        """
        Ti_tensor = torch.tensor([1, 10, 100])  # [eV]
        sigma_v_cx_tensor = torch.tensor([0.8, 1.2, 2.8]) * 1e-14  # sigma_v_cx [m^3/s] for Ta = 1 eV
        sigma_v_cx = torch.abs(self.interp1d(Ti_tensor, sigma_v_cx_tensor, torch.reshape(Ti, (-1,))).squeeze())
        return sigma_v_cx

    def get_elastic_scattering_coefficient(self, Ti: torch.Tensor) -> torch.Tensor:
        """
        Elastic scattering reaction rate coefficient (Ta = 1 eV)

        Reaction: D+ + D0 -> D+ + D0

        :param Ti: ion temperature [eV]
        :return: sigma_v_el [m^3/s]
        """
        Ti_tensor = torch.tensor([1, 10, 100])  # [eV]
        sigma_v_el_tensor = torch.tensor([1.1, 1.5, 1.4]) * 1e-14  # sigma_v_el [m^3/s] for Ta = 1 eV
        sigma_v_el = torch.abs(self.interp1d(Ti_tensor, sigma_v_el_tensor, torch.reshape(Ti, (-1,))).squeeze())
        return sigma_v_el


if __name__ == '__main__':
    pass
    # reaction = Reaction()
