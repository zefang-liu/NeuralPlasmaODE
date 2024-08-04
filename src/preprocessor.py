"""
Data Preprocessor
"""
import errno
import os

import numpy as np
import torch
import xarray as xr

from src import config
from src.utils import Z_a, Z_imp, array2tensor


class Preprocessor(object):
    """
    Data Preprocessor
    """

    def __init__(self, average=True, verbose=False):
        """
        Initialize the preprocessor

        :param average: True for rolling average
        :param verbose: True for verbose
        """
        self.average = average
        self.verbose = verbose

        self.signals_1d = config.signals_1d
        self.signals_1d_gas = config.signals_1d_gas
        self.signals_1d_vol = config.signals_1d_vol
        self.signals_2d = config.signals_2d
        self.signals_2d_var = config.signals_2d_var
        self.rho_core = config.rho_core
        self.rho_edge = config.rho_edge
        self.rho_sol = config.rho_sol
        self.device = config.device
        self.window = config.window
        self.time_step = config.time_step
        self.max_time_length = config.max_time_length

    def preprocess(self, shot_num: int) -> dict:
        """
        Preprocess data

        :param shot_num: shot number
        :return: shot
        """
        path_shot = os.path.join('data', str(shot_num))
        if not os.path.exists(path_shot):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_shot)

        shot_xr_1d = {}
        shot_xr_2d = {}

        # Preprocess 1D signals
        for signal_name in self.signals_1d:
            path_signal = os.path.join(path_shot, signal_name + '.nc')
            if not os.path.exists(path_signal):
                if self.verbose:
                    print('Shot #{:d} {:s} missed'.format(shot_num, signal_name))
                continue
            signal = xr.open_dataarray(path_signal).rename({'dim_0': 'time'}).fillna(0)
            shot_xr_1d[signal_name] = signal

        self.get_gas(shot_num, path_shot, shot_xr_1d)

        # Preprocess 2D signals
        for signal_name in self.signals_2d:
            path_signal = os.path.join(path_shot, signal_name + '.nc')
            if not os.path.exists(path_signal):
                if self.verbose:
                    print('Shot #{:d} {:s} missed'.format(shot_num, signal_name))
                continue
            signal = xr.open_dataarray(path_signal).rename({'dim_0': 'rho', 'dim_1': 'time'}).fillna(0)
            self.normalize_units(signal)
            shot_xr_2d[signal_name] = signal

        self.get_nt(shot_xr_1d, shot_xr_2d)

        # Interpolate by time and transfer to tensors
        time = self.get_time(shot_xr_1d)
        time_tensor = array2tensor(time / 1E3)
        shot = {'num': shot_num, 'time': time_tensor}

        for signal_name, signal in shot_xr_1d.items():
            signal = self.get_rolling_mean(signal.interp(time=time))
            shot[signal_name] = array2tensor(signal.values)

        for signal_name in self.signals_1d_vol:
            if shot[signal_name].size()[0] == 0:
                shot[signal_name] = torch.zeros_like(shot['volume'])
            shot[signal_name + '_vol'] = shot[signal_name] / shot['volume']

        shot['bt0'] = torch.abs(shot['bt0'])
        shot['time'] = array2tensor(self.get_rolling_mean(shot_xr_1d['bt0'].interp(time=time)).time.values / 1E3)
        return shot

    def get_time(self, shot_xr_1d: dict):
        """
        Get time

        :param shot_xr_1d: shot of 1d x-arrays
        :return: time [s]
        """
        time, time_min, time_max = None, None, None

        # Get the intersection of time intervals
        for signal_name in self.signals_2d_var:
            signal = shot_xr_1d[signal_name]
            if time is None:
                time = signal.time.values
                time.sort()
                time_min = time[0]
                time_max = time[-1]
            else:
                time_min = np.max([np.min(signal.time.values), time_min])
                time_max = np.min([np.max(signal.time.values), time_max])

        assert time_min <= time_max
        if self.max_time_length is not None:
            time_max = min(time_max, time_min + self.max_time_length * 1E3)
        time_raw = np.arange(np.floor(time_min / 1E3), np.ceil(time_max / 1E3) + 1, self.time_step) * 1E3
        time = time_raw[np.min(np.where(time_min <= time_raw)):(np.max(np.where(time_raw <= time_max)) + 1)]
        return time

    def get_rolling_mean(self, signal: xr.DataArray):
        """
        Get rolling average

        :param signal: signal
        :return: signal
        """
        if self.average:
            signal = signal.rolling(time=self.window, center=True).mean().dropna(dim='time')
        return signal

    def get_gas(self, shot_num: int, path_shot: str, shot_xr_1d: dict):
        """
        Get gas puffing

        :param shot_num: shot number
        :param path_shot: path of the shot
        :param shot_xr_1d: shot of 1d x-arrays
        :return:
        """
        gas = None
        for signal_name in self.signals_1d_gas:
            path_signal = os.path.join(path_shot, signal_name + '.nc')
            if not os.path.exists(path_signal):
                if self.verbose:
                    print('Shot #{:d} {:s} missed'.format(shot_num, signal_name))
                continue

            signal = xr.open_dataarray(path_signal).rename({'dim_0': 'time'}).fillna(0)
            if gas is None:
                gas = signal
            else:
                signal = signal.interp(time=gas.time)
                gas += signal

        shot_xr_1d['gas'] = self.get_rolling_mean(gas)

    def get_nt(self, shot_xr_1d: dict, shot_xr_2d: dict):
        """
        Set up volume-averaged densities and temperatures

        :param shot_xr_1d: shot of 1d x-arrays
        :param shot_xr_2d: shot of 2d x-arrays
        :return:
        """
        for i, signal_2d_var in enumerate(self.signals_2d_var):
            shot_xr_1d[signal_2d_var] = self.get_vol_average(shot_xr_2d, self.signals_2d[i], 0, 1)
            shot_xr_1d[signal_2d_var + '_core'] = \
                self.get_vol_average(shot_xr_2d, self.signals_2d[i], 0, self.rho_core)
            shot_xr_1d[signal_2d_var + '_edge'] = \
                self.get_vol_average(shot_xr_2d, self.signals_2d[i], self.rho_core, self.rho_edge)
            shot_xr_1d[signal_2d_var + '_sol'] = \
                self.get_vol_average(shot_xr_2d, self.signals_2d[i], self.rho_edge, self.rho_sol)

    def normalize_units(self, signal: xr.DataArray):
        """
        Normalize some units

        :param signal: signal
        :return: None
        """
        if '10^19 m^-3' in signal.attrs['units']:
            signal.values *= 1E19
            signal.attrs['units'] = 'm^-3'
        elif 'keV' in signal.attrs['units']:
            signal.values *= 1E3
            signal.attrs['units'] = 'eV'

    def get_vol_average(self, shot: dict, signal_name: str, rho0=0.0, rho1=1.0):
        """
        Get volume average for one signal from rho_0 to rho_1

        :param shot: shot dictionary
        :param signal_name: signal name
        :param rho0: start rho
        :param rho1: end rho
        :return: signal_avg
        """
        rho = shot[signal_name].rho.sel(rho=slice(rho0, rho1))
        signal = shot[signal_name].sel(rho=rho)
        signal_avg = 2 * (rho * signal).integrate('rho') / (rho1 ** 2 - rho0 ** 2)
        return signal_avg


class PreprocessorITER(Preprocessor):
    """
    Data Preprocessor for ITER
    """

    def __init__(self):
        """
        Initialize the preprocessor for ITER
        """
        super().__init__()
        self.signals_2d_var_iter = config.signals_2d_var_iter
        self.number2tensor = lambda number, t: torch.ones_like(t) * number
        self.scalar2tensor = lambda scalar: torch.tensor(scalar).double().to(self.device)

    def preprocess(self, scenario: int) -> dict:
        """
        Preprocess data

        :param scenario: scenario number
        :return: shot
        """
        shot = {}

        if scenario == 1:  # inductive scenario
            # Time steps:
            total_time = 500  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.2, t)  # [m]
            shot['aminor'] = self.number2tensor(2.0, t)  # [m]
            shot['volume'] = self.number2tensor(831, t)  # [m^3]
            shot['kappa95'] = self.number2tensor(1.7, t)
            shot['delta95'] = self.number2tensor(0.33, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.3, t)  # [T]
            shot['ip'] = self.number2tensor(15E6, t)  # [A]
            shot['q95'] = self.number2tensor(3.0, t)

            # Powers:
            shot['poh'] = self.number2tensor(1, t)  # [W], to be calculated from the plasma current
            shot['pnbi'] = self.number2tensor(33E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(0, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(17E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.0014}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 3.616E18, 'ne': 11.3E19, 'ti': 8.1E3, 'te': 8.9E3})

        elif scenario == 2:  # inductive scenario
            # Time steps:
            total_time = 400  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.2, t)  # [m]
            shot['aminor'] = self.number2tensor(2.0, t)  # [m]
            shot['volume'] = self.number2tensor(831, t)  # [m^3]
            shot['kappa0'] = self.number2tensor(1.85, t)
            shot['kappa95'] = self.number2tensor(1.7, t)
            shot['delta95'] = self.number2tensor(0.33, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.3, t)  # [T]
            shot['ip'] = self.number2tensor(15E6, t)  # [A]
            shot['q0'] = self.number2tensor(1.0, t)
            shot['q95'] = self.number2tensor(3.0, t)

            # Powers:
            shot['poh'] = self.number2tensor(1, t)  # [W], to be calculated from the plasma current
            shot['pnbi'] = self.number2tensor(33E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(0, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(7E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.0012}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 3.232E18, 'ne': 10.1E19, 'ti': 8.0E3, 'te': 8.8E3})

        elif scenario == 3:  # hybrid scenario
            # Time steps:
            total_time = 1070  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.2, t)  # [m]
            shot['aminor'] = self.number2tensor(2.0, t)  # [m]
            shot['volume'] = self.number2tensor(831, t)  # [m^3]
            shot['kappa95'] = self.number2tensor(1.7, t)
            shot['delta95'] = self.number2tensor(0.33, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.3, t)  # [T]
            shot['ip'] = self.number2tensor(13.8E6, t)  # [A]
            shot['q95'] = self.number2tensor(3.3, t)

            # Powers:
            shot['poh'] = self.number2tensor(1, t)  # [W], to be calculated from the plasma current
            shot['pnbi'] = self.number2tensor(33E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(20E6, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(20E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.0019}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 2.325E18, 'ne': 9.3E19, 'ti': 8.4E3, 'te': 9.6E3})

        elif scenario == 4:  # non-inductive weak negative shear (WNS) scenario
            # Time steps:
            total_time = 2000  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.35, t)  # [m]
            shot['aminor'] = self.number2tensor(1.85, t)  # [m]
            shot['volume'] = self.number2tensor(794, t)  # [m^3]
            shot['kappa95'] = self.number2tensor(1.85, t)
            shot['delta95'] = self.number2tensor(0.40, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.18, t)  # [T]
            shot['ip'] = self.number2tensor(9E6, t)  # [A]
            shot['q95'] = self.number2tensor(5.3, t)

            # Powers:
            shot['poh'] = self.number2tensor(0, t)  # [W]
            shot['pnbi'] = self.number2tensor(30E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(0, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(0, t)  # [W]
            shot['plh'] = self.number2tensor(29E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol + ['plh']:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.0026}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 2.747E18, 'ne': 6.7E19, 'ti': 12.5E3, 'te': 12.3E3}, is_inductive=False)

        elif scenario == 6:  # non-inductive strong negative shear (SNS) scenario
            # Time steps:
            total_time = 2000  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.35, t)  # [m]
            shot['aminor'] = self.number2tensor(1.85, t)  # [m]
            shot['volume'] = self.number2tensor(798, t)  # [m^3]
            shot['kappa95'] = self.number2tensor(1.86, t)
            shot['delta95'] = self.number2tensor(0.41, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.18, t)  # [T]
            shot['ip'] = self.number2tensor(9E6, t)  # [A]
            shot['q95'] = self.number2tensor(5.4, t)

            # Powers:
            shot['poh'] = self.number2tensor(0, t)  # [W]
            shot['pnbi'] = self.number2tensor(20E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(0, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(0, t)  # [W]
            shot['plh'] = self.number2tensor(40E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol + ['plh']:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.002}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 2.6E18, 'ne': 6.5E19, 'ti': 12.1E3, 'te': 13.3E3}, is_inductive=False)

        elif scenario == 7:  # non-inductive weak positive shear (WPS) scenario
            # Time steps:
            total_time = 2000  # [s]
            t = self.get_time_iter(total_time)
            shot['time'] = t

            # Geometries:
            shot['r0'] = self.number2tensor(6.35, t)  # [m]
            shot['aminor'] = self.number2tensor(1.85, t)  # [m]
            shot['volume'] = self.number2tensor(798, t)  # [m^3]
            shot['kappa95'] = self.number2tensor(1.86, t)
            shot['delta95'] = self.number2tensor(0.41, t)

            # Electromagnetic parameters:
            shot['bt0'] = self.number2tensor(5.18, t)  # [T]
            shot['ip'] = self.number2tensor(9E6, t)  # [A]
            shot['q95'] = self.number2tensor(5.3, t)

            # Powers:
            shot['poh'] = self.number2tensor(0, t)  # [W]
            shot['pnbi'] = self.number2tensor(28E6, t)  # [W]
            shot['echpwrc'] = self.number2tensor(0, t)  # [W]
            shot['ichpwrc'] = self.number2tensor(0, t)  # [W]
            shot['plh'] = self.number2tensor(29E6, t)  # [W]
            shot['gas'] = self.number2tensor(0, t)  # [Torr-L/s], ignored

            for signal in self.signals_1d_vol + ['plh']:
                shot[signal + '_vol'] = shot[signal] / shot['volume']

            # Impurities:
            shot['impurity_fractions'] = {'be': 0.02, 'ar': 0.0023}  # fractions of electron densities

            # Profiles:
            self.get_profiles(shot, {'na': 2.68E18, 'ne': 6.7E19, 'ti': 12.5E3, 'te': 12.1E3}, is_inductive=False)

        else:
            raise KeyError('The scenario {:d} does not exist.'.format(scenario))

        # ECR parameters:
        if scenario in [1, 2, 3]:
            ecr_parameters = {'alpha_ne': 0.037, 'alpha_te': 1.027, 'beta_te': 1.194, 'r': 0.8}
        elif scenario in [4, 6, 7]:
            ecr_parameters = {'alpha_ne': 0.102, 'alpha_te': 4.079, 'beta_te': 3.278, 'r': 0.8}
        else:
            raise KeyError('The scenario {:d} does not have ECR parameters.'.format(scenario))
        ecr_parameters = {key: self.scalar2tensor(value) for key, value in ecr_parameters.items()}
        shot['ecr_parameters'] = self.get_ecr_parameters(ecr_parameters, shot['rhos_core'], shot['rhos_edge'])
        shot['impurity_fractions'] = {key: self.scalar2tensor(value)
                                      for key, value in shot['impurity_fractions'].items()}

        # IOL parameters:
        iol_parameters = {'tau_ps_iol_edge': 1.0, 'tau_es_iol_edge': 1.0}
        shot['iol_parameters'] = {key: self.scalar2tensor(value) for key, value in iol_parameters.items()}

        return shot

    def get_time_iter(self, total_time):
        """
        Get time

        :param total_time: total time [s]
        :return: t [s]
        """
        if self.max_time_length is not None:
            total_time = min(total_time, self.max_time_length)
        t = array2tensor(np.linspace(0, total_time, int(total_time / self.time_step + 1)))
        return t

    def get_nd(self, na, ne, impurity_fractions: dict):
        """
        Get deuteron densities from the charge neutrality

        :param na: alpha particle density [m^-3]
        :param ne: electron density [m^-3]
        :param impurity_fractions: impurity fractions
        :return: nd [m^-3]
        """
        nd = ne - Z_a * na
        for impurity, impurity_fraction in impurity_fractions.items():
            nd -= Z_imp[impurity] * impurity_fraction * ne
        return nd / 2

    def get_vol_average_iter(self, profile, rho, rho0=0.0, rho1=1.0):
        """
        Get volume average for one signal from rho_0 to rho_1

        :param profile: profile
        :param rho: normalized radius
        :param rho0: start rho
        :param rho1: end rho
        :return: signal_avg
        """
        index = torch.nonzero((rho0 <= rho) & (rho <= rho1)).squeeze()
        signal_avg = 2 * torch.trapz(rho[index] * profile[index], rho[index]) / (rho1 ** 2 - rho0 ** 2)
        return signal_avg, rho[index], profile[index] / signal_avg

    def get_profiles(self, shot: dict, vol_avg: dict, is_inductive: bool = True):
        """
        Get density and temperature profiles

        :param shot: shot
        :param vol_avg: volume averages of signals
        :param is_inductive: True for an inductive scenario
        :return: None
        """
        t = shot['time']
        profiles = {}
        rho = array2tensor(np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]))

        if is_inductive:
            profiles['na'] = 1E18 * array2tensor(np.array(
                [4.48, 4.45, 4.29, 4.1, 3.9, 3.68, 3.48, 3.23, 3.06, 2.94, 2.87, 2.84, 2.81, 2.74, 2.42]))
            profiles['ne'] = 1E19 * array2tensor(np.array(
                [10.42, 10.42, 10.42, 10.42, 10.42, 10.39, 10.32, 10.29, 10.16, 10.0, 9.9, 9.81, 9.55, 8.77, 6.26]))
            profiles['ti'] = 1E3 * array2tensor(np.array(
                [19.27, 18.71, 17.42, 15.65, 13.63, 11.77, 9.44, 7.26, 5.24, 3.87, 3.63, 3.23, 2.9, 2.58, 2.18]))
            profiles['te'] = 1E3 * array2tensor(np.array(
                [23.15, 22.02, 20.4, 18.23, 15.56, 13.15, 10.32, 7.66, 5.24, 3.87, 3.55, 3.23, 2.9, 2.26, 0.97]))

        else:
            profiles['ne'] = 1E19 * array2tensor(np.array(
                [7.1, 7.07, 7.07, 7.06, 7.03, 6.99, 6.91, 6.79, 6.65, 6.35, 6.15, 5.87, 5.28, 4.56, 3.23]))
            profiles['nd'] = 1E19 * array2tensor(np.array(
                [5.38, 5.38, 5.38, 5.42, 5.44, 5.49, 5.57, 5.48, 5.38, 5.14, 5.0, 4.75, 4.42, 3.71, 2.5]))
            profiles['na'] = 1E19 * array2tensor(np.array(
                [0.39, 0.39, 0.35, 0.35, 0.32, 0.29, 0.24, 0.19, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
            profiles['ti'] = 1E3 * array2tensor(np.array(
                [33.23, 33.04, 32.39, 29.98, 25.38, 19.44, 12.02, 5.1, 2.92, 1.62, 1.11, 0.79, 0.51, 0.37, 0.28]))
            profiles['te'] = 1E3 * array2tensor(np.array(
                [30.12, 30.16, 29.65, 27.94, 24.5, 19.72, 13.09, 5.85, 3.29, 1.62, 1.11, 0.79, 0.51, 0.37, 0.28]))

        for signal in self.signals_2d_var_iter:
            if signal in vol_avg:
                signal_avg, _, _ = self.get_vol_average_iter(profiles[signal], rho, 0, self.rho_edge)
                profiles[signal] = profiles[signal] / signal_avg * vol_avg[signal]

        profiles['nd'] = self.get_nd(na=profiles['na'], ne=profiles['ne'],
                                     impurity_fractions=shot['impurity_fractions'])

        # Densities [m^-3] & temperatures [eV]:
        for signal in self.signals_2d_var_iter:
            signal_avg, rhos, signals = self.get_vol_average_iter(profiles[signal], rho, 0, self.rho_edge)
            signal_avg_core, rhos_core, signals_core = \
                self.get_vol_average_iter(profiles[signal], rho, 0, self.rho_core)
            signal_avg_edge, rhos_edge, signals_edge = \
                self.get_vol_average_iter(profiles[signal], rho, self.rho_core, self.rho_edge)

            if 'rho' not in shot:
                shot['rhos'] = rhos
                shot['rhos_core'] = rhos_core
                shot['rhos_edge'] = rhos_edge

            shot[signal] = self.number2tensor(signal_avg, t)
            shot[signal + '_core'] = self.number2tensor(signal_avg_core, t)
            shot[signal + '_edge'] = self.number2tensor(signal_avg_edge, t)

            shot[signal + 's'] = signals
            shot[signal + 's_core'] = signals_core
            shot[signal + 's_edge'] = signals_edge

    def get_ecr_parameters(self, ecr_parameters, rhos_core, rhos_edge):
        """
        Get ECR parameters

        :param ecr_parameters: ECR parameters
        :param rhos_core: normalized radii in the core node
        :param rhos_edge: normalized radii in the edge node
        :return: ecr_parameters
        """
        alpha_T = ecr_parameters['alpha_te']
        beta_T = ecr_parameters['beta_te']

        I1 = torch.trapz(y=(1 - rhos_core ** beta_T) ** alpha_T * rhos_core, x=rhos_core)
        I2 = torch.trapz(y=(1 - rhos_edge ** beta_T) ** alpha_T * rhos_edge, x=rhos_edge)
        div_e0 = 2 * self.rho_core ** 2 * (I1 + I2) - 2 * I1 * self.rho_edge ** 2
        div_ea = self.rho_core ** 2 * (I1 + I2) - I1 * self.rho_edge ** 2

        ecr_parameters['core2e0'] = self.rho_core ** 2 * (self.rho_core ** 2 - self.rho_edge ** 2 + 2 * I2) / div_e0
        ecr_parameters['edge2e0'] = (self.rho_core ** 2 - self.rho_edge ** 2) * (2 * I1 - self.rho_core ** 2) / div_e0
        ecr_parameters['core2ea'] = I2 * self.rho_core ** 2 / div_ea
        ecr_parameters['edge2ea'] = I1 * (self.rho_core ** 2 - self.rho_edge ** 2) / div_ea

        return ecr_parameters


if __name__ == '__main__':
    pass
    preprocessor = PreprocessorITER()
    shot = preprocessor.preprocess(scenario=4)
    print(shot)
