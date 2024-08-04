"""
Dynamical System Solvers
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdiffeq

from src import config
from src.reactor import Reactor0D, Reactor1D, ReactorITER
from src.utils import J2eV, eV2J, keV2J, J2keV, tensor2array, tensors2arrays


class Solver0D(object):
    """
    Solver for the 0D dynamical system
    """

    def __init__(self):
        """
        Initialize the solver
        """
        self.system_name = 'system0d'
        self.dim = 0
        self.device = config.device
        self.window = config.window
        self.figure_width = 4
        self.figure_height = 2

    def solve(self, reactor: Reactor0D, plot: bool = False, full_plot: bool = False, fig_type: str = 'png',
              comment: str = None, show: bool = False, save: bool = True):
        """
        Solve the dynamical system

        :param reactor: reactor
        :param plot: plot a figure
        :param full_plot: plot a full figure
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the figure
        :param save: save the solution
        :return: y, y_exp in [ni [1E19 m^-3], Ti [keV], Te [keV]]
        """
        times = reactor.get_time()
        num_times = times.shape[0]
        y_sol, y_exp, sol_plot, exp_plot = self.solve_steps(reactor=reactor, step_size=2 * num_times)

        if plot:
            self.plot(reactor, sol=sol_plot, exp=exp_plot, full_plot=full_plot, fig_type=fig_type,
                      comment=comment, show=show)

        if save:
            self.save(reactor, y_sol=y_sol, comment=comment)

        return y_sol, y_exp

    def solve_steps(self, reactor: Reactor0D, step_size: int):
        """
        Solve the dynamical system in the step size chunks

        :param reactor: reactor
        :param step_size: step size
        :return: y_sol[ni [1E19 m^-3], Ti [keV], Te [keV]], y_exp, sol_plot, exp_plot
        """
        times = reactor.get_time()
        num_times = times.shape[0]

        ne_exp = reactor.get_ne(times)
        ni_exp = reactor.get_ni(times)
        nc_exp = reactor.get_nc(times)
        Ti_exp = reactor.get_Ti(times)
        Te_exp = reactor.get_Te(times)

        system = reactor.system
        sol = torch.zeros(num_times, reactor.num_vars, dtype=torch.double, device=self.device)

        for index in range(num_times // step_size + 1):
            time_i0 = index * step_size
            time_i1 = min((index + 1) * step_size, num_times)
            times_step = times[time_i0:time_i1]

            ne0 = ne_exp[time_i0]
            ni0 = ni_exp[time_i0]
            Ti0 = Ti_exp[time_i0]
            Te0 = Te_exp[time_i0]

            Ui0 = 3 / 2 * ni0 * eV2J(Ti0)
            Ue0 = 3 / 2 * ne0 * eV2J(Te0)

            y0 = torch.tensor([ni0 / 1E19, J2keV(Ui0) / 1E19, J2keV(Ue0) / 1E19], device=self.device)

            if reactor.tuned:
                sol_step = torchdiffeq.odeint(system, y0, times_step, rtol=config.rtol, atol=config.atol)
            else:
                with torch.no_grad():
                    sol_step = torchdiffeq.odeint(system, y0, times_step, rtol=config.rtol, atol=config.atol)

            sol[time_i0: time_i1] = sol_step

        nc = nc_exp.clone()

        ni = sol[:, 0] * 1E19
        Ui = keV2J(sol[:, 1] * 1E19)
        Ue = keV2J(sol[:, 2] * 1E19)

        ne = ni + nc * reactor.impurity_charge
        Ti = J2eV(Ui / ni * 2 / 3)
        Te = J2eV(Ue / ne * 2 / 3)

        y_sol = torch.stack((ni / 1E19, Ti / 1E3, Te / 1E3), dim=0)
        y_exp = torch.stack((ni_exp / 1E19, Ti_exp / 1E3, Te_exp / 1E3), dim=0)
        sol_plot = (ni, ne, Ti, Te)
        exp_plot = (ni_exp, ne_exp, Ti_exp, Te_exp)

        return y_sol, y_exp, sol_plot, exp_plot

    def plot(self, reactor: Reactor0D, sol: tuple, exp: tuple, full_plot: bool = False, fig_type: str = 'png',
             comment: str = None, show: bool = False):
        """
        Plot results

        :param reactor: reactor
        :param sol: simulation solutions in (ni, ne, Ti, Te)
        :param exp: experiment data in (ni_exp, ne_exp, Ti_exp, Te_exp)
        :param full_plot: plot a full figure with all source terms
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the figure
        """
        times = reactor.get_time()
        t = tensor2array(times)
        ni, ne, Ti, Te = tensors2arrays(sol)
        ni_exp, ne_exp, Ti_exp, Te_exp = tensors2arrays(exp)

        if full_plot:
            num_rows = 5
        else:
            num_rows = 2
        fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))

        ax1 = fig.add_subplot(num_rows, 1, 1)
        ax1.plot(t, ni, 'r-', label=r'$\hat{n}_i$')
        ax1.plot(t, ni_exp, 'r--', label=r'$n_i$')
        ax1.plot(t, ne, 'b-', label=r'$\hat{n}_e$')
        ax1.plot(t, ne_exp, 'b--', label=r'$n_e$')
        ax1.set_ylabel(r'$n$ [m$^{-3}$]')
        ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax1.grid('on')

        ax2 = fig.add_subplot(num_rows, 1, 2)
        ax2.plot(t, Ti / 1E3, 'r-', label=r'$\hat{T}_i$')
        ax2.plot(t, Ti_exp / 1E3, 'r--', label=r'$T_i$')
        ax2.plot(t, Te / 1E3, 'b-', label=r'$\hat{T}_e$')
        ax2.plot(t, Te_exp / 1E3, 'b--', label=r'$T_e$')
        ax2.set_ylabel(r'$T$ [keV]')
        ax2.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax2.grid('on')

        if not full_plot:
            ax2.set_xlabel(r'$t$ [s]')
        else:
            ni, ne, Ti, Te = sol
            Ui = 3 / 2 * ni * eV2J(Ti)
            Ue = 3 / 2 * ne * eV2J(Te)
            Si, Pi, Pe = reactor.get_sources(ni, Ui, Ue, times)
            Si_ext, Si_fus, Si_dif = tensors2arrays(Si)
            Pi_aux, Pi_fus, Q_ie, Pi_dif = tensors2arrays(Pi)
            P_oh, Pe_aux, Pe_fus, P_rad, Pe_dif = tensors2arrays(Pe)

            ax3 = fig.add_subplot(num_rows, 1, 3)
            ax3.plot(t, Si_ext, 'r-', label=r'$S_{i,ext}$')
            ax3.plot(t, Si_fus, 'b-', label=r'$S_{i,fus}$')
            ax3.plot(t, Si_dif, 'g-', label=r'$S_{i,dif}$')
            ax3.set_ylabel(r'$S$ [m$^{-3}$s$^{-1}$]')
            ax3.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            ax3.grid('on')

            ax4 = fig.add_subplot(num_rows, 1, 4)
            ax4.plot(t, Pi_aux / 1E6, 'r-', label=r'$P_{i,aux}$')
            ax4.plot(t, Pi_fus / 1E6, 'b-', label=r'$P_{i,fus}$')
            ax4.plot(t, Q_ie / 1E6, 'g-', label=r'$Q_{ie}$')
            ax4.plot(t, Pi_dif / 1E6, 'm-', label=r'$P_{i,dif}$')
            ax4.set_ylabel(r'$P$ [MW/m$^{3}$]')
            ax4.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            ax4.grid('on')

            ax5 = fig.add_subplot(num_rows, 1, 5)
            ax5.plot(t, P_oh / 1E6, 'k-', label=r'$P_{oh}$')
            ax5.plot(t, Pe_aux / 1E6, 'r-', label=r'$P_{e,aux}$')
            ax5.plot(t, Pe_fus / 1E6, 'b-', label=r'$P_{e,fus}$')
            ax5.plot(t, P_rad / 1E6, 'g-', label=r'$P_{rad}$')
            ax5.plot(t, Pe_dif / 1E6, 'm-', label=r'$P_{e,dif}$')
            ax5.set_xlabel(r'$t$ [s]')
            ax5.set_ylabel(r'$P$ [MW/m$^{3}$]')
            ax5.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            ax5.grid('on')

        ax1.set_title('Shot ' + str(reactor.shot_num))
        plt.tight_layout()

        fig_name = self.get_fig_path(reactor.shot_num, len(times), comment, fig_type)
        fig.savefig(fig_name)
        print('Figure saved: ' + fig_name)

        if show:
            plt.show()
        plt.close()

    def get_fig_path(self, shot_num: int, num_steps: int, comment: str, fig_type: str):
        """
        Get the figure path

        :param shot_num: shot number
        :param num_steps: number of time steps
        :param comment: comment to the figure name
        :param fig_type: figure type
        :return: fig_path
        """
        fig_folder = os.path.join('.', 'figure', self.system_name)
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)

        fig_name = '{:s}_{:d}_t{:d}_w{:d}'.format(self.system_name, shot_num, num_steps, self.window)
        if comment is not None:
            fig_name += '_' + comment.strip().lower()
        fig_name += '.' + fig_type

        fig_path = os.path.join(fig_folder, fig_name)
        return fig_path

    def save(self, reactor: Reactor0D, y_sol: torch.Tensor, comment: str):
        """
        Save the solution

        :param reactor: reactor
        :param comment: comment
        :param y_sol: [ni [10^19 m^-3], Ti [keV], Te [keV]]
        """
        sol_path = self.get_sol_path(reactor.shot_num, reactor.time.shape[0], comment)
        torch.save(y_sol, sol_path)
        print('Solution saved: ' + sol_path)

    def load(self, reactor: Reactor0D, comment: str) -> torch.Tensor:
        """
        Load the solution

        :param reactor: reactor
        :param comment: comment
        :return: y_sol = [ni [10^19 m^-3], Ti [keV], Te [keV]]
        """
        sol_path = self.get_sol_path(reactor.shot_num, reactor.time.shape[0], comment)
        y_sol = torch.load(sol_path)
        return y_sol

    def get_sol_path(self, shot_num: int, num_steps: int, comment: str):
        """
        Get the solution path

        :param shot_num: shot number
        :param num_steps: number of time steps
        :param comment: comment to the solution name
        :return: sol_path
        """
        sol_folder = os.path.join('.', 'solution', self.system_name)
        if not os.path.exists(sol_folder):
            os.mkdir(sol_folder)

        sol_name = '{:s}_{:d}_t{:d}_w{:d}'.format(self.system_name, shot_num, num_steps, self.window)
        if comment is not None:
            sol_name += '_' + comment.strip().lower()
        sol_name += '.pt'

        sol_path = os.path.join(sol_folder, sol_name)
        return sol_path

    def split_sol(self, y_sol: torch.Tensor) -> dict:
        """
        Split the solution into individual variables

        :param y_sol: solution
        :return: sol = {ni [m^-3], Ti [eV], Te [eV]}
        """
        sol = {}
        sol['ni'] = y_sol[0] * 1E19
        sol['Ti'] = y_sol[1] * 1E3
        sol['Te'] = y_sol[1] * 1E3
        return sol


class Solver1D(Solver0D):
    """
    Solver for the 1D dynamical system
    """

    def __init__(self):
        """
        Initialize the solver
        """
        super().__init__()
        self.system_name = 'system1d'
        self.dim = 1

    def solve_core_edge(self, reactor: Reactor1D):
        """
        Solve the dynamical system with only the core and edge nodes

        :param reactor: reactor
        :return: y_sol, y_exp in [nd_core, nd_edge [1E19 m^-3], Td_core, Td_edge [keV], Te_core, Te_edge [keV]]
        """
        times = reactor.get_time()

        ne_core_exp = reactor.get_ne_node(times, node='core')
        nd_core_exp = reactor.get_nd_node(times, node='core')
        nc_core_exp = reactor.get_nc_node(times, node='core')
        Td_core_exp = reactor.get_Td_node(times, node='core')
        Te_core_exp = reactor.get_Te_node(times, node='core')

        ne_edge_exp = reactor.get_ne_node(times, node='edge')
        nd_edge_exp = reactor.get_nd_node(times, node='edge')
        nc_edge_exp = reactor.get_nc_node(times, node='edge')
        Td_edge_exp = reactor.get_Td_node(times, node='edge')
        Te_edge_exp = reactor.get_Te_node(times, node='edge')

        ne0_core = ne_core_exp[0]
        nd0_core = nd_core_exp[0]
        Td0_core = Td_core_exp[0]
        Te0_core = Te_core_exp[0]

        ne0_edge = ne_edge_exp[0]
        nd0_edge = nd_edge_exp[0]
        Td0_edge = Td_edge_exp[0]
        Te0_edge = Te_edge_exp[0]

        Ud0_core = nd0_core * eV2J(Td0_core) * 3 / 2
        Ue0_core = ne0_core * eV2J(Te0_core) * 3 / 2

        Ud0_edge = nd0_edge * eV2J(Td0_edge) * 3 / 2
        Ue0_edge = ne0_edge * eV2J(Te0_edge) * 3 / 2

        y0 = torch.tensor([nd0_core / 1E19, nd0_edge / 1E19, J2keV(Ud0_core) / 1E19, J2keV(Ud0_edge) / 1E19,
                           J2keV(Ue0_core) / 1E19, J2keV(Ue0_edge) / 1E19], device=self.device)
        sol = torchdiffeq.odeint(reactor.system_core_edge, y0, times, rtol=config.rtol, atol=config.atol)

        nc_core = nc_core_exp.clone()
        nc_edge = nc_edge_exp.clone()

        nd_core = sol[:, 0] * 1E19
        nd_edge = sol[:, 1] * 1E19
        Ud_core = keV2J(sol[:, 2] * 1E19)
        Ud_edge = keV2J(sol[:, 3] * 1E19)
        Ue_core = keV2J(sol[:, 4] * 1E19)
        Ue_edge = keV2J(sol[:, 5] * 1E19)

        ne_core = nd_core + nc_core * reactor.impurity_charge
        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)

        ne_edge = nd_edge + nc_edge * reactor.impurity_charge
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)

        y_sol = torch.stack((nd_core / 1E19, nd_edge / 1E19, Td_core / 1E3, Td_edge / 1E3,
                             Te_core / 1E3, Te_edge / 1E3), dim=0)
        y_exp = torch.stack((nd_core_exp / 1E19, nd_edge_exp / 1E19, Td_core_exp / 1E3, Td_edge_exp / 1E3,
                             Te_core_exp / 1E3, Te_edge_exp / 1E3), dim=0)

        return y_sol, y_exp

    def solve_sol(self, reactor: Reactor1D):
        """
        Solve the dynamical system with only the SOL node

        :param reactor: reactor
        :return: y, y_exp in [nd_sol [1E19 m^-3], Td_sol [keV], Te_sol [keV]]
        """
        times = reactor.get_time()
        ne_sol_exp = reactor.get_ne_node(times, node='sol')
        nd_sol_exp = reactor.get_nd_node(times, node='sol')
        nc_sol_exp = reactor.get_nc_node(times, node='sol')
        Td_sol_exp = reactor.get_Td_node(times, node='sol')
        Te_sol_exp = reactor.get_Te_node(times, node='sol')

        ne0_sol = ne_sol_exp[0]
        nd0_sol = nd_sol_exp[0]
        Td0_sol = Td_sol_exp[0]
        Te0_sol = Te_sol_exp[0]
        Ud0_sol = nd0_sol * eV2J(Td0_sol) * 3 / 2
        Ue0_sol = ne0_sol * eV2J(Te0_sol) * 3 / 2

        y0 = torch.tensor([nd0_sol / 1E19, J2keV(Ud0_sol) / 1E19, J2keV(Ue0_sol) / 1E19], device=self.device)
        sol = torchdiffeq.odeint(reactor.system_sol, y0, times, rtol=config.rtol, atol=config.atol)

        nc_sol = nc_sol_exp.clone()
        nd_sol = sol[:, 0] * 1E19
        Ud_sol = keV2J(sol[:, 1] * 1E19)
        Ue_sol = keV2J(sol[:, 2] * 1E19)

        ne_sol = nd_sol + nc_sol * reactor.impurity_charge
        Td_sol = J2eV(Ud_sol / nd_sol * 2 / 3)
        Te_sol = J2eV(Ue_sol / ne_sol * 2 / 3)

        y_sol = torch.stack((nd_sol / 1E19, Td_sol / 1E3, Te_sol / 1E3), dim=0)
        y_exp = torch.stack((nd_sol_exp / 1E19, Td_sol_exp / 1E3, Te_sol_exp / 1E3), dim=0)

        return y_sol, y_exp

    def solve_steps(self, reactor: Reactor1D, step_size: int):
        """
        Solve the dynamical system in n steps

        :param reactor: reactor
        :param step_size: step size
        :return: y_sol, y_exp, sol_plot, exp_plot
        """
        times = reactor.get_time()
        num_times = times.shape[0]

        ne_core_exp = reactor.get_ne_node(times, node='core')
        nd_core_exp = reactor.get_nd_node(times, node='core')
        nc_core_exp = reactor.get_nc_node(times, node='core')
        Td_core_exp = reactor.get_Td_node(times, node='core')
        Te_core_exp = reactor.get_Te_node(times, node='core')

        ne_edge_exp = reactor.get_ne_node(times, node='edge')
        nd_edge_exp = reactor.get_nd_node(times, node='edge')
        nc_edge_exp = reactor.get_nc_node(times, node='edge')
        Td_edge_exp = reactor.get_Td_node(times, node='edge')
        Te_edge_exp = reactor.get_Te_node(times, node='edge')

        ne_sol_exp = reactor.get_ne_node(times, node='sol')
        nd_sol_exp = reactor.get_nd_node(times, node='sol')
        nc_sol_exp = reactor.get_nc_node(times, node='sol')
        Td_sol_exp = reactor.get_Td_node(times, node='sol')
        Te_sol_exp = reactor.get_Te_node(times, node='sol')

        system = reactor.system
        sol = torch.zeros(num_times, reactor.num_vars, dtype=torch.double, device=self.device)

        for index in range(num_times // step_size + 1):
            time_i0 = index * step_size
            time_i1 = min((index + 1) * step_size, num_times)
            times_step = times[time_i0:time_i1]

            ne0_core = ne_core_exp[time_i0]
            nd0_core = nd_core_exp[time_i0]
            Td0_core = Td_core_exp[time_i0]
            Te0_core = Te_core_exp[time_i0]

            ne0_edge = ne_edge_exp[time_i0]
            nd0_edge = nd_edge_exp[time_i0]
            Td0_edge = Td_edge_exp[time_i0]
            Te0_edge = Te_edge_exp[time_i0]

            ne0_sol = ne_sol_exp[time_i0]
            nd0_sol = nd_sol_exp[time_i0]
            Td0_sol = Td_sol_exp[time_i0]
            Te0_sol = Te_sol_exp[time_i0]

            Ud0_core = nd0_core * eV2J(Td0_core) * 3 / 2
            Ue0_core = ne0_core * eV2J(Te0_core) * 3 / 2

            Ud0_edge = nd0_edge * eV2J(Td0_edge) * 3 / 2
            Ue0_edge = ne0_edge * eV2J(Te0_edge) * 3 / 2

            Ud0_sol = nd0_sol * eV2J(Td0_sol) * 3 / 2
            Ue0_sol = ne0_sol * eV2J(Te0_sol) * 3 / 2

            y0 = torch.tensor([nd0_core / 1E19, nd0_edge / 1E19, nd0_sol / 1E19,
                               J2keV(Ud0_core) / 1E19, J2keV(Ud0_edge) / 1E19, J2keV(Ud0_sol) / 1E19,
                               J2keV(Ue0_core) / 1E19, J2keV(Ue0_edge) / 1E19, J2keV(Ue0_sol) / 1E19],
                              device=self.device)

            if reactor.tuned:
                sol_step = torchdiffeq.odeint(system, y0, times_step, rtol=config.rtol, atol=config.atol)
            else:
                with torch.no_grad():
                    sol_step = torchdiffeq.odeint(system, y0, times_step, rtol=config.rtol, atol=config.atol)

            sol[time_i0: time_i1] = sol_step

        nc_core = nc_core_exp.clone()
        nc_edge = nc_edge_exp.clone()
        nc_sol = nc_sol_exp.clone()

        nd_core = sol[:, 0] * 1E19
        nd_edge = sol[:, 1] * 1E19
        nd_sol = sol[:, 2] * 1E19
        Ud_core = keV2J(sol[:, 3] * 1E19)
        Ud_edge = keV2J(sol[:, 4] * 1E19)
        Ud_sol = keV2J(sol[:, 5] * 1E19)
        Ue_core = keV2J(sol[:, 6] * 1E19)
        Ue_edge = keV2J(sol[:, 7] * 1E19)
        Ue_sol = keV2J(sol[:, 8] * 1E19)

        ne_core = nd_core + nc_core * reactor.impurity_charge
        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)

        ne_edge = nd_edge + nc_edge * reactor.impurity_charge
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)

        ne_sol = nd_sol + nc_sol * reactor.impurity_charge
        Td_sol = J2eV(Ud_sol / nd_sol * 2 / 3)
        Te_sol = J2eV(Ue_sol / ne_sol * 2 / 3)

        y_sol = torch.stack((nd_core / 1E19, nd_edge / 1E19, nd_sol / 1E19,
                             Td_core / 1E3, Td_edge / 1E3, Td_sol / 1E3,
                             Te_core / 1E3, Te_edge / 1E3, Te_sol / 1E3), dim=0)
        y_exp = torch.stack((nd_core_exp / 1E19, nd_edge_exp / 1E19, nd_sol_exp / 1E19,
                             Td_core_exp / 1E3, Td_edge_exp / 1E3, Td_sol_exp / 1E3,
                             Te_core_exp / 1E3, Te_edge_exp / 1E3, Te_sol_exp / 1E3), dim=0)

        sol_core = (nd_core, ne_core, Td_core, Te_core)
        sol_edge = (nd_edge, ne_edge, Td_edge, Te_edge)
        sol_sol = (nd_sol, ne_sol, Td_sol, Te_sol)
        exp_core = (nd_core_exp, ne_core_exp, Td_core_exp, Te_core_exp)
        exp_edge = (nd_edge_exp, ne_edge_exp, Td_edge_exp, Te_edge_exp)
        exp_sol = (nd_sol_exp, ne_sol_exp, Td_sol_exp, Te_sol_exp)

        sol_plot = (sol_core, sol_edge, sol_sol)
        exp_plot = (exp_core, exp_edge, exp_sol)

        return y_sol, y_exp, sol_plot, exp_plot

    def plot(self, reactor: Reactor1D, sol: tuple, exp: tuple, full_plot: bool = False, fig_type: str = 'png',
             comment: str = None, show: bool = False):
        """
        Plot results

        :param reactor: reactor
        :param sol: simulation solutions
        :param exp: experiment data
        :param full_plot: plot a full figure with all source terms
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the plot
        """
        shot_num = reactor.shot_num
        times = reactor.get_time()
        sol_core, sol_edge, sol_sol = sol
        exp_core, exp_edge, exp_sol = exp

        if full_plot:
            nd_core, ne_core, Td_core, Te_core = sol_core
            nd_edge, ne_edge, Td_edge, Te_edge = sol_edge
            nd_sol, ne_sol, Td_sol, Te_sol = sol_sol

            Ud_core = nd_core * eV2J(Td_core) * 3 / 2
            Ue_core = ne_core * eV2J(Te_core) * 3 / 2
            Ud_edge = nd_edge * eV2J(Td_edge) * 3 / 2
            Ue_edge = ne_edge * eV2J(Te_edge) * 3 / 2
            Ud_sol = nd_sol * eV2J(Td_sol) * 3 / 2
            Ue_sol = ne_sol * eV2J(Te_sol) * 3 / 2

            Sd, Pd, Pe = reactor.get_sources((nd_core, nd_edge, nd_sol), (Ud_core, Ud_edge, Ud_sol),
                                             (Ue_core, Ue_edge, Ue_sol), times)
            Sd_ext_core, Sd_ext_edge, Sd_ion_sol, Sd_rec_sol, Sd_tran_core, Sd_tran_edge, Sd_tran_sol, \
            Sd_iol_edge, Sd_iol_sol = tensors2arrays(Sd)
            Pd_aux_core, Pd_aux_edge, Q_de_core, Q_de_edge, Q_de_sol, Pd_at_sol, \
            Pd_tran_core, Pd_tran_edge, Pd_tran_sol, Pd_iol_edge, Pd_iol_sol = tensors2arrays(Pd)
            Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, P_rad_core, P_rad_edge, P_rad_sol, \
            Pe_ion_sol, Pe_rec_sol, Pe_tran_core, Pe_tran_edge, Pe_tran_sol \
                = tensors2arrays(Pe)

            Sd_core = (Sd_ext_core, Sd_tran_core, None)
            Pd_core = (Pd_aux_core, Q_de_core, Pd_tran_core, None)
            Pe_core = (Pe_aux_core, P_oh_core, P_rad_core, Pe_tran_core)
            sources_core = (Sd_core, Pd_core, Pe_core)

            Sd_edge = (Sd_ext_edge, Sd_tran_edge, Sd_iol_edge)
            Pd_edge = (Pd_aux_edge, Q_de_edge, Pd_tran_edge, Pd_iol_edge)
            Pe_edge = (Pe_aux_edge, P_oh_edge, P_rad_edge, Pe_tran_edge)
            sources_edge = (Sd_edge, Pd_edge, Pe_edge)

            Sd_sol = (Sd_ion_sol, Sd_rec_sol, Sd_tran_sol, Sd_iol_sol)
            Pd_sol = (Pd_at_sol, Q_de_sol, Pd_tran_sol, Pd_iol_sol)
            Pe_sol = (Pe_ion_sol, Pe_rec_sol, P_rad_sol, Pe_tran_sol)
            sources_sol = (Sd_sol, Pd_sol, Pe_sol)
        else:
            sources_core = None
            sources_edge = None
            sources_sol = None

        self.plot_node(shot_num, 'core', times, sol_core, exp_core, sources_core, full_plot, fig_type, comment, show)
        self.plot_node(shot_num, 'edge', times, sol_edge, exp_edge, sources_edge, full_plot, fig_type, comment, show)
        self.plot_node(shot_num, 'sol', times, sol_sol, exp_sol, sources_sol, full_plot, fig_type, comment, show)

    def plot_node(self, shot_num: int, node: str, times: torch.Tensor, sol_node: tuple, exp_node: tuple,
                  sources_node: tuple = None, full_plot: bool = False, fig_type: str = 'png', comment: str = None,
                  show: bool = False):
        """
        Plot results for one node

        :param shot_num: shot number
        :param node: node name
        :param times: time [s]
        :param sol_node: simulation solutions
        :param exp_node: experiment data
        :param sources_node: source terms
        :param full_plot: plot a full figure with all source terms
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the plot
        """
        t = tensor2array(times)
        nd_node, ne_node, Td_node, Te_node = tensors2arrays(sol_node)
        nd_node_exp, ne_node_exp, Td_node_exp, Te_node_exp = tensors2arrays(exp_node)

        if full_plot:
            num_rows = 5
        else:
            num_rows = 2
        fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))
        node_plot = r'\text{' + node + '}'

        ax1 = fig.add_subplot(num_rows, 1, 1)
        ax1.plot(t, nd_node, 'r-', label=r'$\hat{n}_D^{' + node_plot + '}$')
        ax1.plot(t, nd_node_exp, 'r--', label=r'$n_D^{' + node_plot + '}$')
        ax1.plot(t, ne_node, 'b-', label=r'$\hat{n}_e^{' + node_plot + '}$')
        ax1.plot(t, ne_node_exp, 'b--', label=r'$n_e^{' + node_plot + '}$')
        ax1.set_ylabel(r'$n$ [m$^{-3}$]')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.grid('on')

        ax2 = fig.add_subplot(num_rows, 1, 2)
        ax2.plot(t, Td_node / 1E3, 'r-', label=r'$\hat{T}_D^{' + node_plot + '}$')
        ax2.plot(t, Td_node_exp / 1E3, 'r--', label=r'$T_D^{' + node_plot + '}$')
        ax2.plot(t, Te_node / 1E3, 'b-', label=r'$\hat{T}_e^{' + node_plot + '}$')
        ax2.plot(t, Te_node_exp / 1E3, 'b--', label=r'$T_e^{' + node_plot + '}$')
        ax2.set_ylabel(r'$T$ [keV]')
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.grid('on')

        if not full_plot:
            ax2.set_xlabel(r'$t$ [s]')
        else:
            Sd_node, Pd_node, Pe_node = sources_node

            if node != 'sol':
                Sd_ext_node, Sd_tran_node, Sd_iol_node = Sd_node
                Pd_aux_node, Q_de_node, Pd_tran_node, Pd_iol_node = Pd_node
                Pe_aux_node, P_oh_node, P_rad_node, Pe_tran_node = Pe_node

                ax3 = fig.add_subplot(num_rows, 1, 3)
                ax3.plot(t, Sd_ext_node, 'r-', label=r'$S_{D,\text{ext}}^{' + node_plot + '}$')
                ax3.plot(t, -Sd_tran_node, 'm-', label=r'$-S_{D,\text{tran}}^{' + node_plot + '}$')
                if node == 'edge':
                    ax3.plot(t, -Sd_iol_node, 'c-', label=r'$-S_{D,\text{iol}}^{' + node_plot + '}$')
                ax3.set_ylabel(r'$S$ [m$^{-3}$s$^{-1}$]')
                ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax3.grid('on')

                ax4 = fig.add_subplot(num_rows, 1, 4)
                ax4.plot(t, Pd_aux_node / 1E6, 'r-', label=r'$P_{D,\text{aux}}^{' + node_plot + '}$')
                ax4.plot(t, Q_de_node / 1E6, 'g-', label=r'$Q_{De}^{' + node_plot + '}$')
                if node == 'core':
                    ax4.plot(t, -Pd_tran_node / 1E6, 'm-', label=r'$-P_{D,\text{tran}}^{' + node_plot + '}$')
                elif node == 'edge':
                    ax4.plot(t, Pd_tran_node / 1E6, 'm-', label=r'$P_{D,\text{tran}}^{' + node_plot + '}$')
                    ax4.plot(t, -Pd_iol_node / 1E6, 'c-', label=r'$-P_{D,\text{iol}}^{' + node_plot + '}$')
                ax4.set_ylabel(r'$P$ [MW/m$^{3}$]')
                ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax4.grid('on')

                ax5 = fig.add_subplot(num_rows, 1, 5)
                ax5.plot(t, P_oh_node / 1E6, 'k-', label=r'$P_{\text{oh}}^{' + node_plot + '}$')
                ax5.plot(t, Pe_aux_node / 1E6, 'r-', label=r'$P_{e,\text{aux}}^{' + node_plot + '}$')
                ax5.plot(t, P_rad_node / 1E6, 'y-', label=r'$P_{\text{rad}}^{' + node_plot + '}$')
                ax5.plot(t, -Pe_tran_node / 1E6, 'm-', label=r'$-P_{e,\text{tran}}^{' + node_plot + '}$')
                ax5.set_xlabel(r'$t$ [s]')
                ax5.set_ylabel(r'$P$ [MW/m$^{3}$]')
                ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax5.grid('on')

            else:
                Sd_ion_node, Sd_rec_node, Sd_tran_node, Sd_iol_node = Sd_node
                Pd_at_node, Q_de_node, Pd_tran_node, Pd_iol_node = Pd_node
                Pe_ion_node, Pe_rec_node, P_rad_node, Pe_tran_node = Pe_node

                ax3 = fig.add_subplot(num_rows, 1, 3)
                ax3.plot(t, Sd_ion_node, 'r-', label=r'$S_{D,\text{ion}}^{' + node_plot + '}$')
                ax3.plot(t, -Sd_rec_node, 'b-', label=r'$-S_{D,\text{rec}}^{' + node_plot + '}$')
                ax3.plot(t, -Sd_tran_node, 'm-', label=r'$-S_{D,\text{tran}}^{' + node_plot + '}$')
                ax3.plot(t, Sd_iol_node, 'c-', label=r'$S_{D,\text{iol}}^{' + node_plot + '}$')
                ax3.set_ylabel(r'$S$ [m$^{-3}$s$^{-1}$]')
                ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax3.grid('on')

                ax4 = fig.add_subplot(num_rows, 1, 4)
                ax4.plot(t, -Pd_at_node / 1E6, 'k-', label=r'$-P_{D,\text{at}}^{' + node_plot + '}$')
                ax4.plot(t, Q_de_node / 1E6, 'g-', label=r'$Q_{De}^{' + node_plot + '}$')
                ax4.plot(t, -Pd_tran_node / 1E6, 'm-', label=r'$-P_{D,\text{tran}}^{' + node_plot + '}$')
                ax4.plot(t, Pd_iol_node / 1E6, 'c-', label=r'$P_{D,\text{iol}}^{' + node_plot + '}$')
                ax4.set_ylabel(r'$P$ [MW/m$^{3}$]')
                ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax4.grid('on')

                ax5 = fig.add_subplot(num_rows, 1, 5)
                ax5.plot(t, -Pe_ion_node / 1E6, 'r-', label=r'$-P_{e,\text{ion}}^{' + node_plot + '}$')
                ax5.plot(t, Pe_rec_node / 1E6, 'b-', label=r'$P_{e,\text{rec}}^{' + node_plot + '}$')
                ax5.plot(t, P_rad_node / 1E6, 'y-', label=r'$P_{\text{rad}}^{' + node_plot + '}$')
                ax5.plot(t, -Pe_tran_node / 1E6, 'm-', label=r'$-P_{e,\text{tran}}^{' + node_plot + '}$')
                ax5.set_xlabel(r'$t$ [s]')
                ax5.set_ylabel(r'$P$ [MW/m$^{3}$]')
                ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax5.grid('on')

        ax1.set_title('Shot ' + str(shot_num) + ' ' + node.capitalize())
        plt.tight_layout()

        fig_comment = node
        if comment is not None:
            fig_comment += '_' + comment
        fig_name = self.get_fig_path(shot_num, len(times), fig_comment, fig_type)
        fig.savefig(fig_name)
        print('Figure saved: ' + fig_name)

        if show:
            plt.show()
        plt.close()

    def split_sol(self, y_sol: torch.Tensor) -> dict:
        """
        Split the solution tensor into individual variables

        :param y_sol: solution
        :return: sol = {nd_core, nd_edge, nd_sol [m^-3], Td_core, Td_edge, Td_sol [eV], Te_core, Te_edge, Te_sol [eV]}
        """
        sol = {}
        sol['nd_core'] = y_sol[0] * 1E19
        sol['nd_edge'] = y_sol[1] * 1E19
        sol['nd_sol'] = y_sol[2] * 1E19
        sol['Td_core'] = y_sol[3] * 1E3
        sol['Td_edge'] = y_sol[4] * 1E3
        sol['Td_sol'] = y_sol[5] * 1E3
        sol['Te_core'] = y_sol[6] * 1E3
        sol['Te_edge'] = y_sol[7] * 1E3
        sol['Te_sol'] = y_sol[8] * 1E3
        return sol


class SolverITER(Solver1D):
    """
    Solver for the 1D dynamical system of ITER
    """

    def __init__(self, is_inductive: bool = True):
        """
        Initialize the solver

        :param is_inductive: True for inductive scenarios
        """
        super().__init__()
        if is_inductive:
            self.system_name = 'system1d_iter_ind'
        else:
            self.system_name = 'system1d_iter_non'
        self.time_step = config.time_step
        self.is_inductive = is_inductive

    def solve(self, reactor: ReactorITER, plot: bool = False, full_plot: bool = False, fig_type: str = 'png',
              comment: str = None, show: bool = False, save: bool = True, **kwargs):
        """
        Solve the dynamical system

        :param reactor: reactor
        :param plot: plot a figure
        :param full_plot: plot a full figure
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the figure
        :param save: save the solution
        :key start_up: True to include the start-up process
        :return: y_sol, y_exp
        """
        if 'start_up' in kwargs:
            y_sol, y_exp, sol_plot, exp_plot = self.solve_iter(reactor=reactor, start_up=kwargs['start_up'])
        else:
            y_sol, y_exp, sol_plot, exp_plot = self.solve_iter(reactor=reactor)

        if plot:
            self.plot(reactor, sol=sol_plot, exp=exp_plot, full_plot=full_plot, fig_type=fig_type, comment=comment,
                      show=show)

        if save:
            self.save(reactor, y_sol=y_sol, comment=comment)

        return y_sol, y_exp

    def solve_iter(self, reactor: ReactorITER, start_up: bool = False):
        """
        Solve the dynamical system for ITER

        :param reactor: reactor
        :param start_up: True to include the start-up process
        :return: y_sol, y_exp, sol_plot, exp_plot
        """
        times = reactor.get_time()

        nd_core_exp = reactor.get_nd_node(times, node='core')
        nt_core_exp = reactor.get_nt_node(times, node='core')
        na_core_exp = reactor.get_na_node(times, node='core')
        ne_core_exp = reactor.get_ne_node(times, node='core')
        Td_core_exp = reactor.get_Td_node(times, node='core')
        Tt_core_exp = reactor.get_Tt_node(times, node='core')
        Ta_core_exp = reactor.get_Ta_node(times, node='core')
        Te_core_exp = reactor.get_Te_node(times, node='core')

        nd_edge_exp = reactor.get_nd_node(times, node='edge')
        nt_edge_exp = reactor.get_nt_node(times, node='edge')
        na_edge_exp = reactor.get_na_node(times, node='edge')
        ne_edge_exp = reactor.get_ne_node(times, node='edge')
        Td_edge_exp = reactor.get_Td_node(times, node='edge')
        Tt_edge_exp = reactor.get_Tt_node(times, node='edge')
        Ta_edge_exp = reactor.get_Ta_node(times, node='edge')
        Te_edge_exp = reactor.get_Te_node(times, node='edge')

        if start_up:
            nd0_core = nd_core_exp[0]
            nt0_core = nt_core_exp[0]
            na0_core = torch.ones_like(na_core_exp[0]) * 1e17
            ne0_core, _ = reactor.get_ne_nz_node(nd0_core, nt0_core, na0_core)
            Td0_core = torch.ones_like(Td_core_exp[0]) * 2e3
            Tt0_core = torch.ones_like(Tt_core_exp[0]) * 2e3
            Ta0_core = torch.ones_like(Ta_core_exp[0]) * 2e3
            Te0_core = torch.ones_like(Te_core_exp[0]) * 2e3

            nd0_edge = nd_edge_exp[0]
            nt0_edge = nt_edge_exp[0]
            na0_edge = torch.ones_like(na_edge_exp[0]) * 1e17
            ne0_edge, _ = reactor.get_ne_nz_node(nd0_edge, nt0_edge, na0_edge)
            Td0_edge = torch.ones_like(Td_edge_exp[0]) * 1e3
            Tt0_edge = torch.ones_like(Tt_edge_exp[0]) * 1e3
            Ta0_edge = torch.ones_like(Ta_edge_exp[0]) * 1e3
            Te0_edge = torch.ones_like(Te_edge_exp[0]) * 1e3
        else:
            nd0_core = nd_core_exp[0]
            nt0_core = nt_core_exp[0]
            na0_core = na_core_exp[0]
            ne0_core = ne_core_exp[0]
            Td0_core = Td_core_exp[0]
            Tt0_core = Tt_core_exp[0]
            Ta0_core = Ta_core_exp[0]
            Te0_core = Te_core_exp[0]

            nd0_edge = nd_edge_exp[0]
            nt0_edge = nt_edge_exp[0]
            na0_edge = na_edge_exp[0]
            ne0_edge = ne_edge_exp[0]
            Td0_edge = Td_edge_exp[0]
            Tt0_edge = Tt_edge_exp[0]
            Ta0_edge = Ta_edge_exp[0]
            Te0_edge = Te_edge_exp[0]

        Ud0_core = nd0_core * eV2J(Td0_core) * 3 / 2
        Ut0_core = nt0_core * eV2J(Tt0_core) * 3 / 2
        Ua0_core = na0_core * eV2J(Ta0_core) * 3 / 2
        Ue0_core = ne0_core * eV2J(Te0_core) * 3 / 2

        Ud0_edge = nd0_edge * eV2J(Td0_edge) * 3 / 2
        Ut0_edge = nt0_edge * eV2J(Tt0_edge) * 3 / 2
        Ua0_edge = na0_edge * eV2J(Ta0_edge) * 3 / 2
        Ue0_edge = ne0_edge * eV2J(Te0_edge) * 3 / 2

        y0 = torch.tensor(
            [nd0_core / 1E19, nt0_core / 1E19, na0_core / 1E19,
             nd0_edge / 1E19, nt0_edge / 1E19, na0_edge / 1E19,
             J2keV(Ud0_core) / 1E19, J2keV(Ut0_core) / 1E19, J2keV(Ua0_core) / 1E19, J2keV(Ue0_core) / 1E19,
             J2keV(Ud0_edge) / 1E19, J2keV(Ut0_edge) / 1E19, J2keV(Ua0_edge) / 1E19, J2keV(Ue0_edge) / 1E19],
            device=self.device)
        sol = torchdiffeq.odeint(reactor.system, y0, times, rtol=config.rtol, atol=config.atol)

        nd_core = sol[:, 0] * 1E19
        nt_core = sol[:, 1] * 1E19
        na_core = sol[:, 2] * 1E19
        nd_edge = sol[:, 3] * 1E19
        nt_edge = sol[:, 4] * 1E19
        na_edge = sol[:, 5] * 1E19
        Ud_core = keV2J(sol[:, 6] * 1E19)
        Ut_core = keV2J(sol[:, 7] * 1E19)
        Ua_core = keV2J(sol[:, 8] * 1E19)
        Ue_core = keV2J(sol[:, 9] * 1E19)
        Ud_edge = keV2J(sol[:, 10] * 1E19)
        Ut_edge = keV2J(sol[:, 11] * 1E19)
        Ua_edge = keV2J(sol[:, 12] * 1E19)
        Ue_edge = keV2J(sol[:, 13] * 1E19)

        ne_core, _ = reactor.get_ne_nz_node(nd_core, nt_core, na_core)
        ne_edge, _ = reactor.get_ne_nz_node(nd_edge, nt_edge, na_edge)
        Td_core = J2eV(Ud_core / nd_core * 2 / 3)
        Td_edge = J2eV(Ud_edge / nd_edge * 2 / 3)
        Tt_core = J2eV(Ut_core / nt_core * 2 / 3)
        Tt_edge = J2eV(Ut_edge / nt_edge * 2 / 3)
        Ta_core = J2eV(Ua_core / na_core * 2 / 3)
        Ta_edge = J2eV(Ua_edge / na_edge * 2 / 3)
        Te_core = J2eV(Ue_core / ne_core * 2 / 3)
        Te_edge = J2eV(Ue_edge / ne_edge * 2 / 3)

        if not reactor.sensitivity_analysis:
            y_sol = torch.stack((nd_core / 1E19, na_core / 1E18, ne_core / 1E19,
                                 nd_edge / 1E19, na_edge / 1E18, ne_edge / 1E19,
                                 Td_core / 1E3, Te_core / 1E3, Td_edge / 1E3, Te_edge / 1E3), dim=0)
            y_exp = torch.stack((nd_core_exp / 1E19, na_core_exp / 1E18, ne_core_exp / 1E19,
                                 nd_edge_exp / 1E19, na_edge_exp / 1E18, ne_edge_exp / 1E19,
                                 Td_core_exp / 1E3, Te_core_exp / 1E3, Td_edge_exp / 1E3, Te_edge_exp / 1E3), dim=0)
        else:
            y_sol = torch.stack((nd_core, nt_core, na_core, ne_core, nd_edge, nt_edge, na_edge, ne_edge,
                                 Td_core, Tt_core, Ta_core, Te_core, Td_edge, Tt_edge, Ta_edge, Te_edge), dim=0)
            y_exp = None

        sol_core = (nd_core, nt_core, na_core, ne_core, Td_core, Tt_core, Ta_core, Te_core)
        sol_edge = (nd_edge, nt_edge, na_edge, ne_edge, Td_edge, Tt_edge, Ta_edge, Te_edge)
        sol_plot = (sol_core, sol_edge)

        exp_core = \
            (nd_core_exp, nt_core_exp, na_core_exp, ne_core_exp, Td_core_exp, Tt_core_exp, Ta_core_exp, Te_core_exp)
        exp_edge = \
            (nd_edge_exp, nt_edge_exp, na_edge_exp, ne_edge_exp, Td_edge_exp, Tt_edge_exp, Ta_edge_exp, Te_edge_exp)
        exp_plot = (exp_core, exp_edge)

        return y_sol, y_exp, sol_plot, exp_plot

    def plot(self, reactor: ReactorITER, sol: tuple, exp: tuple, full_plot: bool = False, fig_type: str = 'png',
             comment: str = None, show: bool = False):
        """
        Plot results

        :param reactor: reactor
        :param sol: simulation solutions
        :param exp: experiment data
        :param full_plot: plot a full figure with all source terms
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the plot
        """
        times = reactor.get_time()
        scenario = reactor.scenario
        sol_core, sol_edge = sol
        exp_core, exp_edge = exp

        if full_plot:
            nd_core, nt_core, na_core, ne_core, Td_core, Tt_core, Ta_core, Te_core = sol_core
            nd_edge, nt_edge, na_edge, ne_edge, Td_edge, Tt_edge, Ta_edge, Te_edge = sol_edge

            Ud_core = nd_core * eV2J(Td_core) * 3 / 2
            Ut_core = nt_core * eV2J(Tt_core) * 3 / 2
            Ua_core = na_core * eV2J(Ta_core) * 3 / 2
            Ue_core = ne_core * eV2J(Te_core) * 3 / 2
            Ud_edge = nd_edge * eV2J(Td_edge) * 3 / 2
            Ut_edge = nt_edge * eV2J(Tt_edge) * 3 / 2
            Ua_edge = na_edge * eV2J(Ta_edge) * 3 / 2
            Ue_edge = ne_edge * eV2J(Te_edge) * 3 / 2

            Sd, St, Sa, Pd, Pt, Pa, Pe, _ = \
                reactor.get_sources_iter((nd_core, nd_edge), (nt_core, nt_edge), (na_core, na_edge),
                                         (Ud_core, Ud_edge), (Ut_core, Ut_edge), (Ua_core, Ua_edge),
                                         (Ue_core, Ue_edge), times)

            Sd_ext_core, Sd_ext_edge, Sd_fus_core, Sd_fus_edge, Sd_tran_core, Sd_tran_edge, Sd_iol_edge \
                = tensors2arrays(Sd)
            St_ext_core, St_ext_edge, St_fus_core, St_fus_edge, St_tran_core, St_tran_edge, St_iol_edge \
                = tensors2arrays(St)
            Sa_ext_core, Sa_ext_edge, Sa_fus_core, Sa_fus_edge, Sa_tran_core, Sa_tran_edge, Sa_iol_edge \
                = tensors2arrays(Sa)

            Pd_aux_core, Pd_aux_edge, Pd_fus_core, Pd_fus_edge, Qd_core, Qd_edge, Pd_tran_core, Pd_tran_edge, \
            Pd_iol_edge = tensors2arrays(Pd)
            Pt_aux_core, Pt_aux_edge, Pt_fus_core, Pt_fus_edge, Qt_core, Qt_edge, Pt_tran_core, Pt_tran_edge, \
            Pt_iol_edge = tensors2arrays(Pt)
            Pa_aux_core, Pa_aux_edge, Pa_fus_core, Pa_fus_edge, Qa_core, Qa_edge, Pa_tran_core, Pa_tran_edge, \
            Pa_iol_edge = tensors2arrays(Pa)
            Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, Pe_fus_core, Pe_fus_edge, P_rad_core, P_rad_edge, \
            Qe_core, Qe_edge, Pe_tran_core, Pe_tran_edge = tensors2arrays(Pe)

            Sd_core = (Sd_ext_core, Sd_fus_core, Sd_tran_core, None)
            St_core = (St_ext_core, St_fus_core, St_tran_core, None)
            Sa_core = (Sa_ext_core, Sa_fus_core, Sa_tran_core, None)
            Pd_core = (Pd_aux_core, Pd_fus_core, Qd_core, Pd_tran_core, None)
            Pt_core = (Pt_aux_core, Pt_fus_core, Qt_core, Pt_tran_core, None)
            Pa_core = (Pa_aux_core, Pa_fus_core, Qa_core, Pa_tran_core, None)
            Pe_core = (Pe_aux_core, P_oh_core, Pe_fus_core, P_rad_core, Qe_core, Pe_tran_core)
            sources_core = (Sd_core, St_core, Sa_core, Pd_core, Pt_core, Pa_core, Pe_core)

            Sd_edge = (Sd_ext_edge, Sd_fus_edge, Sd_tran_edge, Sd_iol_edge)
            St_edge = (St_ext_edge, St_fus_edge, St_tran_edge, St_iol_edge)
            Sa_edge = (Sa_ext_edge, Sa_fus_edge, Sa_tran_edge, Sa_iol_edge)
            Pd_edge = (Pd_aux_edge, Pd_fus_edge, Qd_edge, Pd_tran_edge, Pd_iol_edge)
            Pt_edge = (Pt_aux_edge, Pt_fus_edge, Qt_edge, Pt_tran_edge, Pt_iol_edge)
            Pa_edge = (Pa_aux_edge, Pa_fus_edge, Qa_edge, Pa_tran_edge, Pa_iol_edge)
            Pe_edge = (Pe_aux_edge, P_oh_edge, Pe_fus_edge, P_rad_edge, Qe_edge, Pe_tran_edge)
            sources_edge = (Sd_edge, St_edge, Sa_edge, Pd_edge, Pt_edge, Pa_edge, Pe_edge)
        else:
            sources_core = None
            sources_edge = None

        self.plot_node(
            scenario, 'core', times, sol_core, exp_core, sources_core, full_plot, fig_type, comment, show)
        self.plot_node(
            scenario, 'edge', times, sol_edge, exp_edge, sources_edge, full_plot, fig_type, comment, show)

    def plot_node(self, scenario: int, node: str, times: torch.Tensor, sol_node: tuple, exp_node: tuple,
                  sources_node: tuple = None, full_plot: bool = False, fig_type: str = 'png', comment: str = None,
                  show: bool = False):
        """
        Plot results for one node

        :param scenario: scenario number
        :param node: node name
        :param times: time [s]
        :param sol_node: simulation solutions
        :param exp_node: experiment data
        :param sources_node: source terms
        :param full_plot: plot a full figure with all source terms
        :param fig_type: figure type
        :param comment: comment to the figure name
        :param show: show the plot
        """
        t = tensor2array(times)
        nd_node, nt_node, na_node, ne_node, Td_node, Tt_node, Ta_node, Te_node = tensors2arrays(sol_node)
        nd_node_exp, nt_node_exp, na_node_exp, ne_node_exp, Td_node_exp, Tt_node_exp, Ta_node_exp, Te_node_exp = \
            tensors2arrays(exp_node)

        if full_plot:
            num_rows = 7
        else:
            num_rows = 2
        fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))
        node_plot = r'\text{' + node + '}'

        ax1 = fig.add_subplot(num_rows, 1, 1)
        ax1.plot(t, nd_node / 1E19, 'r-', label=r'$n_D^{' + node_plot + '}$')
        ax1.plot(t, nd_node_exp / 1E19, 'r--')
        # ax1.plot(t, nt_node / 1E19, 'g-', label=r'$n_T^{' + node_plot + '}$')
        # ax1.plot(t, nt_node_exp / 1E19, 'g--')
        ax1.plot(t, na_node * 10 / 1E19, 'm-', label=r'$n_{\alpha}^{' + node_plot + r'} \times 10$')
        ax1.plot(t, na_node_exp * 10 / 1E19, 'm--')
        ax1.plot(t, ne_node / 1E19, 'b-', label=r'$n_e^{' + node_plot + '}$')
        ax1.plot(t, ne_node_exp / 1E19, 'b--')
        ax1.set_ylabel(r'$n$ [$10^{19}$ m$^{-3}$]')
        ax1.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax1.grid('on')

        ax2 = fig.add_subplot(num_rows, 1, 2)
        ax2.plot(t, Td_node / 1E3, 'r-', label=r'$T_D^{' + node_plot + '}$')
        ax2.plot(t, Td_node_exp / 1E3, 'r--')
        # ax2.plot(t, Tt_node / 1E3, 'g-', label=r'$T_T^{' + node_plot + '}$')
        # ax2.plot(t, Tt_node_exp / 1E3, 'g--')
        ax2.plot(t, Ta_node / 1E3, 'm-', label=r'$T_{\alpha}^{' + node_plot + '}$')
        # ax2.plot(t, Ta_node_exp / 1E3, 'm--')
        ax2.plot(t, Te_node / 1E3, 'b-', label=r'$T_e^{' + node_plot + '}$')
        ax2.plot(t, Te_node_exp / 1E3, 'b--')
        ax2.set_ylabel(r'$T$ [keV]')
        ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax2.grid('on')

        if not full_plot:
            ax2.set_xlabel(r'$t$ [s]')
        else:
            Sd_node, St_node, Sa_node, Pd_node, Pt_node, Pa_node, Pe_node = sources_node

            Sd_ext_node, Sd_fus_node, Sd_tran_node, Sd_iol_node = Sd_node
            St_ext_node, St_fus_node, St_tran_node, St_iol_node = St_node
            Sa_ext_node, Sa_fus_node, Sa_tran_node, Sa_iol_node = Sa_node

            Pd_aux_node, Pd_fus_node, Qd_node, Pd_tran_node, Pd_iol_node = Pd_node
            Pt_aux_node, Pt_fus_node, Qt_node, Pt_tran_node, Pt_iol_node = Pt_node
            Pa_aux_node, Pa_fus_node, Qa_node, Pa_tran_node, Pa_iol_node = Pa_node
            Pe_aux_node, P_oh_node, Pe_fus_node, P_rad_node, Qe_node, Pe_tran_node = Pe_node

            ax3 = fig.add_subplot(num_rows, 1, 3)
            ax3.plot(t, Sd_ext_node / 1E19, 'r-', label=r'$S_{D,\text{ext}}^{' + node_plot + '}$')
            ax3.plot(t, Sd_fus_node / 1E19, 'b-', label=r'$S_{D,\text{fus}}^{' + node_plot + '}$')
            ax3.plot(t, Sd_tran_node / 1E19, 'm-', label=r'$S_{D,\text{tran}}^{' + node_plot + '}$')
            if node == 'edge':
                ax3.plot(t, Sd_iol_node / 1E19, 'c-', label=r'$S_{D,\text{iol}}^{' + node_plot + '}$')
            ax3.set_ylabel(r'$S$ [$10^{19}$ m$^{-3}$s$^{-1}$]')
            ax3.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax3.grid('on')

            ax4 = fig.add_subplot(num_rows, 1, 4)
            ax4.plot(t, Sa_ext_node / 1E19, 'r-', label=r'$S_{\alpha,\text{ext}}^{' + node_plot + '}$')
            ax4.plot(t, Sa_fus_node / 1E19, 'b-', label=r'$S_{\alpha,\text{fus}}^{' + node_plot + '}$')
            ax4.plot(t, Sa_tran_node / 1E19, 'm-', label=r'$S_{\alpha,\text{tran}}^{' + node_plot + '}$')
            if node == 'edge':
                ax4.plot(t, Sa_iol_node / 1E19, 'c-', label=r'$S_{\alpha,\text{iol}}^{' + node_plot + '}$')
            ax4.set_ylabel(r'$S$ [$10^{19}$ m$^{-3}$s$^{-1}$]')
            ax4.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax4.grid('on')

            ax5 = fig.add_subplot(num_rows, 1, 5)
            ax5.plot(t, Pd_aux_node / 1E6, 'r-', label=r'$P_{D,\text{aux}}^{' + node_plot + '}$')
            ax5.plot(t, Pd_fus_node / 1E6, 'b-', label=r'$P_{D,\text{fus}}^{' + node_plot + '}$')
            ax5.plot(t, Qd_node / 1E6, 'g-', label=r'$Q_{D}^{' + node_plot + '}$')
            ax5.plot(t, Pd_tran_node / 1E6, 'm-', label=r'$P_{D,\text{tran}}^{' + node_plot + '}$')
            if node == 'edge':
                ax5.plot(t, Pd_iol_node / 1E6, 'c-', label=r'$P_{D,\text{iol}}^{' + node_plot + '}$')
            ax5.set_ylabel(r'$P$ [MW/m$^{3}$]')
            ax5.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax5.grid('on')

            ax6 = fig.add_subplot(num_rows, 1, 6)
            ax6.plot(t, Pa_aux_node / 1E6, 'r-', label=r'$P_{\alpha,\text{aux}}^{' + node_plot + '}$')
            ax6.plot(t, Pa_fus_node / 1E6, 'b-', label=r'$P_{\alpha,\text{fus}}^{' + node_plot + '}$')
            ax6.plot(t, Qa_node / 1E6, 'g-', label=r'$Q_{\alpha}^{' + node_plot + '}$')
            ax6.plot(t, Pa_tran_node / 1E6, 'm-', label=r'$P_{\alpha,\text{tran}}^{' + node_plot + '}$')
            if node == 'edge':
                ax6.plot(t, Pa_iol_node / 1E6, 'c-', label=r'$P_{\alpha,\text{iol}}^{' + node_plot + '}$')
            ax6.set_ylabel(r'$P$ [MW/m$^{3}$]')
            ax6.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax6.grid('on')

            ax7 = fig.add_subplot(num_rows, 1, 7)
            ax7.plot(t, P_oh_node / 1E6, 'k-', label=r'$P_{\text{oh}}^{' + node_plot + '}$')
            ax7.plot(t, Pe_aux_node / 1E6, 'r-', label=r'$P_{e,\text{aux}}^{' + node_plot + '}$')
            ax7.plot(t, Pe_fus_node / 1E6, 'b-', label=r'$P_{e,\text{fus}}^{' + node_plot + '}$')
            ax7.plot(t, Qe_node / 1E6, 'g-', label=r'$Q_{e}^{' + node_plot + '}$')
            ax7.plot(t, -P_rad_node / 1E6, 'y-', label=r'$-P_{\text{rad}}^{' + node_plot + '}$')
            ax7.plot(t, Pe_tran_node / 1E6, 'm-', label=r'$P_{e,\text{tran}}^{' + node_plot + '}$')
            ax7.set_xlabel(r'$t$ [s]')
            ax7.set_ylabel(r'$P$ [MW/m$^{3}$]')
            ax7.legend(bbox_to_anchor=(1, 1), loc='upper left')
            ax7.grid('on')

        ax1.set_title('Scenario ' + str(scenario) + ' ' + node.capitalize())
        plt.tight_layout()

        fig_comment = node
        if comment is not None:
            fig_comment += '_' + comment
        fig_name = self.get_fig_path(scenario, int(np.rint(t.max())), fig_comment, fig_type)
        fig.savefig(fig_name)
        print('Figure saved: ' + fig_name)

        if show:
            plt.show()
        plt.close()

    def get_fig_path(self, scenario: int, pulse_len: int, comment: str, fig_type: str):
        """
        Get the figure path

        :param scenario: scenario number
        :param pulse_len: pulse length [s]
        :param comment: comment to the figure name
        :param fig_type: figure type
        :return: fig_path
        """
        fig_folder = os.path.join('.', 'figure', self.system_name)
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)

        fig_name = '{:s}_{:d}_t{:d}'.format(self.system_name, scenario, pulse_len)
        if comment is not None:
            fig_name += '_' + comment.strip().lower()
        fig_name += '.' + fig_type

        fig_path = os.path.join(fig_folder, fig_name)
        return fig_path

    def save(self, reactor: ReactorITER, y_sol: torch.Tensor, comment: str):
        """
        Save the solution

        :param reactor: reactor
        :param comment: comment
        :param y_sol: solution
        """
        t = tensor2array(reactor.time)
        sol_path = self.get_sol_path(reactor.scenario, int(np.rint(t.max())), comment)
        torch.save(y_sol, sol_path)
        print('Solution saved: ' + sol_path)

    def load(self, reactor: ReactorITER, comment: str) -> torch.Tensor:
        """
        Load the solution

        :param reactor: reactor
        :param comment: comment
        :return: y_sol = [ni [10^19 m^-3], Ti [keV], Te [keV]]
        """
        t = tensor2array(reactor.time)
        sol_path = self.get_sol_path(reactor.scenario, int(np.rint(t.max())), comment)
        y_sol = torch.load(sol_path)
        return y_sol

    def get_sol_path(self, scenario: int, pulse_len: int, comment: str):
        """
        Get the solution path

        :param scenario: scenario number
        :param pulse_len: pulse length [s]
        :param comment: comment to the solution name
        :return: sol_path
        """
        sol_folder = os.path.join('.', 'solution', self.system_name)
        if not os.path.exists(sol_folder):
            os.mkdir(sol_folder)

        sol_name = '{:s}_{:d}_t{:d}'.format(self.system_name, scenario, pulse_len)
        if comment is not None:
            sol_name += '_' + comment.strip().lower()
        sol_name += '.pt'

        sol_path = os.path.join(sol_folder, sol_name)
        return sol_path

    def split_sol(self, y_sol: torch.Tensor) -> dict:
        """
        Split the solution tensor into individual variables

        :param y_sol: solution
        :return: sol = {nd_core, nt_core, na_core, ne_core, nd_edge, nt_edge, na_edge, ne_edge [m^-3],
        Td_core, Tt_core, Ta_core, Te_core, Td_edge, Tt_edge, Ta_edge, Te_edge [eV]}
        """
        assert len(y_sol) == 16
        sol = {'nd_core': y_sol[0], 'nt_core': y_sol[1], 'na_core': y_sol[2], 'ne_core': y_sol[3],
               'nd_edge': y_sol[4], 'nt_edge': y_sol[5], 'na_edge': y_sol[6], 'ne_edge': y_sol[7],
               'Td_core': y_sol[8], 'Tt_core': y_sol[9], 'Ta_core': y_sol[10], 'Te_core': y_sol[11],
               'Td_edge': y_sol[12], 'Tt_edge': y_sol[13], 'Ta_edge': y_sol[14], 'Te_edge': y_sol[15]}
        return sol


def solve_all_reactors(dim: int = 0):
    """
    Solve all reactors in the test set

    :param dim: dimension
    :return: None
    """
    for shot_num in config.shots_test:
        t_start = time.time()
        if dim == 0:
            reactor = Reactor0D(shot_num)
            solver = Solver0D()
        else:
            reactor = Reactor1D(shot_num)
            solver = Solver1D()
        try:
            y_sol, y_exp = solver.solve(reactor=reactor, plot=True, full_plot=True, fig_type='png')
        except AssertionError:
            print('Dimension {:d}, shot # {:d} failed [{:.2f} s]'
                  .format(reactor.dim, reactor.shot_num, time.time() - t_start))
        else:
            print('Dimension {:d}, shot # {:d} solved [{:.2f} s]'
                  .format(reactor.dim, reactor.shot_num, time.time() - t_start))


if __name__ == '__main__':
    pass
    # reactor = ReactorITER(scenario=2)
    # solver = SolverITER()
    # solver.solve(reactor, plot=True, full_plot=True, fig_type='png')
