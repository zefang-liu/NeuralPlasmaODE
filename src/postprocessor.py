"""
Postprocessors
"""
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.model import NodalLinearRegression, NodalLinearRegressionITER
from src.reactor import Reactor1D, ReactorITER
from src.solver import Solver1D, SolverITER
from src.trainer import Trainer, TrainerITER
from src.utils import eV2J, tensor2array, tensors2arrays


class Postprocessor0D(object):
    """
    Postprocessor for the 0D Dynamical System
    """

    def __init__(self):
        self.dim = 0


class Postprocessor1D(Postprocessor0D):
    """
    Postprocessor for the 1D Dynamical System
    """

    def __init__(self, net_name: str = None, init_net: bool = True):
        """
        Initialize Postprocessor1D

        :param net_name: network name
        :param init_net: initialize the network, reactor, and solver
        """
        super().__init__()
        self.dim = 1
        self.nodes = config.nodes
        self.num_inputs = config.num_inputs[self.dim]
        self.num_outputs = config.num_outputs[self.dim]
        self.system_name = 'system{:d}d'.format(self.dim)
        self.device = config.device
        self.figure_width = 4
        self.figure_height = 2
        self.figure_size = (self.figure_width, self.figure_height + 0.5)

        if init_net:
            self.net = NodalLinearRegression(self.num_inputs, self.num_outputs).to(self.device)
            self.trainer = Trainer(dim=self.dim)
            self.trainer.init_net(self.net)
            self.trainer.load_net(self.net, net_name)
            self.trainer.print_net(self.net)
            self.reactor = Reactor1D
            self.solver = Solver1D()

    def get_chi_tau(self, reactor: Reactor1D):
        """
        Get nodal diffusivities and inter-nodal transport times

        :param reactor: reactor
        :return: chi [m^2/s], tau [s]
        """
        t = reactor.time
        y_sol = self.solver.load(reactor, 'opt')
        sol = self.solver.split_sol(y_sol)

        nd_core = sol['nd_core']
        nd_edge = sol['nd_edge']
        nd_sol = sol['nd_sol']
        nc_core = reactor.get_nc_node(t, node='core')
        nc_edge = reactor.get_nc_node(t, node='edge')
        nc_sol = reactor.get_nc_node(t, node='sol')
        ne_core = nd_core + nc_core * reactor.impurity_charge
        ne_edge = nd_edge + nc_edge * reactor.impurity_charge
        ne_sol = nd_sol + nc_sol * reactor.impurity_charge
        Te_core = sol['Te_core']
        Te_edge = sol['Te_edge']
        Te_sol = sol['Te_sol']

        a = reactor.get_a(t)
        R0 = reactor.get_R0(t)
        B0 = reactor.get_B0(t)
        kappa = reactor.get_kappa0(t)

        x_core = reactor.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t,
                                    node='core')
        x_edge = reactor.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t,
                                    node='edge')
        x_sol = reactor.get_x_node(ne_core, ne_edge, ne_sol, Te_core, Te_edge, Te_sol, a, R0, kappa, B0, t,
                                   node='sol')

        chi = {'core': reactor.get_chi_node(x_core, node='core'),
               'edge': reactor.get_chi_node(x_edge, node='edge'),
               'sol': reactor.get_chi_node(x_sol, node='sol')}

        tau = {'core': reactor.get_tau_node(chi['core'], a, node='core'),
               'edge': reactor.get_tau_node(chi['edge'], a, node='edge'),
               'sol': reactor.get_tau_node(chi['sol'], a, node='sol')}

        return chi, tau

    def plot_diffusivities(self):
        """
        Plot diffusivities

        :return: None
        """
        for shot_num in config.shots_test:
            reactor = self.reactor(shot_num, tuned=True, net=self.net)
            t = tensor2array(reactor.time)
            chi, _ = self.get_chi_tau(reactor)
            for node in self.nodes:
                chi[node] = tensor2array(chi[node].transpose(0, 1))

            for node in self.nodes:
                node_plot = r'\text{' + node + '}'
                plt.figure(figsize=self.figure_size)
                plt.semilogy(t, chi[node][0, :], 'b-', label=r'$D_{D}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][1, :], 'g-', label=r'$\chi_{D}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][2, :], 'r-', label=r'$\chi_{e}^{' + node_plot + '}$')
                plt.xlabel(r'$t$ [s]')
                plt.ylabel(r'$D$ or $\chi$ [m$^2$/s]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title('Shot ' + str(reactor.shot_num) + ' ' + node.capitalize())
                plt.grid(True)
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_diff')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_diff.pdf'.format(self.system_name, shot_num, node)
                fig_path = os.path.join(fig_folder, fig_name)
                plt.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def plot_transport_times(self):
        """
        Plot transport times

        :return: None
        """
        for shot_num in config.shots_test:
            reactor = self.reactor(shot_num, tuned=True, net=self.net)
            t = tensor2array(reactor.time)
            _, tau = self.get_chi_tau(reactor)
            for node in self.nodes:
                tau[node] = tensors2arrays(tuple(map(torch.squeeze, tau[node].split(1, dim=-1))))
            nodes_plot = self.nodes + ['div']

            for i, node1 in enumerate(self.nodes):
                tau_pd, tau_ed, tau_ee = tau[node1]
                node2 = nodes_plot[i + 1]
                node1_plot = r'\text{' + node1 + '}'
                node2_plot = r'\text{' + node2 + '}'
                plt.figure(figsize=self.figure_size)
                plt.semilogy(t, tau_pd, 'b-', label=r'$\tau_{P,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_ed, 'g-', label=r'$\tau_{E,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_ee, 'r-', label=r'$\tau_{E,e}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.xlabel(r'$t$ [s]')
                plt.ylabel(r'$\tau$ [s]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig_title = 'Shot ' + str(reactor.shot_num) + ' ' + node1.capitalize() + r'$\rightarrow$' \
                            + node2.capitalize()
                plt.title(fig_title)
                plt.grid(True)
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_tran')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_{:s}_tran.pdf'.format(self.system_name, shot_num, node1, node2)
                fig_path = os.path.join(fig_folder, fig_name)
                plt.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def plot_diffusivities_and_transport_times(self):
        """
        Plot diffusivities and transport times

        :return: None
        """
        for shot_num in config.shots_test:
            reactor = self.reactor(shot_num, tuned=True, net=self.net)
            t = tensor2array(reactor.time)
            chi, tau = self.get_chi_tau(reactor)
            for node in self.nodes:
                chi[node] = tensor2array(chi[node].transpose(0, 1))
                tau[node] = tensors2arrays(tuple(map(torch.squeeze, tau[node].split(1, dim=-1))))
            nodes_plot = self.nodes + ['div']
            num_rows = 2

            for i, node1 in enumerate(self.nodes):
                tau_pd, tau_ed, tau_ee = tau[node1]
                node2 = nodes_plot[i + 1]
                node1_plot = r'\text{' + node1 + '}'
                node2_plot = r'\text{' + node2 + '}'
                fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))

                ax1 = fig.add_subplot(num_rows, 1, 1)
                ax1.semilogy(t, chi[node1][0, :], 'b-', label=r'$D_{D}^{' + node1_plot + '}$')
                ax1.semilogy(t, chi[node1][1, :], 'g-', label=r'$\chi_{D}^{' + node1_plot + '}$')
                ax1.semilogy(t, chi[node1][2, :], 'r-', label=r'$\chi_{e}^{' + node1_plot + '}$')
                ax1.set_ylabel(r'$D$ or $\chi$ [m$^2$/s]')
                ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax1.grid('on')

                ax2 = fig.add_subplot(num_rows, 1, 2)
                ax2.semilogy(t, tau_pd, 'b-', label=r'$\tau_{P,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                ax2.semilogy(t, tau_ed, 'g-', label=r'$\tau_{E,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                ax2.semilogy(t, tau_ee, 'r-', label=r'$\tau_{E,e}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                ax2.set_xlabel(r'$t$ [s]')
                ax2.set_ylabel(r'$\tau$ [s]')
                ax2.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax2.grid('on')

                ax1.set_title('Shot ' + str(reactor.shot_num) + ' ' + node1.capitalize())
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_diff_tran')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_diff_tran.pdf'.format(self.system_name, shot_num, node1)
                fig_path = os.path.join(fig_folder, fig_name)
                plt.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()


class PostprocessorITER(Postprocessor1D):
    """
    Postprocessor for the 1D ITER Dynamical System
    """

    def __init__(self, net_name: str = None, is_inductive: bool = True, start_up: bool = False, comment: str = 'opt'):
        """
        Initialize PostprocessorITER

        :param net_name: network name
        :param is_inductive: True for inductive scenarios
        :param start_up: True to include the start-up process
        :param comment: comment in file names
        """
        super().__init__(net_name=None, init_net=False)
        self.nodes = config.nodes_iter
        self.num_outputs = config.num_outputs_iter[self.dim]
        self.net_name = net_name
        self.start_up = start_up
        self.comment = comment + '_start' if self.start_up else ''
        self.num_dashes = 80
        self.figure_width = 3.8
        self.figure_height = 1.8

        if is_inductive:
            self.system_name = 'system{:d}d_iter_ind'.format(self.dim)
            self.scenarios_test = config.inductive_scenarios_test
        else:
            self.system_name = 'system{:d}d_iter_non'.format(self.dim)
            self.scenarios_test = config.non_inductive_scenarios_test

        self.net = NodalLinearRegressionITER(self.num_inputs, self.num_outputs).to(self.device)
        self.trainer = TrainerITER(is_inductive=is_inductive, start_up=start_up)
        self.trainer.init_net(self.net)
        self.trainer.load_net(self.net, self.net_name)
        self.trainer.print_net(self.net)
        self.reactor = ReactorITER
        self.solver = SolverITER(is_inductive=is_inductive)
        self.y_name2index = {'nd_core': 0, 'nt_core': 1, 'na_core': 2, 'ne_core': 3,
                             'nd_edge': 4, 'nt_edge': 5, 'na_edge': 6, 'ne_edge': 7,
                             'Td_core': 8, 'Tt_core': 9, 'Ta_core': 10, 'Te_core': 11,
                             'Td_edge': 12, 'Tt_edge': 13, 'Ta_edge': 14, 'Te_edge': 15}
        self.chi_name2index = {'D_D': 0, 'D_a': 1, 'chi_D': 2, 'chi_a': 3, 'chi_e': 4}

    def check_solutions(self, reactor: ReactorITER = None, scenario: int = None, comment: str = None):
        """
        Check if solutions exist or not

        :param reactor: reactor model
        :param scenario: scenario number
        :param comment: comment string
        :return: None
        """
        if scenario is None:
            scenarios = self.scenarios_test
        else:
            scenarios = [scenario]

        if comment is None:
            comment = self.comment

        with torch.no_grad():
            for scenario in scenarios:
                if reactor is None:
                    reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
                reactor.sensitivity_analysis = True
                try:
                    _ = self.solver.load(reactor, comment=comment)
                except FileNotFoundError:
                    _, _ = self.solver.solve(reactor, plot=False, save=True, start_up=self.start_up, comment=comment)

    def compare_delay_effect(self, scenario: int):
        """
        Compare the delay effect of fusion heating

        :param scenario: scenario number
        :return: None
        """
        with torch.no_grad():
            reactor = self.reactor(scenario=scenario, tuned=True, net=self.net, delayed_fusion=False)
            _, _, sol_plot, _ = self.solver.solve_iter(reactor=reactor, start_up=self.start_up)
            del reactor

            reactor_delay = self.reactor(scenario=scenario, tuned=True, net=self.net, delayed_fusion=True)
            _, _, sol_plot_delay, _ = self.solver.solve_iter(reactor=reactor_delay, start_up=self.start_up)
            self.solver.plot(reactor=reactor_delay, sol=sol_plot_delay, exp=sol_plot, full_plot=False, fig_type='pdf',
                             comment='delay')
            del reactor_delay

    def get_chi_tau(self, reactor: ReactorITER):
        """
        Get nodal diffusivities and inter-nodal transport times

        :param reactor: reactor
        :return: chi [m^2/s], tau [s]
        """
        reactor.sensitivity_analysis = True
        self.check_solutions(reactor, scenario=reactor.scenario, comment=self.comment)
        y_sol = self.solver.load(reactor, comment=self.comment)
        sol = self.solver.split_sol(y_sol)

        ne_core = sol['ne_core']
        ne_edge = sol['ne_edge']
        Te_core = sol['Te_core']
        Te_edge = sol['Te_edge']

        t = reactor.time
        a = reactor.get_a(t)
        R0 = reactor.get_R0(t)
        B0 = reactor.get_B0(t)
        kappa = reactor.get_kappa95(t)

        x_core = reactor.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='core')
        x_edge = reactor.get_x_node(ne_core, ne_edge, None, Te_core, Te_edge, None, a, R0, kappa, B0, t, node='edge')

        chi = {'core': reactor.get_chi_node(x_core, node='core'),
               'edge': reactor.get_chi_node(x_edge, node='edge')}
        tau = {'core': reactor.get_tau_node(chi['core'], a, node='core'),
               'edge': reactor.get_tau_node(chi['edge'], a, node='edge')}

        return chi, tau

    def plot_diffusivities(self):
        """
        Plot nodal diffusivities

        :return: None
        """
        for scenario in self.scenarios_test:
            reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
            t = tensor2array(reactor.time)
            chi, _ = self.get_chi_tau(reactor)
            for node in self.nodes:
                chi[node] = tensor2array(chi[node].transpose(0, 1))

            for node in self.nodes:
                node_plot = r'\text{' + node + '}'
                plt.figure(figsize=self.figure_size)
                plt.semilogy(t, chi[node][0, :], 'r--', label=r'$D_{D}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][1, :], 'm--', label=r'$D_{\alpha}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][2, :], 'r-', label=r'$\chi_{D}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][3, :], 'm-', label=r'$\chi_{\alpha}^{' + node_plot + '}$')
                plt.semilogy(t, chi[node][4, :], 'b-', label=r'$\chi_{e}^{' + node_plot + '}$')
                plt.xlabel(r'$t$ [s]')
                plt.ylabel(r'$D$ or $\chi$ [m$^2$/s]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title('Scenario ' + str(reactor.shot_num) + ' ' + node.capitalize())
                plt.grid(True)
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_diff')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_diff.pdf'.format(self.system_name, scenario, node)
                fig_path = os.path.join(fig_folder, fig_name)
                plt.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def plot_transport_times(self):
        """
        Plot inter-nodal transport times

        :return: None
        """
        for scenario in self.scenarios_test:
            reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
            t = tensor2array(reactor.time)
            _, tau = self.get_chi_tau(reactor)
            for node in self.nodes:
                tau[node] = tensors2arrays(tuple(map(torch.squeeze, tau[node].split(1, dim=-1))))
            nodes_plot = self.nodes + ['sol']

            for i, node1 in enumerate(self.nodes):
                tau_pd, tau_pa, tau_ed, tau_ea, tau_ee = tau[node1]
                node2 = nodes_plot[i + 1]
                node1_plot = r'\text{' + node1 + '}'
                node2_plot = r'\text{' + node2 + '}'
                plt.figure(figsize=self.figure_size)
                plt.semilogy(t, tau_pd, 'r--', label=r'$\tau_{P,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_pa, 'm--', label=r'$\tau_{P,\alpha}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_ed, 'r-', label=r'$\tau_{E,D}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_ea, 'm-', label=r'$\tau_{E,\alpha}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.semilogy(t, tau_ee, 'b-', label=r'$\tau_{E,e}^{' + node1_plot + r' \to ' + node2_plot + '}$')
                plt.xlabel(r'$t$ [s]')
                plt.ylabel(r'$\tau$ [s]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig_title = 'Scenario ' + str(reactor.shot_num) + ' ' + node1.capitalize() + r'$\rightarrow$' \
                            + node2.capitalize()
                plt.title(fig_title)
                plt.grid(True)
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_tran')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_{:s}_tran.pdf'.format(self.system_name, scenario, node1, node2)
                fig_path = os.path.join(fig_folder, fig_name)
                plt.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def plot_temperatures(self, scenario: int = None):
        """
        Plot densities and temperatures

        :param scenario: scenario number
        :return: None
        """
        if scenario is None:
            scenarios = self.scenarios_test
        else:
            scenarios = [scenario]

        for scenario in scenarios:
            reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
            reactor.sensitivity_analysis = True
            solution = self.get_solution(reactor=reactor, comment=self.comment)

            for node in reactor.nodes:
                t = solution['t']
                num_rows = 2
                fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))
                node_plot = r'\text{' + node + '}'

                ax1 = fig.add_subplot(num_rows, 1, 1)
                ax1.plot(t, solution['nd_' + node] / 1E19, 'r-', label=r'$n_{D}^{' + node_plot + '}$')
                ax1.plot(t, solution['na_' + node] * 10 / 1E19, 'm-',
                         label=r'$n_{\alpha}^{' + node_plot + r'} \times 10$')
                ax1.plot(t, solution['ne_' + node] / 1E19, 'b-', label=r'$n_e^{' + node_plot + '}$')
                ax1.set_ylabel(r'$n_{\sigma}^{' + node_plot + '}$ [$10^{19}$ m$^{-3}$]')
                ax1.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
                ax1.grid('on')

                ax2 = fig.add_subplot(num_rows, 1, 2)
                ax2.plot(t, solution['Td_' + node] / 1E3, 'r-', label=r'$T_D^{' + node_plot + '}$')
                ax2.plot(t, solution['Ta_' + node] / 1E3, 'm-', label=r'$T_{\alpha}^{' + node_plot + '}$')
                ax2.plot(t, solution['Te_' + node] / 1E3, 'b-', label=r'$T_e^{' + node_plot + '}$')
                ax2.set_ylabel(r'$T_{\sigma}^{' + node_plot + '}$ [keV]')
                ax2.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
                ax2.grid('on')

                ax1.set_title('Scenario ' + str(scenario) + ' ' + node.capitalize())
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_temp')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_temp.pdf'.format(self.system_name, scenario, node)
                fig_path = os.path.join(fig_folder, fig_name)
                fig.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def plot_powers(self, scenario: int = None, print_final: bool = True):
        """
        Plot powers

        :param scenario: scenario number
        :param print_final: print final results
        :return: None
        """
        if scenario is None:
            scenarios = self.scenarios_test
        else:
            scenarios = [scenario]

        for scenario in scenarios:
            reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
            reactor.sensitivity_analysis = True
            solution = self.get_solution(reactor=reactor, comment=self.comment)

            for node in reactor.nodes:
                t = solution['t']

                if print_final:
                    print('-' * self.num_dashes)
                    print(f'Scenario {scenario} {node} at t = {t[-1]:.2f} s')
                    print(f'ni_{node} = {solution["ni_" + node][-1] / 1e19:.2f} 10^19 m^-3, '
                          f'ne_{node} = {solution["ne_" + node][-1] / 1e19:.2f} 10^19 m^-3')
                    print(f'Ti_{node} = {solution["Ti_" + node][-1] / 1e3:.2f} keV, '
                          f'Te_{node} = {solution["Te_" + node][-1] / 1e3:.2f} keV')
                    print(f'Pi_fus_{node} = {solution["Pi_fus_" + node][-1] / 1e6:.2f} MW, '
                          f'Pe_fus_{node} = {solution["Pe_fus_" + node][-1] / 1e6:.2f} MW')
                    print(f'Pi_aux_{node} = {solution["Pi_aux_" + node][-1] / 1e6:.2f} MW, '
                          f'P_oh_{node} + Pe_aux_{node} = '
                          f'{(solution["P_oh_" + node][-1] + solution["Pe_aux_" + node][-1]) / 1e6:.2f} MW')
                    print(f'Q_ie_{node} = {solution["Qi_" + node][-1] / 1e6:.2f} MW')
                    print(f'P_ecr_{node} = {-solution["P_ecr_" + node][-1] / 1e6:.2f} MW, '
                          f'P_brem_{node} = {-solution["P_brem_" + node][-1] / 1e6:.2f} MW, '
                          f'P_imp_{node} = {-solution["P_imp_" + node][-1] / 1e6:.2f} MW')
                    print(f'Pi_tran_{node} = {solution["Pi_tran_" + node][-1] / 1e6:.2f} MW, '
                          f'Pe_tran_{node} = {solution["Pe_tran_" + node][-1] / 1e6:.2f} MW')
                    print(f'Pi_iol_{node} = {solution["Pi_iol_" + node][-1] / 1e6:.2f} MW')
                    print('-' * self.num_dashes)

                num_rows = 2
                fig = plt.figure(figsize=(self.figure_width, self.figure_height * num_rows))
                node_plot = r'\text{' + node + '}'

                ax1 = fig.add_subplot(num_rows, 1, 1)
                ax1.plot(t, solution["P_oh_" + node] / 1E6, 'k-', label=r'$P_{\text{oh}}^{' + node_plot + '}$')
                ax1.plot(t, solution["Pe_aux_" + node] / 1E6, 'r-', label=r'$P_{e,\text{aux}}^{' + node_plot + '}$')
                ax1.plot(t, solution["Pe_fus_" + node] / 1E6, 'b-', label=r'$P_{e,\text{fus}}^{' + node_plot + '}$')
                ax1.plot(t, solution["Qe_" + node] / 1E6, 'g-', label=r'$Q_{e}^{' + node_plot + '}$')
                ax1.plot(t, -solution["P_rad_" + node] / 1E6, 'y-', label=r'$P_{\text{rad}}^{' + node_plot + '}$')
                ax1.plot(t, solution["Pe_tran_" + node] / 1E6, 'm-', label=r'$P_{e,\text{tran}}^{' + node_plot + '}$')
                ax1.set_ylabel(r'$P_e^{' + node_plot + '}$ [MW]')
                ax1.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
                ax1.grid('on')

                ax2 = fig.add_subplot(num_rows, 1, 2)
                ax2.plot(t, solution["Pi_aux_" + node] / 1E6, 'r-', label=r'$P_{i,\text{aux}}^{' + node_plot + '}$')
                ax2.plot(t, solution["Pi_fus_" + node] / 1E6, 'b-', label=r'$P_{i,\text{fus}}^{' + node_plot + '}$')
                ax2.plot(t, solution["Qi_" + node] / 1E6, 'g-', label=r'$Q_{i}^{' + node_plot + '}$')
                ax2.plot(t, solution["Pi_tran_" + node] / 1E6, 'm-', label=r'$P_{i,\text{tran}}^{' + node_plot + '}$')
                if node == 'edge':
                    ax2.plot(t, solution["Pi_iol_" + node] / 1E6, 'c-', label=r'$P_{i,\text{iol}}^{' + node_plot + '}$')
                ax2.set_xlabel(r'$t$ [s]')
                ax2.set_ylabel(r'$P_i^{' + node_plot + '}$ [MW]')
                ax2.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
                ax2.grid('on')

                ax1.set_title('Scenario ' + str(scenario) + ' ' + node.capitalize())
                plt.tight_layout()

                fig_folder = os.path.join('.', 'figure', self.system_name + '_power')
                if not os.path.exists(fig_folder):
                    os.mkdir(fig_folder)
                fig_name = '{:s}_{:d}_{:s}_power.pdf'.format(self.system_name, scenario, node)
                fig_path = os.path.join(fig_folder, fig_name)
                fig.savefig(fig_path)
                print('Figure saved: ' + fig_path)
                plt.close()

    def get_solution(self, reactor: ReactorITER, comment: str):
        """
        Get a solution

        :param reactor: reactor
        :param comment: comment
        :return: solution
        """
        self.check_solutions(reactor, scenario=reactor.scenario, comment=comment)
        y_sol = self.solver.load(reactor=reactor, comment=comment)
        sol = self.solver.split_sol(y_sol)
        t = reactor.time

        nd_core = sol['nd_core']
        nt_core = sol['nt_core']
        na_core = sol['na_core']
        ne_core = sol['ne_core']
        nd_edge = sol['nd_edge']
        nt_edge = sol['nt_edge']
        na_edge = sol['na_edge']
        ne_edge = sol['ne_edge']
        Td_core = sol['Td_core']
        Tt_core = sol['Tt_core']
        Ta_core = sol['Ta_core']
        Te_core = sol['Te_core']
        Td_edge = sol['Td_edge']
        Tt_edge = sol['Tt_edge']
        Ta_edge = sol['Ta_edge']
        Te_edge = sol['Te_edge']

        Ud_core = nd_core * eV2J(Td_core) * 3 / 2
        Ut_core = nt_core * eV2J(Tt_core) * 3 / 2
        Ua_core = na_core * eV2J(Ta_core) * 3 / 2
        Ue_core = ne_core * eV2J(Te_core) * 3 / 2
        Ud_edge = nd_edge * eV2J(Td_edge) * 3 / 2
        Ut_edge = nt_edge * eV2J(Tt_edge) * 3 / 2
        Ua_edge = na_edge * eV2J(Ta_edge) * 3 / 2
        Ue_edge = ne_edge * eV2J(Te_edge) * 3 / 2

        _, _, _, Pd, Pt, Pa, Pe, Ps = \
            reactor.get_sources_iter((nd_core, nd_edge), (nt_core, nt_edge), (na_core, na_edge),
                                     (Ud_core, Ud_edge), (Ut_core, Ut_edge), (Ua_core, Ua_edge),
                                     (Ue_core, Ue_edge), t)

        Ti_core = reactor.get_Ti_node(nd_node=nd_core, nt_node=nt_core, na_node=na_core,
                                      Td_node=Td_core, Tt_node=Tt_core, Ta_node=Ta_core)
        Ti_edge = reactor.get_Ti_node(nd_node=nd_edge, nt_node=nt_edge, na_node=na_edge,
                                      Td_node=Td_edge, Tt_node=Tt_edge, Ta_node=Ta_edge)
        vol_core = tensor2array(reactor.get_vol_node(vol=reactor.get_vol(t), node='core'))
        vol_edge = tensor2array(reactor.get_vol_node(vol=reactor.get_vol(t), node='edge'))

        nd_core, nt_core, na_core, ne_core, Td_core, Tt_core, Ta_core, Te_core = tensors2arrays(
            (nd_core, nt_core, na_core, ne_core, Td_core, Tt_core, Ta_core, Te_core))
        nd_edge, nt_edge, na_edge, ne_edge, Td_edge, Tt_edge, Ta_edge, Te_edge = tensors2arrays(
            (nd_edge, nt_edge, na_edge, ne_edge, Td_edge, Tt_edge, Ta_edge, Te_edge))

        t = tensor2array(t)
        Ti_core = tensor2array(Ti_core)
        Ti_edge = tensor2array(Ti_edge)
        ni_core = nd_core + nt_core + na_core
        ni_edge = nd_edge + nt_edge + na_edge

        Pd_aux_core, Pd_aux_edge, Pd_fus_core, Pd_fus_edge, Qd_core, Qd_edge, Pd_tran_core, Pd_tran_edge, \
        Pd_iol_edge = tensors2arrays(Pd)
        Pt_aux_core, Pt_aux_edge, Pt_fus_core, Pt_fus_edge, Qt_core, Qt_edge, Pt_tran_core, Pt_tran_edge, \
        Pt_iol_edge = tensors2arrays(Pt)
        Pa_aux_core, Pa_aux_edge, Pa_fus_core, Pa_fus_edge, Qa_core, Qa_edge, Pa_tran_core, Pa_tran_edge, \
        Pa_iol_edge = tensors2arrays(Pa)
        Pe_aux_core, Pe_aux_edge, P_oh_core, P_oh_edge, Pe_fus_core, Pe_fus_edge, P_rad_core, P_rad_edge, \
        Qe_core, Qe_edge, Pe_tran_core, Pe_tran_edge = tensors2arrays(Pe)
        Q_ad_core, Q_at_core, Q_dt_core, Q_de_core, Q_te_core, Q_ae_core, \
        Q_ad_edge, Q_at_edge, Q_dt_edge, Q_de_edge, Q_te_edge, Q_ae_edge, \
        P_ecr_core, P_brem_core, P_imp_core, P_ecr_edge, P_brem_edge, P_imp_edge = tensors2arrays(Ps)

        Pi_aux_core = Pd_aux_core + Pt_aux_core + Pa_aux_core
        Pi_fus_core = Pd_fus_core + Pt_fus_core + Pa_fus_core
        Qi_core = Qd_core + Qt_core + Qa_core
        Pi_tran_core = Pd_tran_core + Pt_tran_core + Pa_tran_core
        Pi_iol_core = np.zeros_like(Pi_tran_core)

        Pi_aux_edge = Pd_aux_edge + Pt_aux_edge + Pa_aux_edge
        Pi_fus_edge = Pd_fus_edge + Pt_fus_edge + Pa_fus_edge
        Qi_edge = Qd_edge + Qt_edge + Qa_edge
        Pi_tran_edge = Pd_tran_edge + Pt_tran_edge + Pa_tran_edge
        Pi_iol_edge = Pd_iol_edge + Pt_iol_edge + Pa_iol_edge

        P_aux_core = Pi_aux_core + Pe_aux_core + P_oh_core
        P_fus_core = Pi_fus_core + Pe_fus_core
        P_tran_core = Pi_tran_core + Pe_tran_core

        P_aux_edge = Pi_aux_edge + Pe_aux_edge + P_oh_edge
        P_fus_edge = Pi_fus_edge + Pe_fus_edge
        P_tran_edge = Pi_tran_edge + Pe_tran_edge

        Qi_core[0] = Qi_core[1]
        Qi_edge[0] = Qi_edge[1]
        Qe_core[0] = Qe_core[1]
        Qe_edge[0] = Qe_edge[1]

        solution = {
            't': t,
            'nd_core': nd_core, 'nt_core': nt_core, 'na_core': na_core, 'ni_core': ni_core, 'ne_core': ne_core,
            'nd_edge': nd_edge, 'nt_edge': nt_edge, 'na_edge': na_edge, 'ni_edge': ni_edge, 'ne_edge': ne_edge,
            'Td_core': Td_core, 'Tt_core': Tt_core, 'Ta_core': Ta_core, 'Ti_core': Ti_core, 'Te_core': Te_core,
            'Td_edge': Td_edge, 'Tt_edge': Tt_edge, 'Ta_edge': Ta_edge, 'Ti_edge': Ti_edge, 'Te_edge': Te_edge,
            'Pi_aux_core': Pi_aux_core * vol_core, 'Pi_aux_edge': Pi_aux_edge * vol_edge,
            'Pe_aux_core': Pe_aux_core * vol_core, 'Pe_aux_edge': Pe_aux_edge * vol_edge,
            'P_oh_core': P_oh_core * vol_core, 'P_oh_edge': P_oh_edge * vol_edge,
            'Pi_fus_core': Pi_fus_core * vol_core, 'Pi_fus_edge': Pi_fus_edge * vol_edge,
            'Pe_fus_core': Pe_fus_core * vol_core, 'Pe_fus_edge': Pe_fus_edge * vol_edge,
            'Qi_core': Qi_core * vol_core, 'Qi_edge': Qi_edge * vol_edge,
            'Qe_core': Qe_core * vol_core, 'Qe_edge': Qe_edge * vol_edge,
            'P_rad_core': P_rad_core * vol_core, 'P_rad_edge': P_rad_edge * vol_edge,
            'P_ecr_core': P_ecr_core * vol_core, 'P_ecr_edge': P_ecr_edge * vol_edge,
            'P_brem_core': P_brem_core * vol_core, 'P_brem_edge': P_brem_edge * vol_edge,
            'P_imp_core': P_imp_core * vol_core, 'P_imp_edge': P_imp_edge * vol_edge,
            'Pi_tran_core': Pi_tran_core * vol_core, 'Pi_tran_edge': Pi_tran_edge * vol_edge,
            'Pe_tran_core': Pe_tran_core * vol_core, 'Pe_tran_edge': Pe_tran_edge * vol_edge,
            'Pi_iol_core': Pi_iol_core * vol_core, 'Pi_iol_edge': Pi_iol_edge * vol_edge,
            'P_aux_core': P_aux_core * vol_core, 'P_aux_edge': P_aux_edge * vol_edge,
            'P_fus_core': P_fus_core * vol_core, 'P_fus_edge': P_fus_edge * vol_edge,
            'P_tran_core': P_tran_core * vol_core, 'P_tran_edge': P_tran_edge * vol_edge
        }

        return solution

    def plot_sensitivity(self, scenario, mode, solutions, labels, y_labels, keys, divisors):
        """
        Plot one sensitivity

        :param scenario: scenario number
        :param mode: sensitivity analysis mode
        :param solutions: solutions
        :param labels: labels
        :param y_labels: y labels
        :param keys: keys in one solution
        :param divisors: divisors for each value
        :return: None
        """
        node = 'core'
        num_rows = 2
        num_cols = 2
        fig = plt.figure(figsize=(2.4 * num_cols, 1.6 * num_rows))
        formats = ['r--', 'b-', 'g-.', 'm:']

        axes = []
        for i in range(num_rows * num_cols):
            axes.append(fig.add_subplot(num_rows, num_cols, i + 1))
            axes[-1].set_ylabel(y_labels[i])
            axes[-1].grid(True)

        for i, label in enumerate(labels):
            solution = solutions[i]
            for j, (key, divisor) in enumerate(zip(keys, divisors)):
                axes[j].plot(solution['t'], solution[key] / divisor, formats[i], label=label)

        axes[2].set_xlabel(r'$t$ [s]')
        axes[3].set_xlabel(r'$t$ [s]')
        axes[1].legend(bbox_to_anchor=(1, 1), loc='upper left')
        fig.suptitle('Scenario ' + str(scenario) + ' ' + node.capitalize())
        plt.tight_layout()

        fig_folder = os.path.join('.', 'figure', self.system_name + '_sens')
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
        fig_name = '{:s}_{:d}_{:s}.pdf'.format(self.system_name, scenario, mode)
        fig_path = os.path.join(fig_folder, fig_name)
        fig.savefig(fig_path)
        print('Figure saved: ' + fig_path)
        plt.close()

    def print_changes(self, parameter_name: str, parameters: list, selected_index: int, solutions: List[dict]):
        """
        Print core temperature changes due to parameters

        :param parameter_name: parameter name
        :param parameters: parameter list
        :param selected_index: selected index in the parameter list
        :param solutions: solutions with core temperatures
        :return: None
        """
        parameter_original = parameters[selected_index]
        solution_original = solutions[selected_index]
        Td_core_original = solution_original['Td_core']
        Te_core_original = solution_original['Te_core']

        parameter_lower = parameters[0]
        solution_lower = solutions[0]
        Td_core_lower = solution_lower['Td_core']
        Te_core_lower = solution_lower['Te_core']

        parameter_upper = parameters[-1]
        solution_upper = solutions[-1]
        Td_core_upper = solution_upper['Td_core']
        Te_core_upper = solution_upper['Te_core']

        print(f'd{parameter_name} = {parameter_lower - parameter_original:.4f}: '
              f'dTd_core = {(Td_core_lower[-1] - Td_core_original[-1]) / 1E3:.3f} keV, '
              f'dTe_core = {(Te_core_lower[-1] - Te_core_original[-1]) / 1E3:.3f} keV')
        print(f'd{parameter_name} = {parameter_upper - parameter_original:.4f}: '
              f'dTd_core = {(Td_core_upper[-1] - Td_core_original[-1]) / 1E3:.3f} keV, '
              f'dTe_core = {(Te_core_upper[-1] - Te_core_original[-1]) / 1E3:.3f} keV')

    def plot_sensitivities(self, scenario: int, mode: str):
        """
        Plot sensitivities

        :param scenario: scenario number
        :param mode: mode for the sensitivity analysis
        :return: None
        """
        if mode == 'ecr_r':
            rs = [0.7, 0.8, 0.9]
            solutions = []

            for r in rs:
                reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
                reactor.ecr_parameters['r'] = r
                reactor.sensitivity_analysis = True

                comment = '{:s}_{:.1f}'.format(mode, r).replace('.', '_')
                solutions.append(self.get_solution(reactor=reactor, comment=comment))

            labels = [r'$r = {:.1f}$'.format(r) for r in rs]
            y_labels = [r'$T_D^{\text{core}}$ [keV]', r'$P_{\text{fus}}^{\text{core}}$ [MW]',
                        r'$T_e^{\text{core}}$ [keV]', r'$P_{\text{ECR}}^{\text{core}}$ [MW]']
            keys = ['Td_core', 'P_fus_core', 'Te_core', 'P_ecr_core']
            divisors = [1E3, 1E6, 1E3, 1E6]
            self.print_changes(mode, rs, 1, solutions)
            self.plot_sensitivity(scenario, mode, solutions, labels, y_labels, keys, divisors)

        elif mode in ['be', 'ar']:
            if mode == 'be':
                fractions = [0.01, 0.02, 0.03]
                num_digits = 3
            else:
                fractions = [0.0008, 0.0012, 0.0016]
                num_digits = 4
            solutions = []

            for fraction in fractions:
                reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
                reactor.impurity_fractions[mode] = fraction
                reactor.sensitivity_analysis = True

                comment = ('{:s}_{:.' + str(num_digits) + 'f}').format(mode, fraction).replace('.', '_')
                solutions.append(self.get_solution(reactor=reactor, comment=comment))

            get_label = lambda fraction: r'$f_{\text{' + mode.capitalize() + '}}$ = ' \
                                         + ('{:.' + str(num_digits - 2) + r'f} \%').format(fraction * 100)

            labels = [get_label(fraction) for fraction in fractions]
            y_labels = [r'$T_D^{\text{core}}$ [keV]', r'$P_{\text{fus}}^{\text{core}}$ [MW]',
                        r'$T_e^{\text{core}}$ [keV]', r'$P_{\text{imp}}^{\text{core}}$ [MW]']
            keys = ['Td_core', 'P_fus_core', 'Te_core', 'P_imp_core']
            divisors = [1E3, 1E6, 1E3, 1E6]
            self.print_changes(mode, fractions, 1, solutions)
            self.plot_sensitivity(scenario, mode, solutions, labels, y_labels, keys, divisors)

        elif mode == 'chi_core':
            chi_multipliers = [0.5, 1.0, 2.0]
            chi_names = ['chi_D', 'chi_a', 'chi_e']
            solutions = []

            for chi_multiplier in chi_multipliers:
                net = NodalLinearRegressionITER(self.num_inputs, self.num_outputs).to(self.device)
                self.trainer.init_net(net)
                self.trainer.load_net(net, self.net_name)

                for chi_name in chi_names:
                    chi_index = self.chi_name2index[chi_name]
                    net.fc_core.bias.data[chi_index] += torch.log(torch.tensor(chi_multiplier))

                self.trainer.print_net(net)
                reactor = self.reactor(scenario=scenario, tuned=True, net=net)
                reactor.sensitivity_analysis = True

                comment = '{:s}_{:.1f}'.format(mode, chi_multiplier).replace('.', '_')
                solutions.append(self.get_solution(reactor=reactor, comment=comment))

            labels = [r'$\chi_{\sigma}^{\text{core}} \times ' + r'{:.1f}$'.format(chi_multiplier)
                      for chi_multiplier in chi_multipliers]
            y_labels = [r'$T_D^{\text{core}}$ [keV]', r'$P_{\text{fus}}^{\text{core}}$ [MW]',
                        r'$T_e^{\text{core}}$ [keV]', r'$-P_{\text{tran}}^{\text{core}}$ [MW]']
            keys = ['Td_core', 'P_fus_core', 'Te_core', 'P_tran_core']
            divisors = [1E3, 1E6, 1E3, -1E6]
            self.print_changes(mode, chi_multipliers, 1, solutions)
            self.plot_sensitivity(scenario, mode, solutions, labels, y_labels, keys, divisors)

        elif mode == 'tau_iol':
            tau_iol_multipliers = [0.1, 1.0, 10.0]
            solutions = []

            for tau_iol_multiplier in tau_iol_multipliers:
                reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
                reactor.iol_parameters['tau_ps_iol_edge'] = tau_iol_multiplier
                reactor.iol_parameters['tau_es_iol_edge'] = tau_iol_multiplier
                reactor.sensitivity_analysis = True

                comment = '{:s}_{:.1f}'.format(mode, tau_iol_multiplier).replace('.', '_')
                solutions.append(self.get_solution(reactor=reactor, comment=comment))

            labels = [r'$\tau_{e,\text{IOL}}^{\text{edge}} \times ' + r'{:.1f}$'.format(tau_iol_multiplier) for
                      tau_iol_multiplier
                      in tau_iol_multipliers]
            y_labels = [r'$T_D^{\text{core}}$ [keV]', r'$P_{\text{fus}}^{\text{core}}$ [MW]',
                        r'$T_e^{\text{core}}$ [keV]', r'$P_{\text{IOL}}^{\text{edge}}$ [MW]']
            keys = ['Td_core', 'P_fus_core', 'Te_core', 'Pi_iol_edge']
            divisors = [1E3, 1E6, 1E3, 1E6]
            self.print_changes(mode, tau_iol_multipliers, 1, solutions)
            self.plot_sensitivity(scenario, mode, solutions, labels, y_labels, keys, divisors)

        else:
            raise KeyError(mode)

    def analyze_diffusivity_sensitivities(self, scenario: int):
        """
        Analyze diffusivity parameter sensitivities

        :param scenario: scenario number
        :return: None
        """
        reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
        reactor.sensitivity_analysis = True

        y_sol, _ = self.solver.solve(reactor, plot=False, save=False, start_up=self.start_up)
        y_names_required_core = ['nd_core', 'na_core', 'Td_core', 'Ta_core', 'Te_core']
        y_names_required_edge = ['nd_edge', 'na_edge', 'Td_edge', 'Ta_edge', 'Te_edge']

        gradients_core = torch.zeros(len(y_names_required_core),
                                     self.net.fc_core.weight.shape[1] + 1).double()
        gradients_edge = torch.zeros(len(y_names_required_edge),
                                     self.net.fc_edge.weight.shape[1] + 1).double()

        for i, y_name in enumerate(y_names_required_core):
            y_index = self.y_name2index[y_name]
            y_sol_i = y_sol[y_index, -1]
            y_sol_i.backward(torch.ones_like(y_sol_i), retain_graph=True)

            gradients_core[i][0] = self.net.fc_core.bias.grad.clone()[i] / y_sol_i
            gradients_core[i][1:] = \
                self.net.fc_core.weight.grad.clone()[i] * self.net.fc_core.weight.clone()[i] / y_sol_i

            self.net.fc_core.bias.grad.zero_()
            self.net.fc_core.weight.grad.zero_()
            print(y_name, gradients_core[i])

        for i, y_name in enumerate(y_names_required_edge):
            y_index = self.y_name2index[y_name]
            y_sol_i = y_sol[y_index, -1]
            y_sol_i.backward(torch.ones_like(y_sol_i), retain_graph=True)

            gradients_edge[i][0] = self.net.fc_edge.bias.grad.clone()[i] / y_sol_i
            gradients_edge[i][1:] = \
                self.net.fc_edge.weight.grad.clone()[i] * self.net.fc_edge.weight.clone()[i] / y_sol_i

            self.net.fc_edge.bias.grad.zero_()
            self.net.fc_edge.weight.grad.zero_()
            print(y_name, gradients_edge[i])

    def analyze_sensitivities(self, scenario: int, parameter_name: str,
                              parameters_required: List[str], y_names_required: List[str]):
        """
        Analyze sensitivities

        :param scenario: scenario number
        :param parameter_name: parameter name in the reactor shot
        :param parameters_required: required parameters
        :param y_names_required: required y names
        :return: None
        """
        reactor = self.reactor(scenario=scenario, tuned=True, net=self.net)
        reactor.sensitivity_analysis = True
        parameters = reactor.shot[parameter_name]

        for parameter in parameters.keys():
            if parameter in parameters_required:
                parameters[parameter].requires_grad = True

        y_sol, _ = self.solver.solve(reactor, plot=False, save=False, start_up=self.start_up)
        gradients = torch.zeros(len(y_names_required), len(parameters_required)).double()
        print(parameters_required)

        for i, y_name in enumerate(y_names_required):
            y_index = self.y_name2index[y_name]
            y_sol_i = y_sol[y_index, -1]
            y_sol_i.backward(torch.ones_like(y_sol_i), retain_graph=True)
            for j, parameter in enumerate(parameters.keys()):
                if parameter in parameters_required:
                    gradients[i][j] = parameters[parameter].grad.clone() \
                                      * parameters[parameter].clone() / y_sol_i
                    parameters[parameter].grad = None
            print(y_name, gradients[i].detach())

    def analyze_ecr_sensitivities(self, scenario: int):
        """
        Analyze ECR parameter sensitivities

        :param scenario: scenario number
        :return: None
        """
        parameters_required = ['alpha_ne', 'alpha_te', 'beta_te', 'r']
        y_names_required = ['Te_core', 'Te_edge']
        self.analyze_sensitivities(scenario=scenario, parameter_name='ecr_parameters',
                                   parameters_required=parameters_required, y_names_required=y_names_required)

    def analyze_impurity_sensitivities(self, scenario: int):
        """
        Analyze impurity fraction sensitivities

        :param scenario: scenario number
        :return: None
        """
        parameters_required = ['be', 'ar']
        y_names_required = ['Te_core', 'Te_edge']
        self.analyze_sensitivities(scenario=scenario, parameter_name='impurity_fractions',
                                   parameters_required=parameters_required, y_names_required=y_names_required)

    def analyze_iol_sensitivities(self, scenario: int):
        """
        Analyze IOL parameter sensitivities

        :param scenario: scenario number
        :return: None
        """
        parameters_required = ['tau_ps_iol_edge', 'tau_es_iol_edge']
        y_names_required = ['nd_edge', 'na_edge', 'Td_edge', 'Ta_edge']
        self.analyze_sensitivities(scenario=scenario, parameter_name='iol_parameters',
                                   parameters_required=parameters_required, y_names_required=y_names_required)


if __name__ == '__main__':
    # postprocessor = Postprocessor1D(net_name='net1d_ep24_re20.pt')
    # postprocessor.plot_diffusivities()
    # postprocessor.plot_transport_times()
    # postprocessor.plot_diffusivities_and_transport_times()

    postprocessor = PostprocessorITER(
        net_name='net1d_ep14_re10.pt', is_inductive=True, start_up=True, comment='ep14')
    # postprocessor = PostprocessorITER(
    #     net_name='net1d_ep02_re10.pt', is_inductive=False, start_up=True, comment='ep02')
    # postprocessor.plot_temperatures()
    # postprocessor.plot_powers()
    # postprocessor.plot_diffusivities()
    # postprocessor.plot_transport_times()
    # postprocessor.compare_delay_effect(scenario=2)
    # postprocessor.analyze_diffusivity_sensitivities(scenario=2)
    # postprocessor.analyze_ecr_sensitivities(scenario=2)
    # postprocessor.analyze_impurity_sensitivities(scenario=2)
    # postprocessor.analyze_iol_sensitivities(scenario=2)
    # postprocessor.plot_sensitivities(scenario=2, mode='ecr_r')
    # postprocessor.plot_sensitivities(scenario=2, mode='chi_core')
    # postprocessor.plot_sensitivities(scenario=2, mode='be')
    # postprocessor.plot_sensitivities(scenario=2, mode='ar')
    # postprocessor.plot_sensitivities(scenario=2, mode='tau_iol')
