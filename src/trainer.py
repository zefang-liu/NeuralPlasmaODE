"""
Trainer for Confinement Time Fitting Models
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src import config
from src.model import LinearRegression, NodalLinearRegression, NodalLinearRegressionITER
from src.reactor import Reactor0D, Reactor1D, ReactorITER
from src.solver import Solver0D, Solver1D, SolverITER


class Trainer(object):
    """
    Regression Trainer
    """

    def __init__(self, dim: int = 0, mini: bool = False, has_sol: bool = True):
        """
        Initialize the trainer

        :param dim: reactor dimension
        :param mini: use mini datasets
        :param has_sol: has the SOL node
        """
        torch.manual_seed(0)  # reproducibility

        self.dim = dim
        self.has_sol = has_sol
        self.num_inputs = config.num_inputs[self.dim]
        self.num_outputs = config.num_outputs[self.dim]
        self.system_name = 'system{:d}d'.format(self.dim)
        self.folder_path = os.path.join('.', 'network', self.system_name)
        self.num_dashes = 80
        self.tensor_template = lambda t: torch.tensor([t] * self.num_outputs, dtype=torch.double, device=self.device)
        self.tensor_template_one = lambda t: torch.tensor(t, dtype=torch.double, device=self.device)

        if mini:
            self.shots_train = [131191, 131195]
            self.shots_test = [131190]
        else:
            self.shots_train = config.shots_train
            self.shots_test = config.shots_test

        self.epoch_num = config.epoch_num
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.eval_step = config.eval_step
        self.eval_start = config.eval_start
        self.eval_end = config.eval_end
        self.save_step = config.save_step
        self.save_reactor = config.save_reactor
        self.load_network = config.load_network
        self.load_pretrained = config.load_pretrained
        self.train_core_edge = config.train_core_edge
        self.train_sol = config.train_sol
        self.device = config.device

        self.epoch_start = 0
        self.reactor_start = 0
        self.solve_step = None
        self.num_train_reactors = len(self.shots_train)
        if self.load_pretrained:
            self.pretrained_network = config.pretrained_network

        if self.dim == 0:
            self.net = LinearRegression(self.num_inputs, self.num_outputs).to(self.device)
            self.reactor = Reactor0D
            self.solver = Solver0D()
        else:
            self.net = NodalLinearRegression(self.num_inputs, self.num_outputs).to(self.device)
            self.reactor = Reactor1D
            self.solver = Solver1D()

        self.criterion = None
        self.optimizer = None

    def init_net_opt(self):
        """
        Initialize the network and optimizer

        :return: None
        """
        self.init_net(self.net)
        if self.load_network:
            self.load_net(self.net)

        self.criterion = nn.MSELoss()

        if self.train_core_edge:
            params = list(self.net.fc_core.parameters()) + list(self.net.fc_edge.parameters())
            self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.train_sol:
            self.optimizer = optim.Adam(self.net.fc_sol.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def init_net(self, net: nn.Module):
        """
        Initialize weights and biases for the network

        :param net: network for initialization
        :return: None
        """
        if self.dim == 0:
            # Initialize weights by tau-h98 laws
            weights = [0.93, 0.15, 0.41, -0.69, 1.97, 0.78, -0.58, 0.19]
            bias = 0.0562
            net.fc.weight.data = self.tensor_template(weights)
            net.fc.bias.data = torch.log(self.tensor_template(bias))

        else:
            if not self.load_pretrained:
                # Weights and bias from the chi-h98 law
                weights = [-3.5, 0.9, 1.0, 1.2, 3.0, -2.9, -0.6, 0.7, -0.2]
                bias = 0.123

                # Initialize weights and bias by the chi-h98 law
                net.fc_core.weight.data = self.tensor_template(weights)
                net.fc_core.bias.data = torch.log(self.tensor_template(bias))
                net.fc_core.bias.data[0] += torch.log(torch.tensor(0.6))

                net.fc_edge.weight.data = self.tensor_template(weights)
                net.fc_edge.bias.data = torch.log(self.tensor_template(bias))
                net.fc_edge.bias.data[0] += torch.log(torch.tensor(0.6))

                if self.has_sol:
                    net.fc_sol.weight.data = self.tensor_template(weights)
                    net.fc_sol.bias.data = torch.log(self.tensor_template(bias))
                    net.fc_sol.bias.data[0] += torch.log(torch.tensor(0.6))

            else:
                # Initialize weights and bias by the pretrained network
                net.fc_core.weight.data = self.tensor_template_one(self.pretrained_network['weights_core'])
                net.fc_core.bias.data = torch.log(self.tensor_template_one(self.pretrained_network['bias_core']))

                net.fc_edge.weight.data = self.tensor_template_one(self.pretrained_network['weights_edge'])
                net.fc_edge.bias.data = torch.log(self.tensor_template_one(self.pretrained_network['bias_edge']))

                if self.has_sol:
                    net.fc_sol.weight.data = self.tensor_template_one(self.pretrained_network['weights_sol'])
                    net.fc_sol.bias.data = torch.log(self.tensor_template_one(self.pretrained_network['bias_sol']))

    def train_net(self):
        """
        Train the network

        :return: None
        """
        self.print_net(self.net)

        # Evaluate unoptimized reactors
        if self.eval_start:
            self.eval_net(epoch=self.epoch_start)

        print('-' * self.num_dashes)
        print('Training system {:d}d ...'.format(self.dim))
        print('-' * self.num_dashes)
        t_train = time.time()

        for epoch in range(self.epoch_start + 1, self.epoch_num + 1):
            print('-' * self.num_dashes)
            print('Epoch {:d}, lr = {:f}'.format(epoch, self.lr))
            print('-' * self.num_dashes)
            t_epoch = time.time()

            for i_reactor in range(self.reactor_start + 1, self.num_train_reactors + 1):
                if not self.check_net(self.net):
                    self.print_net(self.net)
                    raise ValueError('There is nan in the network.')

                t_reactor = time.time()
                shot_num = self.shots_train[i_reactor - 1]
                reactor = self.reactor(shot_num, tuned=True, net=self.net)
                self.optimizer.zero_grad()

                try:
                    if self.train_core_edge:
                        y_sol, y_exp = self.solver.solve_core_edge(reactor)
                    elif self.train_sol:
                        y_sol, y_exp = self.solver.solve_sol(reactor)
                    else:
                        if self.solve_step is None:
                            y_sol, y_exp = self.solver.solve(reactor, plot=False, save=False)
                        else:
                            y_sol, y_exp, _, _ = self.solver.solve_steps(reactor, step_size=self.solve_step)

                except AssertionError:
                    print('Epoch {:d}, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, failed [{:.2f} s/{:.2f} s]'
                          .format(epoch, i_reactor, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, time.time() - t_reactor, time.time() - t_epoch))
                else:
                    loss = self.criterion(y_sol, y_exp)
                    loss.backward()
                    self.optimizer.step()
                    print('Epoch {:d}, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, loss: {:.4f} [{:.2f} s/{:.2f} s]'
                          .format(epoch, i_reactor, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, loss.item(), time.time() - t_reactor,
                                  time.time() - t_epoch))
                finally:
                    del reactor

                if self.save_reactor:
                    self.save_net(epoch=epoch, i_reactor=i_reactor, verbose=False)

            self.reactor_start = 0
            print('-' * self.num_dashes)

            if not self.save_reactor and epoch % self.save_step == 0:
                self.save_net(epoch=epoch, i_reactor=self.num_train_reactors)

            if epoch % self.eval_step == 0:
                self.print_net(self.net)
                self.eval_net(epoch=epoch)

        print('-' * self.num_dashes)
        print('Training finished in [{:.2f} s]'.format(time.time() - t_train))
        print('-' * self.num_dashes)
        self.print_net(self.net)

        # Save and evaluate optimized reactors if not done yet
        if not self.save_reactor and self.epoch_num % self.save_step != 0:
            self.save_net(epoch=self.epoch_num, i_reactor=self.num_train_reactors)

        if self.eval_end and self.epoch_num % self.eval_step != 0:
            self.eval_net(epoch=self.epoch_num)

    def check_net(self, net: nn.Module) -> bool:
        """
        Check if there is nan in the network

        :return: False for nan
        """
        if self.dim == 0:
            if torch.isnan(net.fc.weight.data).any() or torch.isnan(net.fc.bias.data).any():
                return False
        elif self.dim == 1:
            if torch.isnan(net.fc_core.weight.data).any() or torch.isnan(net.fc_core.bias.data).any():
                return False
            if torch.isnan(net.fc_edge.weight.data).any() or torch.isnan(net.fc_edge.bias.data).any():
                return False
            if self.has_sol and (torch.isnan(net.fc_sol.weight.data).any() or torch.isnan(net.fc_sol.bias.data).any()):
                return False
        return True

    def eval_net(self, epoch: int):
        """
        Evaluate the model

        :param epoch: epoch number
        :return: None
        """
        print('-' * self.num_dashes)
        print('Evaluating system {:d}d ...'.format(self.dim))

        t_test = time.time()
        with torch.no_grad():
            for i, shot_num in enumerate(self.shots_test):
                if not self.check_net(self.net):
                    self.print_net(self.net)
                    raise ValueError('There is nan in the network.')

                t_reactor = time.time()
                reactor = self.reactor(shot_num, tuned=True, net=self.net)
                try:
                    if self.train_core_edge:
                        y_sol, y_exp = self.solver.solve_core_edge(reactor)
                    elif self.train_sol:
                        y_sol, y_exp = self.solver.solve_sol(reactor)
                    else:
                        if self.solve_step is None:
                            y_sol, y_exp = self.solver.solve(reactor, plot=True, full_plot=False, fig_type='pdf',
                                                             comment='ep{:02d}'.format(epoch))
                        else:
                            y_sol, y_exp = self.solver.solve(reactor, plot=True, full_plot=False, fig_type='pdf',
                                                             comment='st{:d}_ep{:02d}'.format(self.solve_step, epoch))
                except AssertionError:
                    print('Testing, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, failed [{:.2f} s/{:.2f} s]'
                          .format(i + 1, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, time.time() - t_reactor, time.time() - t_test))
                else:
                    loss = self.criterion(y_sol, y_exp)
                    print('Testing, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, loss: {:.4f} [{:.2f} s/{:.2f} s]'
                          .format(i + 1, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, loss.item(), time.time() - t_reactor, time.time() - t_test))
                finally:
                    del reactor

        print('-' * self.num_dashes)

    def get_net_path(self, epoch: int, i_reactor: int):
        """
        Get the network path

        :param epoch: epoch number
        :param i_reactor: index of the reactor
        :return: net_path
        """
        net_name = 'net{:d}d'.format(self.dim)
        if self.solve_step is not None:
            net_name += '_st{:d}'.format(self.solve_step)
        net_name += '_ep{:02d}_re{:02d}.pt'.format(epoch, i_reactor)

        net_path = os.path.join(self.folder_path, net_name)
        return net_path

    def save_net(self, epoch: int, i_reactor: int, verbose: bool = True):
        """
        Save the model

        :param epoch: epoch number
        :param i_reactor: index of the reactor
        :param verbose: verbose
        :return: None
        """
        if not self.check_net(self.net):
            self.print_net(self.net)
            raise ValueError('There is nan in the network.')

        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

        net_path = self.get_net_path(epoch, i_reactor)
        torch.save(self.net.state_dict(), net_path)

        if verbose:
            print('-' * self.num_dashes)
            print('Network saved: ' + net_path)
            print('-' * self.num_dashes)

    def load_net(self, net: nn.Module, net_name: str = None):
        """
        Load the model if exists

        :param net: network model
        :param net_name: network name
        :return: None
        """
        if net_name is None:
            loaded = False

            for epoch in range(self.epoch_num, 0, -1):
                for i_reactor in range(self.num_train_reactors, 0, -1):
                    net_path = self.get_net_path(epoch, i_reactor)
                    if os.path.exists(net_path):
                        if i_reactor == self.num_train_reactors:
                            self.epoch_start = epoch
                            self.reactor_start = 0
                        else:
                            self.epoch_start = epoch - 1
                            self.reactor_start = i_reactor
                        net.load_state_dict(torch.load(net_path))
                        net.eval()
                        print('-' * self.num_dashes)
                        print('Network loaded: ' + net_path)
                        print('-' * self.num_dashes)
                        loaded = True
                        break
                if loaded:
                    break
        else:
            network_folder_path = os.path.join('.', 'network', self.system_name)
            net_path = os.path.join(network_folder_path, net_name)
            if not os.path.exists(net_path):
                raise FileNotFoundError(net_path)
            net.load_state_dict(torch.load(net_path))
            net.eval()

    def print_net(self, net: nn.Module):
        """
        Print network parameters

        :return: None
        """
        torch.set_printoptions(sci_mode=False)
        print('-' * self.num_dashes)
        if self.dim == 0:
            print('weight = ')
            print(net.fc.weight.data)
            print('bias = ')
            print(torch.exp(net.fc.bias.data))
        else:
            print('weight_core = ')
            print(net.fc_core.weight.data)
            print('bias_core = ')
            print(torch.exp(net.fc_core.bias.data))
            print('weight_edge = ')
            print(net.fc_edge.weight.data)
            print('bias_edge = ')
            print(torch.exp(net.fc_edge.bias.data))
            if self.has_sol:
                print('weight_sol = ')
                print(net.fc_sol.weight.data)
                print('bias_sol = ')
                print(torch.exp(net.fc_sol.bias.data))
        print('-' * self.num_dashes)
        torch.set_printoptions(sci_mode=None)


class TrainerITER(Trainer):
    """
    Regression Trainer for ITER
    """

    def __init__(self, is_inductive: bool = True, start_up: bool = False):
        """
        Initialize the trainer for ITER

        :param is_inductive: True for inductive scenarios
        :param start_up: True to include the start-up process
        """
        super().__init__(dim=1, mini=False, has_sol=False)

        if is_inductive:
            self.system_name = 'system{:d}d_iter_ind'.format(self.dim)
            self.shots_train = config.inductive_scenarios_train
            self.shots_test = config.inductive_scenarios_test
            self.pretrained_network = config.pretrained_network_iter_ind
        else:
            self.system_name = 'system{:d}d_iter_non'.format(self.dim)
            self.shots_train = config.non_inductive_scenarios_train
            self.shots_test = config.non_inductive_scenarios_test
            self.pretrained_network = config.pretrained_network_iter_non

        self.num_train_reactors = len(self.shots_train)
        self.num_outputs = config.num_outputs_iter[self.dim]
        self.folder_path = os.path.join('.', 'network', self.system_name)
        self.net = NodalLinearRegressionITER(self.num_inputs, self.num_outputs).to(self.device)
        self.reactor = ReactorITER
        self.solver = SolverITER(is_inductive=is_inductive)
        self.start_up = start_up

    def eval_net(self, epoch: int):
        """
        Evaluate the model

        :param epoch: epoch number
        :return: None
        """
        print('-' * self.num_dashes)
        print('Evaluating system {:d}d ...'.format(self.dim))

        t_test = time.time()
        with torch.no_grad():
            for i, shot_num in enumerate(self.shots_test):
                if not self.check_net(self.net):
                    self.print_net(self.net)
                    raise ValueError('There is nan in the network.')

                t_reactor = time.time()
                reactor = self.reactor(shot_num, tuned=True, net=self.net)
                try:
                    y_sol, y_exp = self.solver.solve(
                        reactor, plot=True, full_plot=False, fig_type='pdf', start_up=self.start_up,
                        comment='ep{:02d}_start'.format(epoch) if self.start_up else 'ep{:02d}'.format(epoch))
                except AssertionError:
                    print('Testing, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, failed [{:.2f} s/{:.2f} s]'
                          .format(i + 1, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, time.time() - t_reactor, time.time() - t_test))
                else:
                    loss = self.criterion(y_sol, y_exp)
                    print('Testing, reactor {:d}, shot # {:d}, data step {:d}, '
                          'solver step {:d}, loss: {:.4f} [{:.2f} s/{:.2f} s]'
                          .format(i + 1, reactor.shot_num, reactor.shot['time'].size()[0],
                                  reactor.system_count, loss.item(), time.time() - t_reactor, time.time() - t_test))
                finally:
                    del reactor

        print('-' * self.num_dashes)


if __name__ == '__main__':
    # trainer = Trainer(dim=1)
    # trainer.init_net_opt()
    # trainer.train_net()

    trainer = TrainerITER(is_inductive=True, start_up=False)
    trainer.init_net_opt()
    trainer.train_net()
