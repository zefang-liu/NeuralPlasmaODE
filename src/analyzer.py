"""
Data Analyzer
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import sin, cos, pi

from src import config
from src.preprocessor import Preprocessor
from src.reaction import Reaction

plt.style.use('science')


class Analyzer(object):
    """
    Data Analyzer
    """

    def __init__(self, reactor_type):
        """
        Initialize the analyzer
        """
        self.reactor_type = reactor_type

        if self.reactor_type == 'd3d':
            self.R0 = 1.75
            self.a = 0.55
            self.kappa = 1.5
            self.delta = 0.5
        else:  # iter
            self.R0 = 6.2
            self.a = 2.0
            self.kappa = 1.7
            self.delta = 0.33

        self.b = self.a * self.kappa

    def analyze(self):
        """
        Analyze all shots
        """
        shot_nums = config.shots_train + config.shots_test
        shot_nums.sort()
        preprocessor = Preprocessor()

        for shot_num in shot_nums:
            shot = preprocessor.preprocess(shot_num=shot_num)
            if shot_num in config.shots_test:
                output = '%d* \t& ' % shot_num
            else:
                output = '%d \t& ' % shot_num

            # Powers in MW
            for signal_name in ['poh', 'echpwrc', 'ichpwrc', 'pnbi']:
                output += self.add_signal(shot, signal_name, norm=1E6)

            # GAS in Torr-L/s
            output += self.add_signal(shot, 'gas', norm=1)

            # Other signals
            output += self.add_signal(shot, 'bt0', norm=1)
            output += self.add_signal(shot, 'ne', norm=1E19)
            output += self.add_signal(shot, 'te', norm=1E3)

            # Line break
            output = output[:-2] + '\\\\'

            print(output)

    def add_signal(self, shot: dict, signal_name: str, norm: float) -> str:
        """
        Add one signal to the output

        :param shot: shot dictionary
        :param signal_name: signal name
        :param norm: normalization to the signal
        :return: output
        """
        output = ''
        if signal_name in shot:
            signal = shot[signal_name].numpy()
            signal_min = np.min(signal) / norm
            signal_max = np.max(signal) / norm
            if np.abs(signal_min) < 0.01:
                signal_min = 0
            if np.abs(signal_max) < 0.01:
                signal_max = 0
            if signal_min == signal_max:
                output += '%.2f \t& ' % signal_min
            else:
                output += '%.2f-%.2f \t& ' % (signal_min, signal_max)
        else:
            output += '0.00 \t& '
        return output

    def plot_shots(self, fig_type='pdf', show=False):
        """
        Plot shots

        :param fig_type: figure type
        :param show: show the plot or not
        """
        shot_nums = config.shots_train + config.shots_test
        shot_nums.sort()
        preprocessor = Preprocessor()

        for shot_num in shot_nums:
            shot = preprocessor.preprocess(shot_num=shot_num)

            num_row = 4
            num_col = 2
            fig = plt.figure(figsize=(7.4, 1.2 * num_row))

            ax11 = fig.add_subplot(num_row, num_col, 1)
            ax11.plot(shot['time'], shot['ip'] / 1E6, 'b-')
            ax11.xaxis.set_ticklabels([])
            ax11.set_ylabel(r'$I_P$ [MA]')
            ax11.grid('on')

            ax21 = fig.add_subplot(num_row, num_col, 3, sharex=ax11)
            ax21.plot(shot['time'], shot['bt0'], 'b-')
            ax21.set_ylabel(r'$|B_0|$ [T]')
            ax21.grid('on')

            ax31 = fig.add_subplot(num_row, num_col, 5, sharex=ax11)
            ax31.plot(shot['time'], shot['q95'], 'b-')
            ax31.set_ylabel(r'$q_{95}$')
            ax31.grid('on')

            ax41 = fig.add_subplot(num_row, num_col, 7)
            ax41.plot(shot['time'], shot['gas'], 'b-')
            ax41.set_xlabel(r'$t$ [s]')
            ax41.set_ylabel('GAS\n[Torr-L/s]')
            ax41.grid('on')

            ax12 = fig.add_subplot(num_row, num_col, 2)
            ax12.plot(shot['time'], shot['poh'] / 1E6, 'b-')
            ax12.xaxis.set_ticklabels([])
            ax12.set_ylabel(r'$P_{\Omega}$ [MW]')
            ax12.grid('on')

            ax22 = fig.add_subplot(num_row, num_col, 4, sharex=ax12)
            ax22.plot(shot['time'], shot['pnbi'] / 1E6, 'b-')
            ax22.set_ylabel(r'$P_{NBI}$ [MW]')
            ax22.grid('on')

            ax32 = fig.add_subplot(num_row, num_col, 6, sharex=ax12)
            ax32.plot(shot['time'], shot['echpwrc'] / 1E6, 'b-')
            ax32.set_ylabel(r'$P_{ECH}$ [MW]')
            ax32.grid('on')

            ax42 = fig.add_subplot(num_row, num_col, 8)
            ax42.plot(shot['time'], np.round(shot['ichpwrc'] / 1E6, 6), 'b-')
            ax42.set_xlabel(r'$t$ [s]')
            ax42.set_ylabel(r'$P_{ICH}$ [MW]')
            ax42.grid('on')

            ax11.set_title('Shot ' + str(shot_num))
            ax12.set_title('Shot ' + str(shot_num))
            fig.subplots_adjust(hspace=0, wspace=0.25)

            fig_name = './figure/shot/shot_{:d}_t{:d}_w{:d}.{:s}' \
                .format(shot_num, shot['time'].size()[0], config.window, fig_type)
            fig.savefig(fig_name)
            print('Figure saved: {:s}'.format(fig_name))

            if show:
                plt.show()
            plt.close()

    def plot_regions(self):
        """
        Plot the regions of the multi-nodal model
        """
        theta = np.arange(0, 2 * np.pi, np.pi / 180)

        if self.reactor_type == 'd3d':
            regions = config.nodes
            rhos = [config.rho_core, config.rho_edge, config.rho_sol]
            colors = ['red', 'coral', 'gold']
        else:
            regions = ['core', 'edge']
            rhos = [config.rho_core, config.rho_edge]
            colors = ['red', 'gold']

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        for region, rho, color in zip(regions[::-1], rhos[::-1], colors[::-1]):
            R = self.R0 + rho * self.a * np.cos(theta + self.delta * np.sin(theta))
            Z = self.kappa * rho * self.a * np.sin(theta)
            ax.fill(R, Z, c=color, label=region.capitalize())

        # ax.legend(loc='best')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels),
                  bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$R$ [m]')
        ax.set_ylabel(r'$Z$ [m]')

        plt.tight_layout()
        fig_name = './figure/others/regions_' + self.reactor_type + '.pdf'
        fig.savefig(fig_name)
        print('Figure saved: ' + fig_name)
        plt.show()

    def plot_surfaces(self):
        """
        Plot the inner and outer surfaces for the multi-nodal model
        """
        theta, phi = np.meshgrid(np.arange(0, 2 * pi, pi / 360), np.arange(-0.5 * pi, pi, pi / 360))
        X = (self.R0 - self.b + (self.a + self.b * cos(theta)) * cos(theta + self.delta * sin(theta))) * cos(phi)
        Y = (self.R0 - self.b + (self.a + self.b * cos(theta)) * cos(theta + self.delta * sin(theta))) * sin(phi)
        Z = (self.kappa * self.a) * sin(theta)

        a_in = self.a * 0.6
        theta_in, phi_in = np.meshgrid(np.arange(0, 2 * pi, pi / 360), np.arange(-0.5 * pi, pi, pi / 360))
        X_in = (self.R0 - self.b + (a_in + self.b * cos(theta_in))
                * cos(theta_in + self.delta * sin(theta_in))) * cos(phi_in)
        Y_in = (self.R0 - self.b + (a_in + self.b * cos(theta_in))
                * cos(theta_in + self.delta * sin(theta_in))) * sin(phi_in)
        Z_in = (self.kappa * a_in) * sin(theta_in)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.4)
        ax.plot_surface(X_in, Y_in, Z_in, cmap='Reds', alpha=1.0)

        lim = 1.2 * self.R0
        ax.set_xlabel(r'$X$ [m]')
        ax.set_ylabel(r'$Y$ [m]')
        ax.set_zlabel(r'$Z$ [m]')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(30, -135)

        plt.tight_layout()
        plt.savefig('./figure/others/surfaces_' + self.reactor_type + '.pdf')
        plt.show()

    def plot_reactivities(self):
        """
        Plot fusion reactivities
        """
        Ti = torch.arange(0.5, 200, 0.1) * 1E3
        reaction = Reaction()
        reactivities = {}
        line_styles = ['-', '--', ':', '-.']

        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)

        for reaction_type, line_style in zip(config.reaction_types, line_styles):
            if reaction_type == 'ddnh':
                continue
            reactivities[reaction_type] = reaction.get_fusion_coefficient(Ti, reaction_type)
            if reaction_type == 'tdna':
                label = 'D-T'
            elif reaction_type == 'hdpa':
                label = r'D-$^3$He'
            else:
                label = 'D-D'
            ax.loglog(Ti / 1E3, reactivities[reaction_type], c='k',
                      ls=line_style, label=label)

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True)
        ax.set_xlabel(r'$T_i$ [keV]')
        ax.set_ylabel(r'$\langle \sigma v \rangle$ [m$^3$/s]')

        fig.tight_layout()
        fig.savefig('./figure/others/reactivities.pdf')
        fig.show()


if __name__ == '__main__':
    analyzer = Analyzer(reactor_type='iter')
    analyzer.plot_regions()
