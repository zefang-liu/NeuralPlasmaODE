"""
Confinement Time Fitting Models
"""
import torch.nn as nn


class LinearRegression(nn.Module):
    """
    Linear Regression
    """

    def __init__(self, input_shape, output_shape):
        """
        Initialize the network

        :param input_shape: input shape
        :param output_shape: output shape
        """
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        """
        Forward

        :param x: input vector
        :return: output vector
        """
        return self.fc(x)


class NodalLinearRegression(nn.Module):
    """
    Nodal Linear Regression
    """

    def __init__(self, input_shape, output_shape):
        """
        Initialize the network

        :param input_shape: input shape
        :param output_shape: output shape
        """
        super(NodalLinearRegression, self).__init__()
        self.fc_core = nn.Linear(input_shape, output_shape)
        self.fc_edge = nn.Linear(input_shape, output_shape)
        self.fc_sol = nn.Linear(input_shape, output_shape)

    def forward(self, x, node: str):
        """
        Forward

        :param x: input vector
        :param node: node name
        :return: output vector
        """
        if node == 'core':
            return self.fc_core(x)
        elif node == 'edge':
            return self.fc_edge(x)
        else:  # node == 'sol'
            return self.fc_sol(x)


class NodalLinearRegressionITER(nn.Module):
    """
    Nodal Linear Regression for ITER
    """

    def __init__(self, input_shape, output_shape):
        """
        Initialize the network

        :param input_shape: input shape
        :param output_shape: output shape
        """
        super(NodalLinearRegressionITER, self).__init__()
        self.fc_core = nn.Linear(input_shape, output_shape)
        self.fc_edge = nn.Linear(input_shape, output_shape)

    def forward(self, x, node: str):
        """
        Forward

        :param x: input vector
        :param node: node name
        :return: output vector
        """
        if node == 'core':
            return self.fc_core(x)
        else:
            return self.fc_edge(x)


class MultilayerPerceptron(nn.Module):
    """
    Multilayer Perceptron Regression
    """

    def __init__(self, input_shape, hidden_dim, output_shape):
        """
        Initialize the network

        :param input_shape: input shape
        :param output_shape: output shape
        """
        super(MultilayerPerceptron, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_shape)
        )

    def forward(self, x):
        """
        Forward

        :param x: input vector
        :return: output vector
        """
        return self.net(x)
