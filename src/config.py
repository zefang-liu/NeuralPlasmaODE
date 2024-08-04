"""
Configurations
"""
import matplotlib.pyplot as plt
import torch

plt.style.use('science')

# Nodes:
nodes = ['core', 'edge', 'sol']
nodes_iter = ['core', 'edge']
num_nodes = len(nodes)
num_nodes_iter = len(nodes_iter)

# Geometries:
rho_core = 0.9  # normalized radius for the core region
rho_edge = 1.0  # normalized radius for the edge region
rho_sol = 1.1  # normalized radius for the sol region
rhos = {'core': rho_core, 'edge': rho_edge, 'sol': rho_sol}

# Tokamak parameters:
impurity_charge = 6  # carbon for D3D
C_gas = 150  # 1 / gas puffing efficiency

# Reactor parameters for D3D:
num_inputs = {0: 8, 1: 9}
num_outputs = {0: 3, 1: 3}
num_vars = {0: 3, 1: 9}

# Reactor parameters for ITER:
num_outputs_iter = {0: 5, 1: 5}
num_vars_iter = {0: 7, 1: 14}

# Solver parameters:
rtol = 1e-5  # relative tolerance
atol = 1e-5  # absolute tolerance

# Preprocessor parameters:
window = 5  # simple moving average
time_step = 0.2  # time step size [s]
max_time_length = 15  # max time length[s]

# Trainer parameters:
epoch_num = 14  # number of epochs for training
lr = 0.02  # learning rate
weight_decay = 1e-5  # regularization
eval_step = 1  # evaluate per step
eval_start = True  # evaluate at beginning
eval_end = False  # evaluate at end
save_step = 1  # save per step
save_reactor = True  # save per reactor
load_network = True  # load the saved network
load_pretrained = True  # load the pretrained network
train_core_edge = False  # train the core and edge nodes only for D3D
train_sol = False  # train the SOl node only for D3D
assert not (train_core_edge and train_sol)  # only one mode is allowed

# Datasets for D3D:
shots_train = [131191, 131195, 131196, 134350, 135837,
               135843, 140417, 140419, 140421, 140422,
               140423, 140424, 140425, 140428, 140429,
               140430, 140431, 140432, 140440, 140673]
shots_test = [131190, 140418, 140420, 140427, 140535]

# Datasets for ITER
inductive_scenarios_train = [2] * 10
inductive_scenarios_test = [1, 2, 3]
non_inductive_scenarios_train = [4] * 10
non_inductive_scenarios_test = [4, 6, 7]

# Signals:
signals_1d = [signal.lower() for signal in [
    'AMINOR', 'R0', 'VOLUME', 'KAPPA', 'KAPPA0', 'TRITOP', 'TRIBOT', 'ZMAXIS',
    'IP', 'BT0', 'LI', 'Q0', 'Q95', 'BDOTAMPL', 'TAUE',
    'PTOT', 'POH', 'PNBI', 'PBINJ', 'ECHPWR', 'ICHPWR', 'ECHPWRC', 'ICHPWRC']]
signals_1d_gas = [signal.lower() for signal in [
    'GASA_CAL', 'GASB_CAL', 'GASC_CAL', 'GASD_CAL', 'GASE_CAL']]
signals_1d_vol = [signal.lower() for signal in [
    'POH', 'PNBI', 'ECHPWRC', 'ICHPWRC', 'GAS']]
signals_2d = [signal.lower() for signal in [
    'EDENSFIT', 'ETEMPFIT', 'ITEMPFIT', 'ZDENSFIT']]
signals_2d_var = ['ne', 'te', 'ti', 'nc']
signals_2d_var_iter = ['nd', 'na', 'ne', 'ti', 'te']
reaction_types = ['tdna', 'ddpt', 'ddnh', 'hdpa']

# Devices:
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Weights and bias from the pretrained network
pretrained_network = {'weights_core': [
    [-4.2589, 1.0426, -0.8033, -0.4398, 2.5829, -3.2690, -1.2093, 0.0965, 0.2555],
    [-3.5729, -0.1225, 3.2578, 1.1600, 2.3382, -2.7806, -1.0380, 0.1285, 0.3878],
    [-3.9093, 0.6392, 2.3133, 1.6147, 2.4419, -3.3836, -1.0507, 0.2298, 0.2593]
], 'bias_core': [0.0401, 0.0827, 0.0784], 'weights_edge': [
    [-4.2804, 3.2465, 0.6073, -0.3878, 2.4128, -3.1978, -1.2910, -0.0161, 0.2778],
    [-3.5983, 0.2251, 1.4056, 1.1128, 2.9299, -3.0591, -0.6859, 0.6195, -0.0464],
    [-3.3745, -0.9212, 1.4332, 1.4935, 2.9131, -3.0735, -0.5680, 0.7330, -0.1191]
], 'bias_edge': [0.0370, 0.1129, 0.1270], 'weights_sol': [
    [-3.6842, 1.5414, 0.2547, 0.8641, 2.7684, -3.1355, -0.8030, 0.4916, 0.0138],
    [-3.4738, 0.6007, 1.2647, 1.2567, 3.0425, -2.8686, -0.5700, 0.7319, -0.2312],
    [-3.5343, 0.7191, 1.2305, 1.2152, 2.9764, -2.9216, -0.6284, 0.6721, -0.1760]
], 'bias_sol': [0.0602, 0.1267, 0.1196]}

pretrained_network_iter_ind = {'weights_core': [
    [-4.2589, 1.0426, -0.8033, -0.4398, 2.5829, -3.2690, -1.2093, 0.0965, 0.2555],
    [-4.2589, 1.0426, -0.8033, -0.4398, 2.5829, -3.2690, -1.2093, 0.0965, 0.2555],
    [-3.5729, -0.1225, 3.2578, 1.1600, 2.3382, -2.7806, -1.0380, 0.1285, 0.3878],
    [-3.5729, -0.1225, 3.2578, 1.1600, 2.3382, -2.7806, -1.0380, 0.1285, 0.3878],
    [-3.9093, 0.6392, 2.3133, 1.6147, 2.4419, -3.3836, -1.0507, 0.2298, 0.2593]
], 'bias_core': [0.0401, 0.0401, 0.0827, 0.0827, 0.0784], 'weights_edge': [
    [-4.2804, 3.2465, 0.6073, -0.3878, 2.4128, -3.1978, -1.2910, -0.0161, 0.2778],
    [-4.2804, 3.2465, 0.6073, -0.3878, 2.4128, -3.1978, -1.2910, -0.0161, 0.2778],
    [-3.5983, 0.2251, 1.4056, 1.1128, 2.9299, -3.0591, -0.6859, 0.6195, -0.0464],
    [-3.5983, 0.2251, 1.4056, 1.1128, 2.9299, -3.0591, -0.6859, 0.6195, -0.0464],
    [-3.3745, -0.9212, 1.4332, 1.4935, 2.9131, -3.0735, -0.5680, 0.7330, -0.1191]
], 'bias_edge': [0.0370, 0.0370, 0.1129, 0.1129, 0.1270]}

pretrained_network_iter_non = {'weights_core': [
    [-3.7900, 1.4992, -0.3405, 0.0190, 3.0294, -2.7866, -0.7468, 0.5539, 0.7098],
    [-3.6387, 1.6557, -0.1978, 0.1655, 3.1925, -2.6430, -0.5923, 0.7113, 0.8685],
    [-3.7432, -0.2940, 3.1201, 1.0337, 2.1670, -2.9503, -1.2085, -0.0423, 0.2169],
    [-3.6404, -0.1923, 3.2177, 1.1256, 2.2683, -2.8463, -1.1060, 0.0598, 0.3188],
    [-4.0951, 0.2695, 1.9000, 1.2163, 1.9501, -3.3239, -1.2957, -0.1252, -0.0744]
], 'bias_core': [0.0642, 0.0746, 0.0698, 0.0773, 0.0668], 'weights_edge': [
    [-4.5513, 2.9719, 0.3207, -0.6698, 2.1252, -3.4549, -1.5655, -0.2962, -0.0028],
    [-4.3447, 3.1717, 0.5410, -0.4545, 2.3441, -3.2589, -1.3564, -0.0828, 0.2108],
    [-3.6126, 0.2103, 1.4569, 1.1250, 2.9140, -3.0722, -0.7007, 0.6043, -0.0614],
    [-3.6144, 0.1754, 1.4020, 1.0841, 2.8422, -3.0216, -0.7227, 0.5661, -0.0931],
    [-3.4876, -0.9879, 1.4080, 1.4120, 2.7839, -3.1730, -0.6858, 0.6115, -0.2383]
], 'bias_edge': [0.0283, 0.0347, 0.1113, 0.1112, 0.1135]}
