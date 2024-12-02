
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch import jit
import math

import lightning as L
import pdb


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=1,
                 target_size=5,
                 input_size=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=17,
                 device=None,
                 classes=[],
                 model_type='alpha'):
        super(NBeatsNet, self).__init__()
        self.classes = classes
        self.leads = []
        self.target_size = target_size
        self.input_size = input_size
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.device = device
        self.parameters = []

        if model_type == 'alpha':
            linear_input_size = 363 * input_size
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            linear_input_size = input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier
        self.fc_linear = nn.Linear(363 * len(classes), len(classes))

        print(f'| N-Beats, device={self.device}')

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id], self.input_size,
                                   self.target_size, classes=len(self.classes))
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=backcast.shape, device=self.device)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                print(forecast.get_device())
                print(f.get_device)
                forecast = forecast + f

        return backcast, forecast


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, classes=16):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        self.classes = classes

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim)
            self.theta_f_fc = nn.Linear(units, thetas_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, classes=16):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, classes=classes)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, backcast_length)  # forecast_length)

    def forward(self, x):
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))  # tutaj masz thetas_dim rozmiar

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class Nbeats_alpha(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 classes=[],
                 model_type='alpha'):
        super(Nbeats_alpha, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.device=device
        self.classes = classes
        self.relu = nn.ReLU()

        self.nbeats_alpha1 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=self.num_layers,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       device=device,
                                       classes=self.classes,
                                       hidden_layer_units=self.hidden_size)

        self.nbeats_alpha2 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=self.num_layers,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       device=device,
                                       classes=self.classes,
                                       hidden_layer_units=hidden_size)

        self.fc_1 = nn.Linear(self.input_size * 555, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

    def forward(self, rr_x, rr_wavelets):
        _, output_alpha1 = self.nbeats_alpha1(rr_x)  # lstm with input, hidden, and internal state
        _, output_alpha2 = self.nbeats_alpha2(rr_wavelets)  # lstm with input, hidden, and internal state

        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class Nbeats_beta(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 classes=[],
                 model_type='beta'):
        super(Nbeats_beta, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.relu = nn.ReLU()

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        # self.hidden_size = 1
        # self.num_layers = 3
        self.input_size = 1

        self.nbeats_beta = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                     nb_blocks_per_stack=self.num_layers,
                                     target_size=num_classes,
                                     input_size=self.input_size,
                                     thetas_dims=(32, 32),
                                     classes=self.classes,
                                     hidden_layer_units=self.hidden_size)

        self.fc = nn.Linear(input_size * self.linea_multiplier + 370 * self.linea_multiplier + self.linea_multiplier,
                            num_classes)  # hidden_size, 128)  # fully connected 1# fully connected last layer

    def forward(self, pca_features):
        _, output_beta = self.nbeats_beta(pca_features)  # lstm with input, hidden, and internal state

        tmp = torch.squeeze(output_beta)
        out = self.relu(tmp)  # relu
        out = self.fc(out)  # Final Output
        return out


class LSTM_ECG(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(LSTM_ECG, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        print(f'| LSTM_ECG')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstm_alpha1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bidirectional=False)
        if model_type == 'alpha':
            self.lstm_alpha2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                       num_layers=num_layers, batch_first=True, bidirectional=False)

            self.fc_1 = nn.Linear(hidden_size * 555, 128)  # hidden_size, 128)  # fully connected 1
            self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            # self.hidden_size=1
            # self.num_layers=1
            self.input_size = 1
            self.lstm_alpha1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(
                (input_size * self.linea_multiplier + 370 * self.linea_multiplier + self.linea_multiplier) * self.hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, rr_x=None, rr_wavelets=None):
        if self.model_type == 'alpha':
            h_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                            device=self.device))  # internal state
            h_1 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_1 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                            device=self.device))  # internal state

            output_alpha1, (hn_alpha1, cn) = self.lstm_alpha1(rr_x,
                                                              (h_0, c_0))  # lstm with input, hidden, and internal state
            output_alpha2, (hn_alpha2, cn) = self.lstm_alpha2(rr_wavelets,
                                                              (h_1, c_1))  # lstm with input, hidden, and internal state
            tmp = torch.hstack((output_alpha1, output_alpha2))
            tmp = torch.flatten(tmp, start_dim=1)

            out = self.fc_1(tmp)  # first Dense
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
            return out
        else:
            rr_wavelets=rr_x #as we pass only one argument which is pca vector in fact
            h_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_wavelets.size(0), self.hidden_size,
                            device=self.device))  # hidden state
            c_0 = autograd.Variable(
                torch.zeros(self.num_layers * self.when_bidirectional, rr_wavelets.size(0), self.hidden_size,
                            device=self.device))  # internal state

            output_beta, (hn_beta, cn) = self.lstm_alpha1(rr_wavelets, (h_0, c_0))

            out = torch.flatten(output_beta, start_dim=1)
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
        return out


class GRU_ECG_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(GRU_ECG_ALPHA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        print(f'| GRU_ECG_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.gru_alpha1 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True, bidirectional=False)
        self.gru_alpha2 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True, bidirectional=False)

        self.fc_1 = nn.Linear(hidden_size * 555, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, rr_x, rr_wavelets):
        h_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        h_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        output_alpha1, hn_alpha1 = self.gru_alpha1(rr_x, h_0)  # lstm with input, hidden, and internal state
        output_alpha2, hn_alpha2 = self.gru_alpha2(rr_wavelets, h_1)  # lstm with input, hidden, and internal state
        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class GRU_ECG_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(GRU_ECG_BETA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2
        print(f'| GRU_ECG_BETA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        # self.hidden_size=1
        # self.num_layers=1
        self.input_size = 1
        self.gru_beta = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True, bidirectional=False)
        # self.fc_1 = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 370 * self.linea_multiplier + self.linea_multiplier) * hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, pca_features):
        h_0 = autograd.Variable(
            torch.zeros(self.num_layers * self.when_bidirectional, pca_features.size(0), self.hidden_size,
                        device=self.device))  # hidden state
        output_beta, hn_beta = self.gru_beta(pca_features, h_0)

        # out = torch.squeeze(output_beta)
        out = torch.flatten(output_beta, start_dim=1)
        out = self.relu(out)  # relua
        # out = self.fc_1(out)
        out = self.fc(out)  # Final Output
        return out


class JitLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(JitLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.weight_ch_i = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_f = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_o = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_size, sequence_size, input_size = input.size()
        hidden_seq = []
        hx, cx = state
        for t in range(sequence_size):
            inp = input[:, t, :]
            xh = (torch.mm(inp, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

            i, f, _c, o = xh.chunk(4, 1)

            i = torch.sigmoid(i + (self.weight_ch_i * cx))
            f = torch.sigmoid(f + (self.weight_ch_f * cx))
            _c = torch.tanh(_c)

            cy = (f * cx) + (i * _c)

            o = torch.sigmoid(o + (self.weight_ch_o * cy))
            hy = o * torch.tanh(cy)
            hidden_seq.append(hy.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (hx, cx)


class LSTMPeephole_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(LSTMPeephole_ALPHA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device=device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_alpha1 = JitLSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.lstmpeephole_alpha2 = JitLSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.fc_1 = nn.Linear(hidden_size * 541, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, rr_x, rr_wavelets):
        h_0 = autograd.Variable(torch.zeros(rr_x.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_0 = autograd.Variable(torch.zeros(rr_x.size(0), self.hidden_size,
                                            device=self.device))  # internal state
        h_1 = autograd.Variable(torch.zeros(rr_wavelets.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_1 = autograd.Variable(torch.zeros(rr_wavelets.size(0), self.hidden_size,
                                            device=self.device))  # internal state


        oa1, _ = self.lstmpeephole_alpha1(rr_x,
                                                   (h_0, c_0))  # lstm with input, hidden, and internal state
        oa2, _ = self.lstmpeephole_alpha2(rr_wavelets,
                                                   (h_1, c_1))  # lstm with input, hidden, and internal state

        tmp = torch.hstack((oa1, oa2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        del tmp, h_0, c_0, h_1, c_1
        out = self.relu(out)  # relu
        out = self.fc(out) # Final Output
        return out


class LSTMPeephole_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='beta',
                 classes=[]):
        super(LSTMPeephole_BETA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.sigmoid = nn.Sigmoid()
        self.device=device
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_BETA')

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        # self.hidden_size=1
        # self.num_layers=1
        self.input_size = 1

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_beta = JitLSTMCell(input_size=self.input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier) * hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, pca_features):
        h_0 = autograd.Variable(torch.zeros(pca_features.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        c_0 = autograd.Variable(torch.zeros(pca_features.size(0), self.hidden_size,
                                            device=self.device))  # hidden state


        oa1, _  = self.lstmpeephole_beta(pca_features,
                                                 (h_0, c_0))  # lstm with input, hidden, and internal state

        out = torch.flatten(oa1, start_dim=1)
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class CustomLSTMPeephole(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=True):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication

            if self.peephole:
                gates = (torch.mm(x_t, self.U) + torch.mm(c_t, self.W) + self.bias)
            else:
                gates = x_t @ self.U + h_t @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ self.U + self.bias)[:, HS * 2:HS * 3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)



class CustomLSTMPeephole_ALPHA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='alpha',
                 classes=[]):
        super(CustomLSTMPeephole_ALPHA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_ALPHA')

        # self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstmpeephole_alpha1 = CustomLSTMPeephole(input_size=input_size, hidden_size=hidden_size)
        self.lstmpeephole_alpha2 = CustomLSTMPeephole(input_size=input_size, hidden_size=hidden_size)

        self.fc_1 = nn.Linear(hidden_size * 541, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, rr_x, rr_wavelets):
        h_0 = autograd.Variable(torch.zeros(rr_x.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_0 = autograd.Variable(torch.zeros(rr_x.size(0), self.hidden_size,
                                            device=self.device))  # internal state
        h_1 = autograd.Variable(torch.zeros(rr_wavelets.size(0), self.hidden_size,
                                            device=self.device))  # hidden state
        c_1 = autograd.Variable(torch.zeros(rr_wavelets.size(0), self.hidden_size,
                                            device=self.device))  # internal state

        output = []
        oa1, (h_0, c_0) = self.lstmpeephole_alpha1(rr_x, (h_0, c_0))  # lstm with input, hidden, and internal state
        oa2, (h_1, c_1) = self.lstmpeephole_alpha2(rr_wavelets, (h_1, c_1))  # lstm with input, hidden, and internal state

        tmp = torch.hstack((oa1, oa2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out



class CustomLSTMPeephole_BETA(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 device,
                 model_type='beta',
                 classes=[]):
        super(CustomLSTMPeephole_BETA, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1  # if bidirectional = True, then it has to be equal to 2

        print(f'| LSTM_PEEPHOLE_BETA')

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        #self.hidden_size=1
        #self.num_layers=1
        self.input_size = 1

        self.lstmpeephole_beta = CustomLSTMPeephole(input_size=self.input_size, hidden_size=self.hidden_size)

        self.fc = nn.Linear(
            (input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier) * self.hidden_size,
            num_classes)
        self.relu = nn.ReLU()

    def forward(self, pca_features):
        h_0 = autograd.Variable(torch.zeros(pca_features.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        c_0 = autograd.Variable(torch.zeros(pca_features.size(0), self.hidden_size,
                                            device=self.device))  # hidden state

        oa1, (h_0, c_0) = self.lstmpeephole_beta(pca_features, (h_0, c_0))  # lstm with input, hidden, and internal state

        out = torch.flatten(oa1, start_dim=1)
        out = self.relu(out)  # relu
        out = self.fc(out) # Final Output
        return out


class BlendMLP(nn.Module):
    def __init__(self, modelA, modelB, classes):
        super(BlendMLP, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classes = classes
        self.linear = nn.Linear(2 * len(classes), len(classes))

    def forward(self, rr_x, rr_wavelets, pca_features):
        x1 = self.modelA(rr_x, rr_wavelets)
        #x2 = self.modelB(rr_x, pca_features) # FOR LSTM
        x2 = self.modelB(pca_features)  # FOR NBEATS, PEEPHOLE and GRU

        if x1.shape == x2.shape:
            out = torch.cat((x1, x2), dim=1)
            out = self.linear(F.relu(out))
            return out
        else:
            return x1
        return x2

def get_single_network(network, hs, layers, leads, selected_classes, single_peak_length, as_branch, device):
    torch.manual_seed(17)
    if network == "GRU":
        if as_branch == "alpha":
            return GRU_ECG_ALPHA(input_size=len(leads),
                    num_classes=len(selected_classes),
                    hidden_size=hs,
                    num_layers=layers,
                    seq_length=single_peak_length,
                    device=device,
                    model_type=as_branch,
                    classes=selected_classes)
        else:
            return GRU_ECG_BETA(input_size=len(leads),
                        num_classes=len(selected_classes),
                        hidden_size=hs,
                        num_layers=layers,
                        seq_length=single_peak_length,
                        device=device,
                        model_type=as_branch,
                        classes=selected_classes)
    
    if network == "LSTM":
        return LSTM_ECG(input_size=len(leads),
            num_classes=len(selected_classes),
            hidden_size=hs,
            num_layers=layers,
            seq_length=single_peak_length,
            device=device,
            model_type=as_branch,
            classes=selected_classes)
    
    if network == "LSTM_PEEPHOLE":
        if as_branch == "alpha":
            return LSTMPeephole_ALPHA(input_size=len(leads),
                    num_classes=len(selected_classes),
                    hidden_size=hs,
                    num_layers=layers,
                    seq_length=single_peak_length,
                    device=device,
                    model_type=as_branch,
                    classes=selected_classes)
        else:
            return LSTMPeephole_BETA(input_size=len(leads),
                        num_classes=len(selected_classes),
                        hidden_size=hs,
                        num_layers=layers,
                        seq_length=single_peak_length,
                        device=device,
                        model_type=as_branch,
                        classes=selected_classes)

    if network == "NBEATS":
        if as_branch == "alpha":
            return Nbeats_alpha(input_size=len(leads),
                        num_classes=len(selected_classes),
                        hidden_size=hs,
                        num_layers=layers,
                        seq_length=single_peak_length,
                        device=device,
                        model_type=as_branch,
                        classes=selected_classes)
        else:
            return Nbeats_beta(input_size=len(leads),
                            num_classes=len(selected_classes),
                            hidden_size=hs,
                            seq_length=single_peak_length,
                            device=device,
                            model_type=as_branch,
                            classes=selected_classes,
                            num_layers=layers)


class BranchConfig:
    network_name = ""
    single_peak_length = -1
    hidden_size = -1 
    layers = -1
    
    def __init__(self,network_name, hidden_size, layers, single_peak_length) -> None:
        self.network_name = network_name
        self.single_peak_length=single_peak_length
        self.hidden_size=hidden_size
        self.layers=layers



def get_BlendMLP(alpha_config: BranchConfig, beta_config: BranchConfig, classes: list, device, leads: list) -> BlendMLP:
    alpha_branch = get_single_network(alpha_config.network_name, alpha_config.hidden_size, alpha_config.layers, leads, classes, alpha_config.single_peak_length, "alpha", device)
    beta_branch = get_single_network(beta_config.network_name, beta_config.hidden_size, beta_config.layers, leads, classes, beta_config.single_peak_length, "beta", device)

    return BlendMLP(alpha_branch, beta_branch, classes)

