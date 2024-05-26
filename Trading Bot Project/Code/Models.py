# Baseline Models
from DataHandling import * 
from UtilFunctions import *
from UtilStructures import *
from AGENT import *
from ENVIRONMENT import *

class RandomWalk():
    def __init__(self):
        self.action_idx_lst = []

    def random_selection(self):
        if Positions.COUNTER == 0:
            self.action_idx_lst = [0, 1, 3, 4, 6]
            return random.choice(self.action_idx_lst)
        elif abs(Positions.COUNTER) > 8:
            self.action_idx_lst = 5
            return self.action_idx_lst
        else:
            return random.choice(self.action_idx_lst)


class MomentumFollowing():
    def __init__(self):
        self.action_idx_lst = []

    def trend_selection(self, data, cut_out):
        if abs(Positions.COUNTER) > 10:
            return 5 #close all
        # if data[0][0][0] > data[0][0][cut_out]:
        curr_price = data[0][0][-1]
        threshold = np.mean(float(data[0][0][-cut_out]) + float(data[0][0][-(cut_out+1)]) +
                            float(data[0][0][-(cut_out+2)]) + float(data[0][0][-(cut_out+3)]))
        if curr_price > threshold:   #if the latest data is bigger than 5 timestamps old data
            if Positions.COUNTER < 0 :
                return 5
            else:
                self.action_idx_lst = [3,4,6]
                return random.choice(self.action_idx_lst)
        elif curr_price < threshold:
            if Positions.COUNTER > 0:
                return 5
            else:
                self.action_idx_lst = [0, 1]
                return random.choice(self.action_idx_lst)
        else:
            return 2

# Convolutional DQN
class ConvDQN(nn.Module):
    def __init__(self, seq_len_in, actions_n, kernel_size=8):
        super(ConvDQN, self).__init__()
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.maxPool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.LRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)

        self.hidden_dim = n_filters * (
                            (
                                (
                                    (seq_len_in - kernel_size + 1) -
                                    max_pool_kernel + 1) -
                                    kernel_size // 2 + 1) -
                                    max_pool_kernel + 1)

        self.out_layer = nn.Linear(self.hidden_dim, actions_n)

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.maxPool(self.LRelu(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.maxPool(self.LRelu(c2_out))
        # print("c1_out:\t%s"%str(c1_out.shape))
        # print("max_pool_1:\t%s"%str(max_pool_1.shape))
        # print("c2_out:\t%s"%str(c2_out.shape))
        # print("max_pool_2:\t%s"%str(max_pool_2.shape))
        # print(self.hidden_dim)
        max_pool_2 = max_pool_2.view(-1, self.hidden_dim)
        # print("max_pool_2_view:\t%s"%str(max_pool_2.shape))

        return self.LRelu(self.out_layer(max_pool_2))

class ConvDuelingDQN(nn.Module):
    def __init__(self, seq_len_in, actions_n, kernel_size=8):
        super(ConvDuelingDQN, self).__init__()
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.maxPool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.LRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)

        self.advantage_hidden_dim = n_filters * (
                            (
                                (
                                    (seq_len_in - kernel_size + 1) -
                                    max_pool_kernel + 1) -
                                    kernel_size // 2 + 1) -
                                    max_pool_kernel + 1)

        self.value_hidden_dim = n_filters * (
                            (
                                (
                                    (seq_len_in - kernel_size + 1) -
                                    max_pool_kernel + 1) -
                                    kernel_size // 2 + 1) -
                                    max_pool_kernel + 1)

        self.advantage_layer = nn.Linear(self.advantage_hidden_dim, actions_n)
        self.value_layer = nn.Linear(self.value_hidden_dim, 1)

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.maxPool(self.LRelu(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.maxPool(self.LRelu(c2_out))
        max_pool_2 = max_pool_2.view(-1, self.advantage_hidden_dim)

        advantage = self.advantage_layer(max_pool_2)
        value = self.value_layer(max_pool_2).expand_as(advantage)
        q_values = value + (advantage - advantage.mean(1, keepdim=True))
        #print(q_values)

        return q_values
