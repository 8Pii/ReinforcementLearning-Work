from DataHandling import * 
from UtilFunctions import *
from UtilStructures import *
from AGENT import *
from ENVIRONMENT import *
from Models import *

#IMPORTS
import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from functools import reduce
from typing import Any, List
from collections import namedtuple

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from prettytable import PrettyTable as PrettyTable

import pickle




analytic_dic = {
    "5MIN" : {
        'LinearDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'MomentumFollowing' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'RandomWalk' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
    },
    "15MIN" : {
        'LinearDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'MomentumFollowing' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'RandomWalk' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
    },
    "1H" : {
        'LinearDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'MomentumFollowing' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'RandomWalk' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
    },
    "6H" : {
        'LinearDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'MomentumFollowing' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'RandomWalk' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
    },
    "1D" : {
        'LinearDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDuelingDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'MomentumFollowing' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'RandomWalk' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
        'ConvDQN' : {
            "Time Index" : None,
            "Price" : None,
            "Profit" : None,
            "Action Type" : None,
            "Position Type" : None,
            "Time Action Idx" : None
        },
    },
}

def main_():
    global analytic_dic
    #----------------------------- LOAD DATA ---------------------------------------------------------------------------
    path = './'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path)


    # ----------------------------- AGENTS INPUT --------------------------------
    # ACTION_NUMBER = len(list(Actions))
    ACTION_NUMBER = len(list(Actions))
    REPLAY_MEM_SIZE = 100
    BATCH_SIZE = 10
    DISCOUNT = 0.98
    EPS_START = 1
    EPS_END = 0.12
    EPS_STEPS = 100
    LEARNING_RATE = 0.001
    INPUT_DIM = 14
    HIDDEN_DIM = 120
    TARGET_UPDATE = 10
    N_TEST = 1
    TRADING_PERIOD = 1000
    MODEL_LIST = ['LinearDuelingDQN','ConvDuelingDQN','MomentumFollowing', 'RandomWalk', 'ConvDQN']

        # "Action Type" : [],
        # "Position Type" : [],

    df_list = resampling(df)

    df_list_names = ["5MIN","15MIN","1H","6H","1D"]
    for i, df_ in enumerate(df_list):
      print()
      print()
      print(df_list_names[i])
      index = random.randrange(len(df_) - TRADING_PERIOD - 1)
      for MODEL in MODEL_LIST:
          dqn_agent = Agent(ACTION_NUMBER,
                          REPLAY_MEM_SIZE,
                          BATCH_SIZE,
                          DISCOUNT,
                          EPS_START,
                          EPS_END,
                          EPS_STEPS,
                          LEARNING_RATE,
                          INPUT_DIM,
                          HIDDEN_DIM,
                          TARGET_UPDATE,
                          MODEL)

          train_size = int(TRADING_PERIOD * 0.8)
          profit_dqn_return = []

          profit_train_env = Environment(df_[index:index + train_size], "profit")

          # Profit Double DQN
          cr_profit_dqn = dqn_agent.train(profit_train_env, path)
          profit_train_env.reset()
          print()
          j = 0
          while j < N_TEST:
              print("Test nr. %s" % str(i+1))
              # index = random.randrange(len(df) - TRADING_PERIOD - 1)

              profit_test_env = Environment(df_[index + train_size:index + TRADING_PERIOD], "profit")

              # Profit Double DQN
              cr_profit_dqn_test, _ = dqn_agent.test(profit_test_env , path=path)
              profit_dqn_return.append(profit_test_env.profits)
              # profit_dqn_return = profit_dqn_return[0][:-1]
              del profit_dqn_return[0][-1]

              slice_df = df_[index + train_size:index + TRADING_PERIOD]

              analytic_dic[df_list_names[i]][MODEL]["Time Index"] = slice_df.index
              analytic_dic[df_list_names[i]][MODEL]["Price"] = slice_df.close
              analytic_dic[df_list_names[i]][MODEL]["Profit"] = profit_dqn_return
              analytic_dic[df_list_names[i]][MODEL]["Action Type"] = list(Ledgers.HIST['Action Type'])
              analytic_dic[df_list_names[i]][MODEL]["Entry Price"] = list(Ledgers.HIST['Entry Price'])
              analytic_dic[df_list_names[i]][MODEL]["Position Type"] = list(Ledgers.HIST['Position Type'])
              analytic_dic[df_list_names[i]][MODEL]["Time Action Idx"] = list(Ledgers.HIST['Time Index'])

              profit_test_env.reset()


              #--------------------------------------- Print Test Stats ---------------------------------------------------------
              t = PrettyTable(["Trading System", "Avg. Return ($)", "Max Return ($)", "Min Return ($)", "Std. Dev."])
              print_stats(f'Profit {MODEL}', profit_dqn_return, t)

              print(t)


              # plot_pnl("Profit C-DQN", profit_dqn_return, slice)
              # print(len(temp_lst))

              j += 1




if __name__ == "__main__":
    main_()


# Save dictionary to a file using pickle
with open('analytic_dic.pkl', 'wb') as pickle_file:
    pickle.dump(analytic_dic, pickle_file)