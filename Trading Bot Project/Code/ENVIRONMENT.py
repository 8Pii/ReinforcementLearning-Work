#ENVIRONMENT
from DataHandling import * 
from UtilFunctions import *
from UtilStructures import *
from AGENT import *
from Models import *

class Environment:
    def __init__(self, data, aggressive = False):
        self.data = data
        self.reward_f = "profit"
        self.aggressive = aggressive
        self.reset()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        self.t = 13
        self.done = False
        self.aggressive = False
        self.agent_pos = Positions.FLAT
        Positions.COUNTER = 0
        self.init_price = self.data.iloc[0, :]['close']


        Ledgers.ACTIVE_LONG = {
            "Entry Price" : [],
            "Current Price" : [],
            "Dollar Profit" : [],
            "% Return" : []
        }

        Ledgers.ACTIVE_SHORT = {
            "Entry Price" : [],
            "Current Price" : [],
            "Dollar Profit" : [],
            "% Return" : []
        }

        Ledgers.HIST = {
            "Entry Price" : [],
            "Action Type" : [],
            "Position Type" : [],
            "Dollar Profit, Realized"   : 0,
            "Dollar Profit, Unrealized" : 0,
            "Time Index" : []
        }

        self.agent_init_pos_real = Ledgers.HIST['Dollar Profit, Realized']
        self.agent_init_pos_unreal = Ledgers.HIST['Dollar Profit, Realized']
        self.agent_pos_total = 0 # realized + unrealized

        self.profits            = [0 for e in range(len(self.data))]
        self.cumulative_return  = [1 for e in range(len(self.data))]

    def get_state(self):
        if not self.done:
            return torch.tensor([price for price in self.data.iloc[self.t - 13:self.t + 1, :]['close']], device=self.device,
                                dtype=torch.float)
        else:
            return None

    def step(self, act):

        reward = 0

        # GET CURRENT STATE
        state = self.data.iloc[self.t, :]['close']
        data_idx = self.data.index
        index = data_idx[self.t]

        # NEW ACTIONS :
        # print()
        # print()
        # print(f'Agent Position, before       : {self.agent_pos}')
        # print(f'Position Counter, before     : {Positions.COUNTER}')
        # print(f'Action Chosen                : {act}')
        self.agent_pos, _realized_profits, _unrealized_profits, _ = transform(self.agent_pos, act, state, index)
        # print(f'Agent Position               : {self.agent_pos}')
        # print(f'Position Counter             : {Positions.COUNTER}')
        # print(f'Agent Realized Profits       : {_realized_profits}')
        # print(f'Agent Unrealized Profits     : {_unrealized_profits}')
        # print(f'Ledger Hist $ Prof Realized  : {Ledgers.HIST["Dollar Profit, Realized"]}')
        # print(f'Ledger Hist $ Prof Unrealized: {Ledgers.HIST["Dollar Profit, Unrealized"]}')
        # print(f'Ledger Hist Entry Price      : {Ledgers.HIST["Entry Price"][::-1]}')
        # print(f'Ledger Hist Action Type      : {Ledgers.HIST["Action Type"][::-1]}')
        # print(f'Ledger Hist Position Type    : {Ledgers.HIST["Position Type"][::-1]}')
        # print(f'Ledger Active Long CP        : {Ledgers.ACTIVE_LONG["Current Price"][::-1]}')
        # print(f'Ledger Active Long EP        : {Ledgers.ACTIVE_LONG["Entry Price"][::-1]}')
        # print(f'Ledger Active Short CP       : {Ledgers.ACTIVE_SHORT["Current Price"][::-1]}')
        # print(f'Ledger Active Short EP       : {Ledgers.ACTIVE_SHORT["Entry Price"][::-1]}')

        self.profits[self.t] = _realized_profits + _unrealized_profits

        self.agent_pos_total += Ledgers.HIST['Dollar Profit, Realized'] + Ledgers.HIST['Dollar Profit, Realized']


        self.cumulative_return[self.t] += (reduce(lambda x, y : x * y , Ledgers.ACTIVE_LONG['% Return'], 1)
                                           * reduce(lambda x , y : x * y, Ledgers.ACTIVE_SHORT['% Return'], 1))

        # COLLECT THE REWARD
        reward = 0
        risk_free_rate = 0.03
        annual_factor = 252

        if self.reward_f == "profit":
            curr_profits = self.profits[self.t]
            if curr_profits > 0:
                reward = 10
            elif curr_profits < 0:
                reward = -10
            elif curr_profits == 0:
                if self.aggressive:
                    reward = -2
                else:
                    reward = 0

        if self.agent_pos == Positions.FLAT and (reward > -5): #penalize not trying to improve
            reward = -5

        # UPDATE THE STATE
        self.t += 1

        if (self.t == len(self.data) - 1):
            self.done = True

        return torch.tensor([reward], device=self.device, dtype=torch.float), self.done, torch.tensor([state],
                                                                                                 dtype=torch.float)  # reward, done, current_state