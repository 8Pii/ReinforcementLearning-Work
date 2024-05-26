#GLOBALS :

market_impact = 0.015

class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4
    CLOSE_ALL = 5
    COMBO_BUY = 6


class Positions:
    SHORT = -1
    FLAT = 0
    LONG = 1
    COUNTER = 0


class Ledgers:
    ACTIVE_LONG = {
        "Entry Price" : [],
        "Current Price" : [],
        "Dollar Profit" : [],
        "% Return" : []
    }
    ACTIVE_SHORT = {
        "Entry Price" : [],
        "Current Price" : [],
        "Dollar Profit" : [],
        "% Return" : []
    }
    HIST = {
        "Entry Price" : [],
        "Action Type" : [],
        "Position Type" : [],
        "Dollar Profit, Realized"   : 0,
        "Dollar Profit, Unrealized" : 0,
        "Time Index" : []
    }

#UTIL FUNCTIONS

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward') )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


