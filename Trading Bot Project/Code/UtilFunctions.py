from DataHandling import * 
from UtilStructures import *
from AGENT import *
from ENVIRONMENT import *
from Models import *


def print_stats(model, c_return, t):
    c_return = np.array(c_return).flatten()
    t.add_row([str(model), "%.2f" % np.mean(c_return), "%.2f" % np.amax(c_return), "%.2f" % np.amin(c_return),
               "%.2f" % np.std(c_return)])

def plot_pnl(name, cum_returns, slice):
    """ NB. cum_returns must be 2-dim """
    # Mean
    M = np.mean(np.array(cum_returns), axis=0)
    # std dev
    S = np.std(np.array(cum_returns), axis=0)
    # upper and lower limit of confidence intervals
    LL = M - 0.95 * S
    UL = M + 0.95 * S

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # 1 row, 2 columns

    axs[0].plot(range(len(M)), M, linewidth=2)  # Plot the mean curve on the first subplot
    axs[0].fill_between(range(len(M)), LL, UL, color='b', alpha=.2)  # Fill between for the first subplot
    axs[0].grid(True)
    axs[0].set_xlabel("Trading Instant (h)")
    axs[0].set_ylabel("Return")
    axs[0].legend(['Cumulative Average Return (%)'], loc='upper left')

    # axs[1].plot(x=slice.index, y = slice)
    axs[1].plot(slice)

    plt.tight_layout()
    plt.show()


def transform(position: Positions, action: int, state : float, index) -> Any: #Returns : Position, Realized Profits, Unrealized Profits, Closing (False)/Opening(True) a Trade

    """
    If the 'portfolio' (Ledger) is long more than 1 one unit
    We use FIFO (first in first out) to remove the earliest Long in the Active_Long ledger / short in the Active_Short ledger
    """

    # fees = 0.0033 * state
    fees = 5
    realized_profit = 0
    realized_profit_0 = 0
    realized_profit_1 = 0

    #Updating Dollar Profits for each Ledger:
    #1. Update the State:
    Ledgers.ACTIVE_LONG["Current Price"] = [state] * len(Ledgers.ACTIVE_LONG["Entry Price"])
    Ledgers.ACTIVE_SHORT["Current Price"] = [state] * len(Ledgers.ACTIVE_SHORT["Entry Price"])
    #2. Update Profit Figures:
    Ledgers.ACTIVE_LONG["Dollar Profit"] = [current - entry for entry, current in zip(Ledgers.ACTIVE_LONG["Entry Price"], Ledgers.ACTIVE_LONG["Current Price"])]
    Ledgers.ACTIVE_SHORT["Dollar Profit"] = [entry - current for entry, current in zip(Ledgers.ACTIVE_SHORT["Entry Price"], Ledgers.ACTIVE_SHORT["Current Price"])]
    Ledgers.ACTIVE_LONG["% Return"] = [((current - entry)/entry+1) for entry, current in zip(Ledgers.ACTIVE_LONG["Entry Price"], Ledgers.ACTIVE_LONG["Current Price"])]
    Ledgers.ACTIVE_SHORT["% Return"] = [(1-(current - entry)/entry) for entry, current in zip(Ledgers.ACTIVE_SHORT["Entry Price"], Ledgers.ACTIVE_SHORT["Current Price"])]

    if action == Actions.HOLD:
        Ledgers.HIST['Entry Price'].append(state)
        Ledgers.HIST['Action Type'].append(int(action))
        Ledgers.HIST['Position Type'].append(position)
        Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])
        Ledgers.HIST['Time Index'].append(index)
        return position, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

    elif action == Actions.BUY:
        Ledgers.HIST['Time Index'].append(index)
        #Update the counter
        Positions.COUNTER += 1
        if position == Positions.SHORT:  #CLOSING A POSITION BECAUSE WE ARE IN THE SHORT
            #Save the profit of the earliest Long position & update profits positions
            realized_profit = Ledgers.ACTIVE_SHORT['Dollar Profit'][0]
            Ledgers.HIST['Dollar Profit, Realized'] += realized_profit - fees

            #Remove the earliest position : first in, first out
            Ledgers.ACTIVE_SHORT = {key: value[1:] for key, value in Ledgers.ACTIVE_SHORT.items()}

            #Update Unrealized Profits
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])

            if Positions.COUNTER < 0:
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.SHORT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) > 0 #check

                return Positions.SHORT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

            elif Positions.COUNTER == 0:
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.FLAT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) == 0 #check

                return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

        elif position == Positions.LONG or position == Positions.FLAT:  #OPENING A POSITION BECAUSE WE WERE LONG OR FLAT
            Ledgers.ACTIVE_LONG["Entry Price"].append(state)
            Ledgers.ACTIVE_LONG["Current Price"].append(state)
            Ledgers.ACTIVE_LONG["Dollar Profit"].append(0)
            Ledgers.ACTIVE_LONG["% Return"].append(1)

            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))
            Ledgers.HIST['Position Type'].append(Positions.LONG)
            Ledgers.HIST['Dollar Profit, Realized'] -= fees
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])


            # print(Ledgers.ACTIVE_SHORT['Entry Price'])
            assert len(Ledgers.ACTIVE_SHORT['Entry Price']) == 0

            return Positions.LONG, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

    elif action == Actions.SELL:
        Ledgers.HIST['Time Index'].append(index)
        #Update the counter
        Positions.COUNTER -= 1
        if position == Positions.LONG:  #CLOSING A POSITION BECAUSE WE ARE IN THE LONG
            #Save the profit of the earliest Long position & update profits positions
            realized_profits = Ledgers.ACTIVE_LONG['Dollar Profit'][0]
            Ledgers.HIST['Dollar Profit, Realized'] += realized_profits - fees

            #Remove the earliest position : first in, first out
            Ledgers.ACTIVE_LONG = {key: value[1:] for key, value in Ledgers.ACTIVE_LONG.items()}

            #Update Unrealized Profits & Ledgers.HIST
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])
            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))

            if Positions.COUNTER > 0:
                Ledgers.HIST['Position Type'].append(Positions.LONG)
                assert len(Ledgers.ACTIVE_LONG['Entry Price']) > 0 #check

                return Positions.LONG, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

            elif Positions.COUNTER == 0:
                Ledgers.HIST['Position Type'].append(Positions.FLAT)
                assert len(Ledgers.ACTIVE_LONG['Entry Price']) == 0 #check

                return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

        elif position == Positions.SHORT or position == Positions.FLAT:  #OPENING A POSITION BECAUSE WE WERE SHORT OR FLAT
            Ledgers.ACTIVE_SHORT["Entry Price"].append(state)
            Ledgers.ACTIVE_SHORT["Current Price"].append(state)
            Ledgers.ACTIVE_SHORT["Dollar Profit"].append(0)
            Ledgers.ACTIVE_SHORT["% Return"].append(1)


            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))
            Ledgers.HIST['Position Type'].append(Positions.SHORT)
            Ledgers.HIST['Dollar Profit, Realized'] -= fees

            assert len(Ledgers.ACTIVE_LONG['Entry Price']) == 0

            return Positions.SHORT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

    elif action == Actions.DOUBLE_BUY:
        Ledgers.HIST['Time Index'].append(index)
        #Update the counter
        Positions.COUNTER += 2
        if position == Positions.SHORT and Positions.COUNTER <= 0:  #CLOSING TWO POSITIONS BECAUSE WE ARE IN THE SHORT
            #Save the profit of the earliest Long position & update profits positions
            realized_profit_0 = Ledgers.ACTIVE_SHORT['Dollar Profit'][0]
            realized_profit_1 = Ledgers.ACTIVE_SHORT['Dollar Profit'][1]
            Ledgers.HIST['Dollar Profit, Realized'] += realized_profit_0 + realized_profit_1 - fees * 2

            #Remove the earliest position : first in, first out
            Ledgers.ACTIVE_SHORT = {key: value[2:] for key, value in Ledgers.ACTIVE_SHORT.items()}

            #Update Unrealized Profits
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])


            if Positions.COUNTER < 0:
                #Updating Hist Ledger
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.SHORT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) > 0 #check

                return Positions.SHORT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

            elif Positions.COUNTER == 0:
                #Updating Hist Ledger
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.FLAT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) == 0 #check

                return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

        else:  #OPENING A POSITION BECAUSE WE WERE LONG OR FLAT

            if Positions.COUNTER - 2 == -1: # that means we were Short 1 unit before and must match it / close it with a Long in FIFO mode
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) == 1
                #Closing the last Short position:
                #Saving the earliest profit
                realized_profit = Ledgers.ACTIVE_SHORT['Dollar Profit'][0]
                Ledgers.HIST['Dollar Profit, Realized'] += realized_profit - fees
                #Remove the earliest position : first in, first out
                Ledgers.ACTIVE_SHORT = {key: value[1:] for key, value in Ledgers.ACTIVE_SHORT.items()}

            else: #Opening the first Long
                Ledgers.ACTIVE_LONG["Entry Price"].append(state)
                Ledgers.ACTIVE_LONG["Current Price"].append(state)
                Ledgers.ACTIVE_LONG["Dollar Profit"].append(0)
                Ledgers.ACTIVE_LONG["% Return"].append(1)

            #Open the second Long
            Ledgers.ACTIVE_LONG["Entry Price"].append(state)
            Ledgers.ACTIVE_LONG["Current Price"].append(state)
            Ledgers.ACTIVE_LONG["Dollar Profit"].append(0)
            Ledgers.ACTIVE_LONG["% Return"].append(1)

            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))
            Ledgers.HIST['Position Type'].append(Positions.LONG)
            Ledgers.HIST['Dollar Profit, Realized'] -= fees * 2
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])

            # print(Ledgers.ACTIVE_SHORT['Entry Price'])
            assert(len(Ledgers.ACTIVE_SHORT['Entry Price'])) == 0

            return Positions.LONG, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

    elif action == Actions.DOUBLE_SELL:
        Ledgers.HIST['Time Index'].append(index)
        #Update the counter
        Positions.COUNTER -= 2
        if position == Positions.LONG and Positions.COUNTER >= 0:  #CLOSING A POSITION BECAUSE WE ARE IN THE LONG
            #Save the profit of the earliest Long position & update profits positions
            realized_profit_0 = Ledgers.ACTIVE_LONG['Dollar Profit'][0]
            realized_profit_1 = Ledgers.ACTIVE_LONG['Dollar Profit'][1]
            Ledgers.HIST['Dollar Profit, Realized'] += realized_profit_0 + realized_profit_1 - fees * 2

            #Remove the earliest position : first in, first out
            Ledgers.ACTIVE_LONG = {key: value[2:] for key, value in Ledgers.ACTIVE_LONG.items()}

            #Update Unrealized Profits
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])
            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))

            if Positions.COUNTER > 0:
                Ledgers.HIST['Position Type'].append(Positions.LONG)
                assert len(Ledgers.ACTIVE_LONG['Entry Price']) > 0 #check
                return Positions.LONG, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

            elif Positions.COUNTER == 0:
                Ledgers.HIST['Position Type'].append(Positions.FLAT)
                assert len(Ledgers.ACTIVE_LONG['Entry Price']) == 0 #check
                return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

        else:  #OPENING A POSITION BECAUSE WE WERE SHORT OR FLAT
            if Positions.COUNTER + 2 == 1: # that means we were Long 1 unit before and must match it / close it with a Short FIFO mode
                #First closing:
                #Remove the earliest position : first in, first out
                realized_profit = Ledgers.ACTIVE_LONG['Dollar Profit'][0]
                Ledgers.HIST['Dollar Profit, Realized'] += realized_profit - fees
                Ledgers.ACTIVE_LONG = {key: value[1:] for key, value in Ledgers.ACTIVE_LONG.items()}

            else: #First Shorting
                Ledgers.ACTIVE_SHORT["Entry Price"].append(state)
                Ledgers.ACTIVE_SHORT["Current Price"].append(state)
                Ledgers.ACTIVE_SHORT["Dollar Profit"].append(0)
                Ledgers.ACTIVE_SHORT["% Return"].append(1)

            #Second Shorting
            Ledgers.ACTIVE_SHORT["Entry Price"].append(state)
            Ledgers.ACTIVE_SHORT["Current Price"].append(state)
            Ledgers.ACTIVE_SHORT["Dollar Profit"].append(0)
            Ledgers.ACTIVE_SHORT["% Return"].append(1)

            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))
            Ledgers.HIST['Position Type'].append(Positions.SHORT)
            Ledgers.HIST['Dollar Profit, Realized'] -= fees * 2
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])

            assert len(Ledgers.ACTIVE_LONG['Current Price']) == 0

            return Positions.SHORT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

    elif action == Actions.COMBO_BUY:
        Ledgers.HIST['Time Index'].append(index)
        #Update the counter
        Positions.COUNTER += 6
        if position == Positions.SHORT and Positions.COUNTER <= 0:  #CLOSING x POSITIONS BECAUSE WE ARE IN THE SHORT
            #Save the profit of the earliest Long position & update profits positions
            realized_profit_temp = 0
            for i in range(6):
                realized_profit_temp =+ Ledgers.ACTIVE_SHORT['Dollar Profit'][i]

            Ledgers.HIST['Dollar Profit, Realized'] += realized_profit_temp - fees * 6

            #Remove the earliest position : first in, first out
            # print(Ledgers.ACTIVE_LONG['Current Price'])
            Ledgers.ACTIVE_SHORT = {key: value[6:] for key, value in Ledgers.ACTIVE_SHORT.items()}

            #Update Unrealized Profits
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])


            if Positions.COUNTER < 0:
                #Updating Hist Ledger
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.SHORT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) > 0 #check

                return Positions.SHORT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

            elif Positions.COUNTER == 0:
                #Updating Hist Ledger
                Ledgers.HIST['Entry Price'].append(state)
                Ledgers.HIST['Action Type'].append(int(action))
                Ledgers.HIST['Position Type'].append(Positions.FLAT)

                # print(Ledgers.ACTIVE_SHORT['Entry Price'])
                assert len(Ledgers.ACTIVE_SHORT['Entry Price']) == 0 #check

                return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

        else:  #OPENING A POSITION BECAUSE WE WERE LONG OR FLAT
            if 0 < Positions.COUNTER < 6: # that means we were Short Position.Count - 6 units before and must match it / close it with a Long in FIFO mode
                #Closing the last Short position:
                realized_profit_temp = 0
                indexer = abs(Positions.COUNTER - 6)
                for i in range(indexer):
                    realized_profit_temp += Ledgers.ACTIVE_SHORT['Dollar Profit'][i]

                Ledgers.HIST['Dollar Profit, Realized'] += realized_profit_temp - fees * indexer
                #Remove the earliest position : first in, first out
                Ledgers.ACTIVE_SHORT = {key: value[indexer:] for key, value in Ledgers.ACTIVE_SHORT.items()}

                #Open the rest of the longs
                for i in range(6 - indexer):
                    Ledgers.ACTIVE_LONG["Entry Price"].append(state)
                    Ledgers.ACTIVE_LONG["Current Price"].append(state)
                    Ledgers.ACTIVE_LONG["Dollar Profit"].append(0)
                    Ledgers.ACTIVE_LONG["% Return"].append(1)

            else: #Updating Long positions
                #to improve : simulate order book thinening
                for i in range(6):
                    Ledgers.ACTIVE_LONG["Entry Price"].append(state)
                    Ledgers.ACTIVE_LONG["Current Price"].append(state)
                    Ledgers.ACTIVE_LONG["Dollar Profit"].append(0)
                    Ledgers.ACTIVE_LONG["% Return"].append(1)


            Ledgers.HIST['Entry Price'].append(state)
            Ledgers.HIST['Action Type'].append(int(action))
            Ledgers.HIST['Position Type'].append(Positions.LONG)
            Ledgers.HIST['Dollar Profit, Realized'] -= fees * 2
            Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])

            # print(Ledgers.ACTIVE_SHORT['Entry Price'])
            assert(len(Ledgers.ACTIVE_SHORT['Entry Price'])) == 0

            return Positions.LONG, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], True

    elif action == Actions.CLOSE_ALL:
        Ledgers.HIST['Time Index'].append(index)
        Ledgers.HIST['Entry Price'].append(state)
        Ledgers.HIST['Action Type'].append(int(action))
        Ledgers.HIST['Position Type'].append(Positions.FLAT)
        Ledgers.HIST['Dollar Profit, Unrealized'] = sum(Ledgers.ACTIVE_LONG['Dollar Profit']) + sum(Ledgers.ACTIVE_SHORT['Dollar Profit'])
        Ledgers.HIST['Dollar Profit, Realized'] += (Ledgers.HIST['Dollar Profit, Unrealized']
                                                    - fees * (abs(Positions.COUNTER)))

        #Penalty to simulate cost of market impact from market/panic selling:
        if Ledgers.HIST['Dollar Profit, Realized'] > 0:
            Ledgers.HIST['Dollar Profit, Realized'] *= (1 - market_impact)
        else:
            Ledgers.HIST['Dollar Profit, Realized'] *= (1 + market_impact)

        Ledgers.HIST['Dollar Profit, Unrealized'] = 0

        #Resetting:
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

        Positions.COUNTER = 0

        return Positions.FLAT, Ledgers.HIST['Dollar Profit, Realized'], Ledgers.HIST['Dollar Profit, Unrealized'], False

