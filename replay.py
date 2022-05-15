
from collections import namedtuple
from datetime import date
import pandas as pd

from hist_analysis import History

NUM_PER_LOT = 10000 # 1 lot of contract contains 10000 units

#Pos = namedtuple("Position", contract, vol, openDate, closeDate)
Tran = namedtuple("Transaction", "vol price date")

# option position.
# example contract: 510050C2109M03000
class Pos:
    def __init__(self, contract) -> None:
        self.tranx = []
        self.contract = contract
        self.kind = contract[6]
        self.vol, self.cost_price, self.pl = 0, 0.0, 0.0

    @classmethod
    def Build(cls, contract, vol, price, date):
        pos = Pos(contract)
        pos.build(vol, price, date)
        return pos

    def build(self, vol, price, date):
        self.change(vol, price)
        self.tranx.append(Tran(vol, price, date))

    def decrease(self, vol, price, date):
        self.change(-vol, price)
        self.tranx.append(Tran(-vol, price, date))

    def close(self, price, date):
        self.change(-self.vol, price)
        self.tranx.append(Tran(-self.vol, price, date))

    def change(self, delta, price, date):
        # normalize cost price
        newVol = self.vol + delta
        if newVol != 0:
            self.cost_price = (self.cost_price * self.vol + delta * price) / newVol
        else:
            # position's final profit&loss is only available after it's closed
            self.pl = self.p_l(price)
        self.vol = newVol
        self.tranx.append(Tran(delta, price, date))

    def p_l(self, price):
        return (self.vol * (price - self.cost_price) if self.kind == 'C' else self.vol * (self.cost_price - price)) * NUM_PER_LOT
            

class Account:
    _histcache = {}

    def __init__(self, initCap) -> None:
        self._initCap = initCap
        self._posMap = {}

    # trade the contract for given volume (in lots) at given time (if none take starting date of the contract)
    def trade(self, contract:str, vol:int, date:date=None):
        # new position to build
        if contract not in self._posMap and vol < 0:
            raise Exception('negative vol to build position')

        if contract not in self.__class__._histcache:
            self.__class__._histcache[contract] = History.hist(contract)
        # daily k-line df for the contract with columns: 
        # ['date', 'T', 'S', 'intrisicVal', 'timeVal', 'delta', 'gamma', 'theta', 'vega', 'iv']
        hist_df = self.__class__._histcache[contract]
        if hist_df is None or len(hist_df) == 0:
            raise Exception('history data not available for', contract)
        
        date = hist_df.index[0] if date is None else pd.Timestamp(date)
        price = hist_df.loc[date, 'T']
        if contract not in self._posMap:
            self._posMap[contract] = Pos.Build(contract, vol, price, date)
        else:
            self._posMap[contract].change(vol, price, date)
