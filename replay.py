from collections import namedtuple
from datetime import date, timedelta
from doctest import master
import re
import os
import numpy
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sympy import N
from hist_analysis import History

NUM_PER_LOT = 10000 # 1 lot of contract contains 10000 units

Tran = namedtuple("Transaction", "vol price date")

#Trade = namedtuple('Trade', 'name symbol direction volume date closePolicy triggers done', defaults=['', '', 'L', 0, None, None, None, False])

class Trade:
    def __init__(self, **args) -> None:
        self.name = args.get('name', '')
        self.symbol = args.get('symbol', '')
        self.direction = args.get('direction', 'L')
        self.volume = args.get('volume', 0)
        self.date = args.get('date')
        self.closePolicy = args.get('closePolicy')
        self.triggers = args.get('triggers')
        self.master = args.get('master')    # for trade affilicated to a master one, e.g. closetrade created implicitly after opentrade
        self.done = False

# option position.
class Position:
    # symbol: contract num like '510050C2109M03000'
    # direction: L (long) or S (short)
    # exmaple: Pos('510050C2109M03000', 'L')
    def __init__(self, symbol, direction, username=None) -> None:
        #self.tranx = []
        self.symbol, self.direction = symbol, direction
        self.username = username
        self.kind = symbol[6]
        self.vol, self.cost_price, self.pl = 0, 0.0, 0.0
        self.closed = False
    
    def autoname(self):
        return self.symbol + self.direction

    def name(self):
        return self.autoname() if not self.username else self.username

    @classmethod
    def hist_lookup(cls, func):
        cls.hist_lookup = func

    @classmethod
    def Build(cls, symbol, direction, vol, price, date):
        pos = Position(symbol, direction)
        pos.change(vol, price, date)
        print(f"{date}: position built: [{symbol}, {direction}, {vol} at {price}]")
        return pos

    @classmethod
    def Build_w_trade(cls, opening_trade:Trade, date):
        pos = Position(opening_trade.symbol, opening_trade.direction, opening_trade.name)
        price = cls.hist_lookup(pos.symbol, date)
        pos.change(opening_trade.volume, price, date)
        opening_trade.done = True
        print(f"{date}: position built: [{pos.symbol}, {pos.direction}, {pos.vol} at {price}]")
        return pos

    def close(self, price, date):
        if self.closed:
            return

        self.change(-self.vol, price, date)

    def execute(self, trade:Trade, date):
        if self.closed:
            trade.done = True
            return

        delta = trade.volume
        price = Position.hist_lookup(self.symbol, date)
        self.change(delta, price, date)
        trade.done = True

    def change(self, delta, price, date):
        # normalize cost price
        newVol = self.vol + delta
        if newVol > 0:
            self.cost_price = (self.cost_price * self.vol + delta * price) / newVol
            self.vol = newVol
        else:
            if self.vol == 0:
                raise Exception('try to close an empty position.')

            # position's final profit&loss is only available after it's closed
            self.pl = self.p_l(price)
            self.closed = True
        #self.tranx.append(Tran(delta, price, date))
        print(f"{date}: position change: [{delta} {self.symbol}, {self.direction}, at {price}],", f"remaining: {self.vol}" if not self.closed else 'position closed')

    def p_l(self, price):
        if self.closed:
            raise Exception('realtime p_l not applicable for clsoed position')
        amt = (self.vol * (price - self.cost_price) if self.direction == 'L' else self.vol * (self.cost_price - price)) * NUM_PER_LOT
        cost = self.vol * self.cost_price * NUM_PER_LOT
        return amt, amt/cost


class Trigger(dict):
    def __init__(self, type, trade, **kargs) -> None:
        self.type = type
        self.trade = trade
        super().__init__(kargs)


CLOSE_POSITION_VOL = - 1 << 32


class Account:
    _histcache = {}
    _posMap = {}

    def __init__(self, account_spec:dict) -> None:
        self._posMap = {}
        self.name = account_spec.get('name', '')
        self.start = account_spec.get('start', None)
        self.end = account_spec.get('end', date.today())
        
        # get all trades
        #breakpoint()
        self.trades = [Trade(**x) for x in account_spec['trades']]
        for trade in self.trades:
            if trade.closePolicy:
                self.trades.append(Account.gen_close_trade(trade.closePolicy, trade))
        

    # get hist daily k-line for given contract in df of following columns
    # ['date', 'T', 'S', 'intrisicVal', 'timeVal', 'delta', 'gamma', 'theta', 'vega', 'iv']
    @classmethod
    def gethist(cls, symbol:str):
        if symbol not in cls._histcache:
            cls._histcache[symbol] = History.hist(symbol)
        return cls._histcache[symbol]


    @classmethod
    def hist_data(cls, symbol:str, date:date):
        hist_df = cls.gethist(symbol)
        date_index = pd.Timestamp(date)
        if hist_df is None:
            raise Exception(f'history data not available for {symbol}')
        if date_index not in hist_df.index:
            return None
            #raise Exception(f'hist data missing for {symbol} on {date}')
        return hist_df.loc[date_index].to_dict()


    @classmethod
    def price(cls, symbol:str, date:date):
        data = cls.hist_data(symbol, date)
        if not data:
            raise Exception(f'hist data missing for {symbol} on {date}')
        return data['T']


    @staticmethod
    def gen_close_trade(closepolicy:list, opentrade:Trade):
        return Trade(
            symbol=opentrade.symbol, 
            direction=opentrade.direction, 
            volume=CLOSE_POSITION_VOL, 
            triggers=closepolicy,
            master=opentrade)


    @staticmethod
    def position_name(trade:Trade):
        return trade.symbol + trade.direction


    def getposition(self, trade:Trade):
        return self._posMap.get(Account.position_name(trade))


    def execute(self, trade, date):
        pos_name = Account.position_name(trade)
        if pos_name not in self._posMap:
            if not re.match(r'510050[CP]\d{4}M\d{5}', trade.symbol):
                if not date:
                    raise Exception('date missing for resolving dynmaic name.')
                trade.symbol = Account.resolveContract(trade.symbol, date)
                for tr in self.trades:
                    if tr.master == trade:
                        tr.symbol = trade.symbol

            pos = Position.Build_w_trade(trade, date)
            # add position to map using both autoname (contract + direction) and name (user-given name or autoname)
            self._posMap[pos.autoname()] = pos
            self._posMap[pos.name()] = pos
        else:
            self._posMap[pos_name].execute(trade, date)


    @staticmethod
    def atm(spot_price:float):
        x = round(spot_price, 1)
        if x < 3.0:
            mid = x + (0.05 if x < spot_price else -0.05)
            return mid if abs(spot_price - x) > abs(spot_price - mid) else x
        else: 
            return x


    ''' resolve a dynamic 50ETF option contract name on given date, in format {CxMy} or {PxMy}
       - C is Call, P is Put
       - x is in [1, 9], x=5 is at-the-money strike; x1 < x2 < x3 < x4 < x5 < x6 < x7 < x8 < x9
       - y is in [1, 4], meaning current month, next month, next season month, next next season month
       - passed date is used to get spot price on that day
       e.g. resolveContract('{P9M3}', '20211220') => 510050P2203M03600, spot price on 20211220 is 3.243.
    ''' 
    @staticmethod
    def resolveContract(dynamicname:str, date:date, adjust=False):
        m = re.match(r'([CP]\dM\d)', dynamicname)
        if not m:
            raise Exception(f"{dynamicname} can't be resolved to a valid contract.")
        var = m.group(1)
        ts = pd.Timestamp(date)
        History.loaddata()
        if ts not in History.spot_daily_df.index:
            raise Exception(f'spot price not found on {date}')
        spot_price = History.spot_daily_df.at[ts, 'close']
        type, strike_level, month_ord = var[0], int(var[1]), int(var[3])

        strike = Account.atm(spot_price)
        print(f'atm:{strike}')
        n = strike_level - 5    # levels away from ATM
        if n != 0:
            direction = n / abs(n)      # move up or down the ATM strike price
            for i in range(0, abs(n)):
                step = 0.05 if strike < 3.0 or (strike == 3.0 and direction == -1) else 0.1
                strike += step * direction
                print(f'step {step} => {strike}')
        
        expiredate = ts     # current month
        # when adjust flag is on, if choosing current month and it's apporaching month end move to next month
        if (month_ord == 1 and adjust and ts.day >= 20) or month_ord == 2:
            expiredate = ts + pd.DateOffset(months=1)   # next month
        elif month_ord == 3 or month_ord == 4:
            offset = pd.offsets.QuarterEnd(1 if month_ord == 3 else 2, startingMonth=3)
            expiredate = offset.rollforward(ts) + offset    # ending month of next quarter or next next quater
        
        contract_code = f"510050{type}{expiredate.strftime('%y%m')}M{'{:05d}'.format(int(round(strike, 2) * 1000))}"
        print(f"dynmaic name '{dynamicname} on {date} resolved to {contract_code}")
        return contract_code


    def rerun(self):
        #breakpoint()
        if self.start is None:
            self.start = min((tr.date for tr in self.trades if tr.date is not None), default=None)
            if self.start is None:
                raise Exception('no start date.')
        
        res = {}
        today = self.start
        while True:
            if self.end is not None and today > self.end:
                break

            # execute date-triggered trades
            for tr in self.trades:
                if tr.date == today and not tr.done:
                    self.execute(tr, today)

            # forcibly close position on last day of past contract
            for pos in self._posMap.values():
                if not History.is_active(pos.symbol) and not pos.closed and today == Account.gethist(pos.symbol).index[-1]:
                    pos.close(Account.hist_data(pos.symbol, today)['T'], today)
            
            # test trades' triggers
            while True:
                trade_executed = False
                for trade in filter(lambda x: not x.done, self.trades):
                    for trigger in trade.triggers:
                        type, match = trigger['type'], False
                        if type == 'pl':
                            pos = self.getposition(trade)
                            if pos and not pos.closed:
                                data = Account.hist_data(pos.symbol, today)
                                if data:
                                    _, pl_pct = pos.p_l(data['T'])
                                    exp = trigger['expression'].replace('$', str(pl_pct))
                                    match = eval(exp) is True
                                    #print(f'{exp} => {eval(exp)}')
                        elif type == 'linked':
                            position = self._posMap.get(trigger['position'])
                            on = trigger['on']
                            if on == 'open':
                                match = position is not None
                            elif on == 'close':
                                match = position is not None and position.closed
                        elif type == 'price':
                            price = Account.hist_data(trade.symbol, today)['T']
                            match = eval(trigger['expression'].replace('$', str(price))) is True
                        
                        if match:
                            self.execute(trade, today)
                            trade_executed = True
                            break
                
                # if no trade is executed, no need for another round of test of triggers
                if not trade_executed:
                    break
            
            total_pl_amount, total_cost, total_delta, total_active_vol, total_long_vol, total_short_vol = 0.0, 0.0, 0.0, 0, 0, 0
            hasdata = False
            for pos in self._posMap.values():
                data = Account.hist_data(pos.symbol, today)
                if data is None:
                    continue

                hasdata = True
                total_pl_amount += pos.p_l(data['T'])[0] if not pos.closed else pos.pl[0]
                total_cost += pos.vol * pos.cost_price * NUM_PER_LOT
                total_delta += pos.vol * data['delta'] * (1 if pos.direction == 'L' else -1) if not pos.closed else 0
                #total_active_vol += pos.vol if not pos.closed else 0
                if pos.direction == "L":
                    total_long_vol += pos.vol if not pos.closed else 0
                else:
                    total_short_vol += pos.vol if not pos.closed else 0
            
            if hasdata:
                day_res = {}
                day_res['pl_amt'] = round(total_pl_amount, 1)
                day_res['pl_pct'] = round(total_pl_amount / total_cost, 3) if total_cost != 0 else numpy.nan
                day_res['delta'] = (total_delta / (total_short_vol + total_long_vol)) if total_short_vol + total_long_vol > 0 else 0
                day_res['long_vol'] = total_long_vol
                day_res['short_vol'] = total_short_vol
                res[today] = day_res

            # if all positions are closed and no more trades to be done, break out
            if all(x.closed for x in self._posMap.values()) and all(x.done for x in self.trades):
                break

            today += timedelta(days=1)
        
        res = pd.DataFrame.from_dict(res, orient='index')
        res.index = pd.to_datetime(res.index)
        return res


    @classmethod
    def replay(cls, csv_playbook_path:str, end:date):
        if not os.path.isfile(csv_playbook_path):
            raise FileNotFoundError(csv_playbook_path)

        cls._posMap.clear()
        playbook_df = pd.read_csv(csv_playbook_path, parse_dates=True, names=['contract', 'direction', 'vol'], skipinitialspace=True)
        playbook_df.sort_index(inplace=True)

        start = playbook_df.index[0]
        current = start
        oneday = pd.Timedelta('1d')
        res = {}
        while True:
            if end is not None and current > end:
                break
            
            trades = playbook_df[playbook_df.index == current]
            for trade in playbook_df[playbook_df.index == current].itertuples():
                hist_df = Account.gethist(trade.contract)
                if current not in hist_df.index:
                    print(f'trade {trade.contract} not possible on {current}')
                    continue

                data = hist_df.loc[current].to_dict()
                posname = trade.contract + trade.direction
                if posname not in cls._posMap:
                    cls._posMap[posname] = Position.Build(trade.contract, trade.direction, trade.vol, data['T'], current)
                else:
                    cls._posMap[posname].change(trade.vol, data['T'], current)

            
            day_res = {}
            total_pl_amount, total_cost, total_delta, total_active_vol, total_long_vol, total_short_vol = 0.0, 0.0, 0.0, 0, 0, 0
            has_data = False
            for pos in cls._posMap.values():
                hist_df = Account.gethist(pos.symbol)
                if current not in hist_df.index:
                    continue

                has_data = True
                data = hist_df.loc[current].to_dict()
                # close position on last day of contract if it expires already
                if current == hist_df.index[-1] and not History.is_active(pos.symbol):
                    pos.close(data['T'], current)

                total_pl_amount += pos.p_l(data['T'])[0] if not pos.closed else pos.pl[0]
                total_cost += pos.vol * pos.cost_price * NUM_PER_LOT
                total_delta += pos.vol * data['delta'] * (1 if pos.direction == 'L' else -1) if not pos.closed else 0
                total_active_vol += pos.vol if not pos.closed else 0
                day_res['spot'] = data['S']
                if pos.direction == "L":
                    total_long_vol += pos.vol
                else:
                    total_short_vol += pos.vol
            
            # if it's not trading day, has_data is false and that date can be omited in result
            if has_data:
                day_res['pl_amt'] = round(total_pl_amount, 1)
                day_res['pl_pct'] = round(total_pl_amount / total_cost, 3) if total_cost != 0 else numpy.nan
                day_res['delta'] = (total_delta / total_active_vol) if total_active_vol > 0 else 0
                day_res['long_vol'] = total_long_vol
                day_res['short_vol'] = total_short_vol
                res[current] = day_res

            # if all positions are closed, break out
            if len(cls._posMap) > 0 and all(pos.closed for pos in cls._posMap.values()):
                break

            current += oneday
        
        return pd.DataFrame.from_dict(res, orient='index')

    @classmethod
    def visual(cls, replay_df:pd.DataFrame):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
            specs=[
                [{'secondary_y':True}], 
                [{}]])
        
        spot_tr = go.Line(x=replay_df.index, y=replay_df['spot'], name='spot')
        pl_amt_tr = go.Line(x=replay_df.index, y=replay_df['pl_amt'], name='p&l-amount')
        fig.add_traces([spot_tr, pl_amt_tr], rows=1, cols=1, secondary_ys=[False, True])
        
        pl_pct_tr = go.Line(x=replay_df.index, y=replay_df['pl_pct']*100, name='p&l-percent')
        delta_tr = go.Line(x=replay_df.index, y=replay_df['delta'], name='delta')
        long_vol_tr = go.Bar(x=replay_df.index, y=replay_df['long_vol'], name='long vol')
        short_vol_tr = go.Bar(x=replay_df.index, y=replay_df['short_vol'], name='short vol')
        fig.add_traces([pl_pct_tr, delta_tr, long_vol_tr, short_vol_tr], rows=2, cols=1)

        # setting yaxis in go.Scatter doesn't work
        # so update this property explicitly
        fig['data'][2]['yaxis'] = 'y4'  # p&l series
        fig['data'][3]['yaxis'] = 'y5'  # delta series

        fig['layout']['yaxis4']=dict(
            title="pl",
            overlaying="y3",
            anchor="x2",
            side="right"
        )
        fig['layout']['yaxis5']=dict(
            title="delta",
            anchor="free",
            overlaying='y3',
            range=[-1, 1],
            side="right",
            position=1
        )
        fig.update_layout(barmode='stack', height=600, title='replay')
        fig.show()

Position.hist_lookup = Account.price



