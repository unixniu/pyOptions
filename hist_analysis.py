import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from calc import greeks, call_iv, put_iv

INTEREST_RATE = 0.03
HISTORICAL_IV = 0.262

class History:
    past_contracts_lookup_df = None
    active_contracts_lookup_df = None
    current_months = None
    spot_daily_df = None
    loaded = False

    def __init__(self) -> None:
        pass

    @classmethod
    def loaddata(cls):
        if cls.loaded:
            return
            
        past_contracts_lookup_df = pd.read_csv('past-50ETF-options-lookup.csv')
        past_contracts_lookup_df['exp'] = pd.to_datetime(
            past_contracts_lookup_df['expiry'], 
            format='%Y/%m/%d')
        cls.past_contracts_lookup_df = past_contracts_lookup_df

        active_df = ak.option_value_analysis_em()
        active_df['expire'] = pd.to_datetime(active_df['到期日'])
        cls.active_contracts_lookup_df = active_df

        cls.current_months = ak.option_sse_list_sina(symbol="50ETF", exchange="null")

        spot_daily_df = ak.fund_etf_hist_sina(symbol="sh510050")
        spot_daily_df['date'] = pd.to_datetime(spot_daily_df['date'])
        cls.spot_daily_df = spot_daily_df

        cls.loaded = True

    @classmethod
    def ensure_loaded(cls):
        if not cls.loaded:
            raise Exception('data not pre-loaded. call \'loaddata\' upfront.')

    @classmethod
    def get_code(cls, contractNum:str):
        cls.ensure_loaded()
        if len(contractNum) != 17:
            raise Exception('Not a valid contract number.')
        
        spot_name = contractNum[3:6].lstrip('0') + 'ETF'
        type = '购' if contractNum[6] == 'C' else '沽'
        month = contractNum[9:11].lstrip('0')
        strike = contractNum[-5:].lstrip('0')
        name = f'{spot_name}{type}{month}月{strike}'
        found = cls.active_contracts_lookup_df.loc[cls.active_contracts_lookup_df['期权名称']==name, ['期权代码', 'expire']]
        if len(found) == 0:
            raise Exception(f'code of {contractNum} not found.')
        return found.values[0]

    @classmethod
    def is_active(cls, contract:str):
        return ('20' + contract[7:11]) in cls.current_months

    # get history daily data for given contract (e.g. 510050C2109M03000)
    @classmethod
    def hist(cls, contractNum:str):
        cls.ensure_loaded()
        if len(contractNum) != 17:
            raise Exception('not a valid contract number.')
        
        symbol, expire = '', None
        if cls.is_active(contractNum):
            symbol, expire = cls.get_code(contractNum)
        else:
            symbol, expire = cls.past_contracts_lookup_df.loc[
                cls.past_contracts_lookup_df['code']==contractNum, 
                ['num', 'exp']].values[0]
        k = int(contractNum[-4:])/1000
        type = contractNum[6]
        print(f'symbol:{symbol}, expiredate: {expire}, strike: {k}')
        
        # get daily kline for the contract
        df = ak.option_sse_daily_sina(symbol=str(symbol))
        df.rename(columns={'收盘':'T'}, inplace=True)
        df['date'] = pd.to_datetime(df['日期'])
        
        # merge with underlying (ETF50) daily kline
        merged = pd.merge(df, cls.spot_daily_df, how='left', on='date')
        merged['intrisicVal'] = merged['close'].map(lambda x: max(0, (x - k) * (1 if type == 'C' else -1)))
        merged['timeVal'] = merged['T'] - merged['intrisicVal']
        merged['tte'] = merged['date'].map(lambda x: (expire - x).days)
        #merged['s_diff'] = merged['close'].diff()
        #merged['t_diff'] = merged['T'].diff()
        greeks_df = merged.apply( 
            lambda x: 
                [round(x, 3) for x in greeks(x['close'], k, HISTORICAL_IV, INTEREST_RATE, x['tte']/365, type)] 
                if x['tte'] > 0 
                else np.nan, 
            axis=1,
            result_type='expand')
        merged = merged.join(greeks_df)
        iv_func = call_iv if type == 'C' else put_iv
        merged['iv'] = merged.apply(
            lambda x: 
                round(iv_func(x['T'], x['close'], k, x['tte']/365), 3) 
                if x['tte'] > 0 
                else np.nan, 
            axis=1)
        merged.rename(columns={0:'delta',1:'gamma',2:'theta',3:'vega', 'close':'S'}, inplace=True)
        merged = merged.reindex(columns=['date', 'T', 'S', 'intrisicVal', 'timeVal', 'delta', 'gamma', 'theta', 'vega', 'iv'])
        merged.set_index('date', inplace=True)
        return merged

    @classmethod
    def visual(cls, contractNum:str):
        df = History.hist(contractNum)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
            specs=[
                [{'secondary_y':True}], 
                [{'secondary_y':True}]])
        delta_tr = go.Line(x=df.index, y=df['delta'], name='delta')
        iv_tr = go.Line(x=df.index, y=df['iv'], name='iv')
        theta_tr = go.Line(x=df.index, y=df['theta'], name='theta')
        fig.add_traces([delta_tr, iv_tr, theta_tr], rows=1, cols=1, secondary_ys=[False, False])
        timeval_tr = go.Bar(x=df.index, y=df['timeVal'], name='TimeValue')
        intrval_tr = go.Bar(x=df.index, y=df['intrisicVal'], name='IntrisicValue')
        spot_tr = go.Line(x=df.index, y=df['S'], name='Spot')
        option_tr = go.Line(x=df.index, y=df['T'], name='Option')
        fig.add_traces([spot_tr, option_tr, intrval_tr, timeval_tr], rows=2, cols=1, 
            secondary_ys=[True, False, False, False])
        fig.update_layout(barmode='stack', height=800, title=contractNum)
        fig.show()