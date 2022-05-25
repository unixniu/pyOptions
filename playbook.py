from bcrypt import os
import yaml
import pandas as pd
from hist_analysis import History
from replay import Account
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Playbook:
    def __init__(self, filepath:str):
        if not os.path.isfile(filepath):
            raise Exception(f'playbook at {filepath} not existing. ')

        self.filepath = filepath

    def run(self):
        account_replay_dfs = []
        with open(self.filepath, 'r', encoding='utf_8') as f:
            ymlmap = yaml.safe_load(f)
            for acc_spec in ymlmap['accounts']:
                # only support account spec having 'trades'
                if 'trades' in acc_spec:
                    account = Account(acc_spec)
                    account_replay_dfs.append(account.rerun())
            concated = pd.concat(account_replay_dfs, axis=1, join='outer', keys=[x['name'] for x in ymlmap['accounts']])
            spot = History.spot_daily_df.reindex(columns=['close']).rename(columns={'close': 'spot'}, copy=False)
            return concated.join(spot)


    def visual_multiaccounts(self, concated_df:pd.DataFrame):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
            specs=[
                [{'secondary_y':True}], 
                [{'secondary_y':True}]])
        
        spot_tr = go.Line(x=concated_df.index, y=concated_df['spot'], name='spot')
        pl_amt_traces, pl_pct_traces = [], []
        for col in concated_df.columns:
            if 'pl_amt' == col[1]:
                pl_amt_traces.append(go.Bar(x=concated_df.index, y=concated_df[col], name=col[0]))
            if 'pl_pct' == col[1]:
                pl_pct_traces.append(go.Scatter(x=concated_df.index, y=concated_df[col], name=col[0]))
        
        fig.add_traces(pl_amt_traces + [spot_tr], rows=1, cols=1, secondary_ys=[False] * len(pl_amt_traces) + [True])
        fig.add_traces(pl_pct_traces + [spot_tr], rows=2, cols=1, secondary_ys=[False] * len(pl_pct_traces) + [True])
        
        fig.update_layout(height=600, title='replay')
        fig.show()


    def visual_singleaccount(self, replay_df:pd.DataFrame):
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