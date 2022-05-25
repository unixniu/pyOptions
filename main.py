import datetime
from hist_analysis import History
import sys
from playbook import Playbook

from replay import Account

def wrongarguments():
    print('''wrong arguments. examples:
        1. python main.py hist 510050C2109M03000
        2. python main.py replay ./playbook.csv 20220330
    ''')
    exit(2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        wrongarguments()

    History.loaddata()

    cmd = sys.argv[1]
    if cmd.lower() == 'hist':
        if len(sys.argv) < 3:
            wrongarguments()
        contractNum = sys.argv[2]
        History.visual(contractNum)
    elif cmd.lower() == 'replay': 
        if len(sys.argv) < 3:
            wrongarguments()
        playbook = sys.argv[2]
        if playbook.endswith('csv'):    # csv playbook runs for single account
            end = datetime.datetime.strptime(sys.argv[3], "%Y%m%d") if len(sys.argv) >= 4 else datetime.date.today()
            df = Account.replay(playbook, end)
            print(df)
            #df.to_csv('res.csv')
            Account.visual(df)
        elif playbook.endswith('yml'):  # yml playbook runs multiple accounts
            pb = Playbook(playbook)
            df = pb.run()
            #print(df)
            df.to_csv('res.csv')
            pb.visual_multiaccounts(df)



