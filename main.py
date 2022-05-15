import imp
from hist_analysis import History
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('wrong arguments.')
        exit(2)

    contractNum = sys.argv[1]
    History.loaddata()
    History.visual(contractNum)