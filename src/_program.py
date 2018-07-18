import sys
import time as t
import datetime as dt
import utils
import math
import pandas as pd
#from week1_intro import header, run
#from week2_leakages import header, run
from week3_metrics import header, run
from ideas.plural_stacking import header, run
#from exam import header, run


def main(args=None):
    args = args or sys.argv[1:]

    utils.PRINT.HEADER(header())
    print('STARTED at ', dt.datetime.now(), 'with args: ', args)
    start = t.time()
    run()
    end = t.time()
    utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

    return

if __name__=='__main__':
    main(sys.argv[1:])
