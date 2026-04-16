from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap

ratesSymbol = "USDSEK"
ratesTimeFrame = mt5.TIMEFRAME_M1
dateEnd = datetime(2026, 3, 20) + timedelta(days=1) #Set YYYYMMDD for the last day you want data
dateStart = dateEnd - timedelta(days=(1))


def main():
    print("This is main")
    ratesData = GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd)


def GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    ratesData = pd.DataFrame(mt5.copy_rates_range(ratesSymbol,ratesTimeFrame,dateStart,dateEnd))
    print(ratesData.columns)

    mt5.shutdown()
    return(ratesData)

if __name__ == '__main__':
    main()