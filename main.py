from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import pytz
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import tight_layout


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def main():
    ratesSymbol = "USDSEK"
    ratesTimeFrame = mt5.TIMEFRAME_M1
    #utcTimeFrom = datetime(2026, 2, 26, tzinfo=pytz.timezone("Etc/UTC"))
    dateStart = datetime.today() - timedelta(days=(10*365))
    #print(dateStart)
    dateEnd = datetime.now()
    #print(dateEnd)
    periodCount = 1440

    MAWindowSize = 30
    MAPrice = "close"

    BBPeriod = 20
    BBStandardDeviations = 2

    print("This is the main function")
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    ratesData = pd.DataFrame(mt5.copy_rates_range(ratesSymbol,ratesTimeFrame,dateStart,dateEnd))

    mt5.shutdown()

    ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
    ratesData = MACalculator(ratesData, MAWindowSize, MAPrice)
    ratesData = BollingerBandsCalculator(ratesData, BBPeriod, BBStandardDeviations)
    ratesData = RelativeStrengthIndexCalculator(ratesData)
    ratesData = OnBalanceVolume(ratesData)
    ratesData = ForwardReturns(ratesData)



    #Prepare data for plot
    ratesData.set_index('time', inplace=True)
    ratesData.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    print(ratesData.columns)
    print(ratesData.head())
    print(ratesData.tail())

    ratesDatalast30 = ratesData.tail(30)
    makePlot(ratesDatalast30)



def MACalculator(data, windowSize, MAPrice, name=None):
    if name is None:
        name = "ma" + str(windowSize)
    data[name] = data[MAPrice].rolling(window=windowSize).mean()
    return data

def BollingerBandsCalculator(data, period, standardDeviations):
    data = MACalculator(data, period, "close", "BBMiddle")
    data["stdDev"] = data["close"].rolling(window=period).std()
    data["BBUpper"] = data["BBMiddle"] + (data["stdDev"] * standardDeviations)
    data["BBLower"] = data["BBMiddle"] - (data["stdDev"] * standardDeviations)
    return data

def RelativeStrengthIndexCalculator(data, period=14):
    data["delta"] = data["close"].diff()
    data["up"] = data["delta"].clip(lower=0)
    data["down"] = -1 * data["delta"].clip(upper=0)
    data["upMean"] = data["up"].rolling(window=period).mean()
    data["downMean"] = data["down"].rolling(window=period).mean()
    data["RS"] = data["upMean"] / data["downMean"]
    data["RSI"] = 100 - 100 / (1 + data["RS"])

    print(data.tail())
    #print(up.tail())
    #print(down.tail())
    return data

def OnBalanceVolume(data):
    data["direction"] = np.sign(data["close"].diff())
    data["OBV"] = (data["tick_volume"]*data["direction"]).cumsum()
    return data

def ForwardReturns(data, period=10):
    data["futureMax"] = data["close"].shift(-period).rolling(window=period, min_periods=1).max()
    data["futureMin"] = data["close"].shift(-period).rolling(window=period, min_periods=1).min()
    data["maxUpp"] = data["futureMax"] - data["close"]
    data["maxDown"] = data["close"] - data["futureMin"]
    data["upp90"] = data["maxUpp"].quantile(0.90)
    data["down90"] = data["maxDown"].quantile(0.90)
    data["significantUpp"] = np.sign(data["maxUpp"]-data["upp90"])
    data["significantDown"] = np.sign(data["maxDown"]-data["down90"])
    data["target"] = 0
    data.loc[data["significantUpp"] > 0, "target"] = 1
    data.loc[data["significantDown"] > 0, "target"] = 2
    return data

def MakeTrainingDataset(data):
    trainingDataset = data[""]
    return trainingDataset

def makePlot(ratesData):
    TIplots = [
        mpf.make_addplot(ratesData["ma30"], color="black", width=0.3),
        mpf.make_addplot(ratesData["BBUpper"], color="cyan", width=0.3),
        mpf.make_addplot(ratesData["BBMiddle"], color="orange", width=0.3),
        mpf.make_addplot(ratesData["BBLower"], color="cyan", width=0.3),
        mpf.make_addplot(ratesData["RSI"], color="purple", width=0.5, secondary_y=True),
        mpf.make_addplot(ratesData["OBV"], panel=1, color="purple", width=0.5,)
    ]

    customPlotStyle = mpf.make_mpf_style(base_mpf_style='charles', facecolor='none', figcolor='none', gridstyle='')
    fig, axlist = mpf.plot(ratesData,
                           type='candle',
                           style=customPlotStyle,
                           addplot=TIplots,
                           figsize=(2.24, 2.24),
                           axisoff=True,
                           scale_padding={'left': 0.1, 'top': 0.1, 'right': 0.1, 'bottom': 0.1},
                           returnfig=True)
    ax = axlist[0]

    y_min, y_max = axlist[0].get_ylim()
    start_pct = np.clip((y_min - 5) / 8, 0, 1)
    stop_pct = np.clip((y_max - 5) / 8, 0, 1)

    res = 224
    grad_array = np.linspace(start_pct, stop_pct, res).reshape(-1, 1)

    grad_2d = np.tile(grad_array, (1, res))

    cmap = LinearSegmentedColormap.from_list('bp', ['blue', 'yellow'])
    rgb_image = cmap(grad_2d)

    fig.figimage(rgb_image, resize=True, origin='lower', zorder=-10)

    current_dpi = fig.get_dpi()
    width_inch, height_inch = fig.get_size_inches()

    print(f"DPI: {current_dpi}")
    print(f"Storlek i tum: {width_inch} x {height_inch}")
    print(f"Faktisk storlek i pixlar: {width_inch * current_dpi} x {height_inch * current_dpi}")

    file_path = "trading_plot.png"

    # Spara figuren
    fig.savefig(file_path)

    print(f"Grafen är sparad som: {file_path}")

    plt.show()
    return ""

if __name__ == '__main__':
    main()
