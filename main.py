from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

def main():
    print("This is the main function")

    #VARIABLES OK TO CHANGE/TEST########################################################################################
    ratesSymbol = "USDSEK"
    ratesTimeFrame = mt5.TIMEFRAME_M1
    dateEnd = datetime(2026, 2, 27) + timedelta(days=1) #Set YYYYMMDD for the last day you want data
    dateStart = dateEnd - timedelta(days=(360))
    trainingDaysRequested = -(20) #Only change value in parentheses
    validationDaysRequested = -(5) #Only change value in parentheses
    trainingFolderName = "train"
    validationFolderName = "val"

    numberOfClasses = 3 #Set to number of classes requested to be generated

    generateDataset = 0 #Set to 1 if you want a dataset generated


    ####################################################################################################################


    #VARIABLES THAT CAN BE CHANGED BUT PROBABLY SHOULD NOT BE CHANGED###################################################
    MAWindowSize = 30
    MAPrice = "close"

    BBPeriod = 20
    BBStandardDeviations = 2

    # atrFactor = 5
    # atrPeriod = 14

    # RSIPeriod = 14

    # significantMovementPeriod = 10

    #timeOfDayStart = "08:30"
    #timeOfDayEnd = "15:00"

    windowSize = 30 #in function: GenerateDataSet
    getNoMovementEvery = 1 #in function: GenerateDataSet3class
    ####################################################################################################################


    #CHART VARIABLES: 0 is not included. 1 is included##################################################################
    #includeMA30 = 0
    #includeBBUpper = 1
    #includeBBMiddle = 1
    #includeBBLower = 1
    #includeOBV = 1
    #includeRSI = 1
    ####################################################################################################################
    print("Main variables set")

    print("Getting data from MT5")
    ratesData = GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd)

    print("MT5 data collected")


    fiveDaysEnable = 0

    #TECHNICAL INDICATORS CALCULATIONS##################################################################################
    ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
    ratesData = ratesData.set_index("time")
    ratesData = ratesData.between_time('08:30', '15:00')
    ratesData = MACalculator(ratesData, MAWindowSize, MAPrice)
    ratesData = BollingerBandsCalculator(ratesData, BBPeriod, BBStandardDeviations)
    ratesData = RelativeStrengthIndexCalculator(ratesData)
    ratesData = OnBalanceVolume(ratesData)
    ratesData = AverageTrueRangeCalculator(ratesData)
    ratesData = ForwardReturns(ratesData)
    uniqueDays = sorted(list(set(ratesData.index.date)))  # All days contained within the dataset
    #Prepares data for chart
    ratesData.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    ####################################################################################################################

    #Generate validation dataset
    requestedValidationDays = uniqueDays[validationDaysRequested:]
    validationRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedValidationDays)]

    # Generate training dataset
    requestedTrainingDays = uniqueDays[((trainingDaysRequested+validationDaysRequested)):validationDaysRequested]
    trainingRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedTrainingDays)]

    if generateDataset == 1:
        GenerateDataSet(validationRatesData, validationFolderName, windowSize, getNoMovementEvery, numberOfClasses)
        GenerateDataSet(trainingRatesData, trainingFolderName, windowSize, getNoMovementEvery, numberOfClasses)


def GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    ratesData = pd.DataFrame(mt5.copy_rates_range(ratesSymbol,ratesTimeFrame,dateStart,dateEnd))

    mt5.shutdown()
    return(ratesData)

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
    return data

def OnBalanceVolume(data):
    data["direction"] = np.sign(data["close"].diff())
    data["OBV"] = (data["tick_volume"]*data["direction"]).cumsum()
    return data

def ForwardReturns(data, period=10, atrFactor=5):
    data["futureMax"] = data["close"].shift(-period).rolling(window=period, min_periods=1).max()
    data["futureMin"] = data["close"].shift(-period).rolling(window=period, min_periods=1).min()
    data["maxUpp"] = data["futureMax"] - data["close"]
    data["maxDown"] = data["close"] - data["futureMin"]
    data["significantUpp"] = np.sign(data["maxUpp"] - (data["averageTrueRange"]*atrFactor))
    data["significantDown"] = np.sign(data["maxDown"] - (data["averageTrueRange"]*atrFactor))
    data["target"] = 0
    maskUpp = data["significantUpp"] > 0
    maskDown = data["significantDown"] > 0
    mask_conflict = (data["significantUpp"] > 0) & (data["significantDown"] > 0)
    data.loc[maskUpp, "target"] = 1
    data.loc[maskDown, "target"] = 2
    data["conflict"] = 0
    data.loc[mask_conflict, "conflict"] = 1
    print(f"Total conflict points found: {data['conflict'].sum()}")

    return data

def AverageTrueRangeCalculator (data, period=14):
    data["prevClose"] = data["close"].shift(1)
    data["tr1"] = data["high"] - data["low"]
    data["tr2"] = (data["high"] - data["prevClose"]).abs()
    data["tr3"] = (data["low"] - data["prevClose"]).abs()
    data["trueRange"] = data[["tr1", "tr2", "tr3"]].max(axis=1)

    data["averageTrueRange"] = data["trueRange"].rolling(window=period).mean()
    return data



def makePlotForReport(chartData):
    print("HELLO")
    TIplots = [
        mpf.make_addplot(chartData["ma30"], color="black", width=0.3),
        mpf.make_addplot(chartData["BBUpper"], color="cyan", width=0.3),
        mpf.make_addplot(chartData["BBMiddle"], color="orange", width=0.3),
        mpf.make_addplot(chartData["BBLower"], color="cyan", width=0.3),
        mpf.make_addplot(chartData["OBV"], panel=1, color="purple", width=0.5),
        mpf.make_addplot(chartData["RSI"], panel=2, color="purple", width=0.5)
    ]

    #customPlotStyle = mpf.make_mpf_style(base_mpf_style='charles', facecolor='none', figcolor='none', gridstyle='')
    fig, axlist = mpf.plot(chartData,
                           type='candle',
                           style='charles',
                           addplot=TIplots,
                           figsize=(12, 8),
                           returnfig=True)

    # Manually set y-axis labels for each panel
    axlist[0].set_ylabel("Price + MA + BB")
    axlist[2].set_ylabel("RSI                  OBV", labelpad=10)  # main panel
    axlist[2].yaxis.set_label_coords(1.06, -0.05)

    file_path_pdf = "trading_plot.pdf"
    fig.savefig(file_path_pdf, format='pdf', bbox_inches='tight')  # PDF

    print(f"Grafen är sparad som: {file_path_pdf}")

    plt.show()
    return ""

def makePlot(ratesData, saveFolderName):
    # 1. Day-Change Guard
    if ratesData.index[0].date() != ratesData.index[-1].date():
        print("Skipping: Day change detected.")
        return False

    # 2. Connection/Gap Guard
    # Calculate the difference between consecutive timestamps
    time_deltas = ratesData.index.to_series().diff().dropna()

    # Check if any gap is NOT equal to 1 minute
    if not (time_deltas == pd.Timedelta(minutes=1)).all():
        print("Skipping: Gap in minute data detected.")
        return False
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
    start_pct = np.clip((y_min - 5) / (13 - 5), 0, 1) - 0.01
    stop_pct = np.clip((y_max - 5) / (13 - 5), 0, 1) + 0.01

    res = 224
    grad_array = np.linspace(start_pct, stop_pct, res).reshape(-1, 1)

    grad_2d = np.tile(grad_array, (1, res))

    cmap = LinearSegmentedColormap.from_list('bp', ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green'])
    rgb_image = cmap(grad_2d)

    #fig.figimage(rgb_image, resize=True, origin='lower', zorder=-10)

    current_dpi = fig.get_dpi()
    width_inch, height_inch = fig.get_size_inches()

    print(f"DPI: {current_dpi}")
    print(f"Storlek i tum: {width_inch} x {height_inch}")
    print(f"Faktisk storlek i pixlar: {width_inch * current_dpi} x {height_inch * current_dpi}")

    if saveFolderName and not os.path.exists(saveFolderName):
        os.makedirs(saveFolderName)
    #Spara på specificerad mapp
    file_path = os.path.join(saveFolderName, f"{ratesData.index[0].strftime('%Y-%m-%d_%H%M')}.png")

    # Spara figuren
    fig.savefig(file_path)

    print(f"Grafen är sparad som: {file_path}")

    plt.show()
    plt.close()
    return ""

def GenerateDataSet(ratesData, saveFolderName, window_size = 30, getNoMovementEvery = 1, numberOfClasses = 3):
    saveFolderName=os.path.join("dataset", saveFolderName)
    total_rows = len(ratesData)
    noMovementCounter = 0
    if numberOfClasses == 2:
        folders = ["upMovement", "downMovement"]
    elif numberOfClasses == 3:
        folders = ["upMovement", "downMovement", "noMovement"]
    else:
        print("Wrong number of classes.")
        return ""

    # Pre-create all paths
    for folder in folders:
        os.makedirs(os.path.join(saveFolderName, folder), exist_ok=True)



    #We start the loop at 'window_size' so the first slice [0:30] is valid
    for i in range(window_size, total_rows + 1):
        # Slice from (current position - window) to (current position)
        subset = ratesData.iloc[i - window_size: i]

        #Check what movement is valid and send to plotting function
        current_target = subset["target"].iloc[-1]
        if (subset["conflict"] == 0).all(): #ratesData = ratesData[ratesData["conflict"] == 0]
            if current_target == 1:
                makePlot(subset, os.path.join(saveFolderName, "upMovement"))
            elif current_target == 2:
                makePlot(subset, os.path.join(saveFolderName, "downMovement"))
            elif current_target == 0:  # Explicitly check for noMovement
                # Only plot if the counter hits the interval
                if noMovementCounter % getNoMovementEvery == 0:
                    makePlot(subset, os.path.join(saveFolderName, "noMovement"))

    return ""


if __name__ == '__main__':
    main()

#CODE BELLOW HERE IS OUTDATED AND NOT USED##############################################################################

def makePlot_OUTDATED(ratesData):
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
    start_pct = np.clip((y_min - 5) / (13 - 5), 0, 1)
    stop_pct = np.clip((y_max - 5) / (13 - 5), 0, 1)

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