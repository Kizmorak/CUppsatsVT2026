from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap

#VARIABLES OK TO CHANGE/TEST########################################################################################
ratesSymbol = "USDSEK"
ratesTimeFrame = mt5.TIMEFRAME_M1
dateEnd = datetime(2026, 3, 13) + timedelta(days=1) #Set YYYYMMDD for the last day you want data
dateStart = dateEnd - timedelta(days=(360))
trainingDaysRequested = -(80) #Only change value in parentheses
validationDaysRequested = -(20) #Only change value in parentheses
trainingFolderName = "train"
validationFolderName = "val"


numberOfClasses = 2 #Set to number of classes requested to be generated
generateDataset = 1 #Set to 1 if you want a dataset generated

createCSV = 1 #Set to 1 if you want a CSV of the complete dataset
####################################################################################################################


#VARIABLES THAT CAN BE CHANGED BUT PROBABLY SHOULD NOT BE CHANGED###################################################
MAWindowSize = 30
MAPrice = "close"

BBPeriod = 20
BBStandardDeviations = 2

atrFactor = 5
atrPeriod = 14

RSIPeriod = 14

significantMovementPeriod = 10

timeOfDayStart = "08:30"
timeOfDayEnd = "15:00"

windowSize = 30 #in function: GenerateDataSet
getNoMovementEvery = 1 #in function: GenerateDataSet
####################################################################################################################


#CHART VARIABLES: 0 is not included. 1 is included##################################################################
includeTIs = 1

includeMA30 = 0
includeBB = 1
includeOBV = 1
includeRSI = 1
####################################################################################################################

def main():
    print("This is the main function")


    print("Main variables set")

    print("Getting data from MT5")
    ratesData = GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd)
    print("MT5 data collected")

    print(ratesData.head())

    #TECHNICAL INDICATORS CALCULATIONS##################################################################################
    ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
    ratesData = ratesData.set_index("time")
    #ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
    ratesData = MACalculator(ratesData, MAWindowSize, MAPrice)
    ratesData = BollingerBandsCalculator(ratesData, BBPeriod, BBStandardDeviations)
    ratesData = RelativeStrengthIndexCalculator(ratesData, RSIPeriod)
    ratesData = OnBalanceVolume(ratesData)
    ratesData = AverageTrueRangeCalculator(ratesData, atrPeriod)
    ratesData = ForwardReturns(ratesData, significantMovementPeriod, atrFactor)
    uniqueDays = sorted(list(set(ratesData.index.date)))  # All days contained within the dataset
    ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
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
        if os.path.exists("datasetNew"):
            shutil.rmtree("datasetNew")
            print(f"Deleted existing folder: {"datasetNew"}")
        GenerateDataSet(validationRatesData, validationFolderName, windowSize, getNoMovementEvery, numberOfClasses)
        GenerateDataSet(trainingRatesData, trainingFolderName, windowSize, getNoMovementEvery, numberOfClasses)

    if createCSV == 1:
        createCSVFromDataset(ratesData, "fullDataset")
        createCSVFromDataset(trainingRatesData, "trainingDataset")
        createCSVFromDataset(validationRatesData, "validationDataset")

#Starts MT5 and gets rates data according to configuration of input to function
#Returns an array of the rates
def GetDataFromMT5(symbol, timeFrame, start, end):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    ratesData = pd.DataFrame(mt5.copy_rates_range(symbol,timeFrame,start,end))
    print(ratesData.columns)

    mt5.shutdown()
    return(ratesData)

#Takes an array of rates
#Calculates a Moving Average according to configuration of input to function.
#Returns the array with moving average column attached for all points.
#MA column name can be chosen if name is part of input to function.
#MA standar name is based on number of points used to calculate MA.
def MACalculator(data, period, MAPrice, name=None):
    if name is None:
        name = "ma" + str(period)
    data[name] = data[MAPrice].rolling(window=period).mean()
    return data

#Takes an array of rates
#Calculates Bollinger Bands according to configuration of input to function
#Attaches all calculated results the function needs to the array of rates (STDDev)
#Returns the array with Bollinger Bands (BBUpper, BBMiddle, BBLower) column attached for all points
def BollingerBandsCalculator(data, period, standardDeviations):
    data = MACalculator(data, period, "close", "BBMiddle")
    data["stdDev"] = data["close"].rolling(window=period).std()
    data["BBUpper"] = data["BBMiddle"] + (data["stdDev"] * standardDeviations)
    data["BBLower"] = data["BBMiddle"] - (data["stdDev"] * standardDeviations)
    return data

#Takes an array of rates
#Calculates Relative Strength Index according to configuration of input to function
#Attaches all calculated results the function needs to the array of rates (delta, up, down, upmean, downmean, RS)
#Returns the array with Relative Strength Index (RSI) column attached for all points
def RelativeStrengthIndexCalculator(data, period=14):
    data["delta"] = data["close"].diff()
    data["up"] = data["delta"].clip(lower=0)
    data["down"] = -1 * data["delta"].clip(upper=0)
    data["upMean"] = data["up"].rolling(window=period).mean()
    data["downMean"] = data["down"].rolling(window=period).mean()
    data["RS"] = data["upMean"] / data["downMean"]
    data["RSI"] = 100 - 100 / (1 + data["RS"])
    return data


#Takes an array of rates
#Calculates On-Balance Volume according to configuration of input to function
#Attaches all calculated results the function needs to the array of rates (direction)
#Returns the array with On-Balance Volume (OBV) column attached for all points
def OnBalanceVolume(data):
    data["direction"] = np.sign(data["close"].diff())
    data["OBV"] = (data["tick_volume"]*data["direction"]).cumsum()
    return data

#Takes an array of rates
#Calculates if there are possible forward returns according to configuration of input to function
#Attaches all calculated results the function needs to the array of rates (futureMax, futureMin, maxUp, maxDown,
# significantUp, significantDown, conflict)
#Returns the array with target (0=no movement, 1=up movement 2=down movement) column attached for all points
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

#Takes an array of rates
#Calculates Average True Range according to configuration of input to function
#Attaches all calculated results the function needs to the array of rates (prevClose, tr1, tr2, tr3, trueRange)
#Returns the array with Average True Range (averageTrueRange) column attached for all points
def AverageTrueRangeCalculator (data, period=14):
    data["prevClose"] = data["close"].shift(1)
    data["tr1"] = data["high"] - data["low"]
    data["tr2"] = (data["high"] - data["prevClose"]).abs()
    data["tr3"] = (data["low"] - data["prevClose"]).abs()
    data["trueRange"] = data[["tr1", "tr2", "tr3"]].max(axis=1)

    data["averageTrueRange"] = data["trueRange"].rolling(window=period).mean()
    return data




def makePlot(ratesData, saveFolderName):
    #Day-Change Guard
    if ratesData.index[0].date() != ratesData.index[-1].date():
        print("Skipping: Day change detected.")
        return False

    #Connection/Gap Guard
    #Calculate the difference between consecutive timestamps
    time_deltas = ratesData.index.to_series().diff().dropna()

    # Check if any gap is NOT equal to 1 minute
    if not (time_deltas == pd.Timedelta(minutes=1)).all():
        print("Skipping: Gap in minute data detected.")
        return False

    #TI inclusion based on configuration
    if includeTIs:
        plotsConfig = []
        if includeMA30 == 1:
            plotsConfig.append({"col": "ma30", "color": "black", "width": 0.3})
        if includeBB == 1:
            plotsConfig.append({"col": "BBUpper", "color": "cyan", "width": 0.3})
            plotsConfig.append({"col": "BBMiddle", "color": "orange", "width": 0.3})
            plotsConfig.append({"col": "BBLower", "color": "cyan", "width": 0.3})
        if includeRSI == 1:
            plotsConfig.append({"col": "RSI", "color": "purple", "width": 0.5, "secondary_y": True})
        if includeOBV == 1:
            plotsConfig.append({"col": "OBV", "color": "blue", "width": 0.5, "panel": 1})
        TIplots = [
            mpf.make_addplot(ratesData[item["col"]], **{k: v for k, v in item.items() if k != "col"})
            for item in plotsConfig
        ]

    #Generate base plot
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

    # Set plot resolution
    res = 224

    #Calculate gradient based on plot values
    y_min, y_max = axlist[0].get_ylim()
    start_pct = np.clip((y_min - 5) / (13 - 5), 0, 1) - 0.01
    stop_pct = np.clip((y_max - 5) / (13 - 5), 0, 1) + 0.01
    grad_array = np.linspace(start_pct, stop_pct, res).reshape(-1, 1)
    grad_2d = np.tile(grad_array, (1, res))
    cmap = LinearSegmentedColormap.from_list('bp', ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green'])
    rgb_image = cmap(grad_2d)

    #Add gradient to plot
    #fig.figimage(rgb_image, resize=True, origin='lower', zorder=-10)

    #Create save folder if not available
    if saveFolderName and not os.path.exists(saveFolderName):
        os.makedirs(saveFolderName)
    #Specify folder to be saved in
    file_path = os.path.join(saveFolderName, f"{ratesData.index[0].strftime('%Y-%m-%d_%H%M')}.png")

    #Save figure
    fig.savefig(file_path)

    print(f"Grafen är sparad som: {file_path}")

    #Show graph
    #plt.show()
    #Close graph
    plt.close()
    return ""

def GenerateDataSet(ratesData, saveFolderName, window = 30, getNoMovementEvery = 1, numberOfClasses = 3):
    # Create variables used by function
    saveFolderName=os.path.join("datasetNew", saveFolderName)
    total_rows = len(ratesData)
    noMovementCounter = 0

    #Check number of classes configured
    if numberOfClasses == 2:
        folders = ["upMovement", "downMovement"]
    elif numberOfClasses == 3:
        folders = ["upMovement", "downMovement", "noMovement"]
    else:
        print("Wrong number of classes.")
        return ""

    # Pre-create all paths if not available
    for folder in folders:
        os.makedirs(os.path.join(saveFolderName, folder), exist_ok=True)



    #We start the loop at "window" so the first slice is valid
    for i in range(window, total_rows + 1):
        # Slice from (current position - window) to (current position)
        subset = ratesData.iloc[i - window: i]

        #Check what movement is valid and send to plotting function
        current_target = subset["target"].iloc[-1]
        if (subset["conflict"] == 0).all():
            if current_target == 1:
                makePlot(subset, os.path.join(saveFolderName, "upMovement"))
            elif current_target == 2:
                makePlot(subset, os.path.join(saveFolderName, "downMovement"))
            elif current_target == 0:
                if numberOfClasses == 3:
                    if noMovementCounter % getNoMovementEvery == 0:
                        makePlot(subset, os.path.join(saveFolderName, "noMovement"))
                    noMovementCounter += 1

    return ""

def createCSVFromDataset(dataset, filename="output.csv"):
    """
    Saves a pandas DataFrame to a CSV file.
    """
    # index=False avoids saving the row numbers (0, 1, 2...) as a column
    dataset.to_csv(filename, encoding='utf-8')
    print(f"File saved successfully as {filename}")

if __name__ == '__main__':
    main()

#NON-CRITICAL FUNCTIONS FOR DATA COLLECTION#############################################################################

#Function makes a graph of the input array.
#Only used to make example plots with guides for reports.
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
########################################################################################################################

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