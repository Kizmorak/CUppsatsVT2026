from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap

inputDays = 72
dateEnd = datetime(2026, 1, 31) + timedelta(days=1-38) #Set YYYYMMDD for the last day you want data
dateStart = dateEnd - timedelta(days=(inputDays)) #Can be set to a larger number than needed
trainingDaysRequested = -(72) #Only change value in parentheses
validationDaysRequested = -(18) #Only change value in parentheses
threshold_estimationDaysRequested = -(10) #Only change value in parentheses
backtestDaysRequested = -(10) #Only change value in parentheses

timeOfDayStart = "08:30"
timeOfDayEnd = "15:00"

#Dataset folder structure naming
trainingFolderName = "train"
validationFolderName = "val"
backtestFolderName = "backtesting"
thresholdFolderName = "threshold_estimation"

#Set how many classes you want to generate graphs for in the different datasets
numberOfClassesTraining = 2
numberOfClassesValidation = 2
numberOfClassesBacktesting = 3
numberOfClassesthreshold_estimation = 3

#List of dataset names, use 1 or choose your own name
#datasetName = "all_TIs"
#datasetName = "No_TIs"
#datasetName = "No_BB"
#datasetName = "No_BB_No_RSI"
#datasetName = "No_BB_No_OBV"
#datasetName = "No_RSI"
#datasetName = "No_RSI_No_OBV"
#datasetName = "No_OBV"
datasetName = "No_TI"

#CHART VARIABLES: 0 is not included. 1 is included
includeMA30 = 0
includeBB = 0
includeOBV = 0
includeRSI = 0

atrFactor = 1
significantMovementPeriod = 4

#Choose what files to create
generateDataset = 0 #Set to 1 if you want dataset graphs generated
createCSV = 1 #Set to 1 if you want a CSV of the complete dataset data
####################################################################################################################


#VARIABLES THAT CAN BE CHANGED BUT PROBABLY SHOULD NOT BE CHANGED###################################################
ratesSymbol = "XAUUSD"
ratesTimeFrame = mt5.TIMEFRAME_M5

includeTIs = 1

MAWindowSize = 15
MAPrice = "close"

BBPeriod = 20
BBStandardDeviations = 2


atrPeriod = 14

RSIPeriod = 14



windowSize = 10 #in function: GenerateDataSet
getNoMovementEvery = 1 #in function: GenerateDataSet
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
    print(f"Total upp movement: {(ratesData['target'] == 1).sum()}")
    print(f"Total down movement: {(ratesData['target'] == 2).sum()}")
    print(f"Total data rows: {len(ratesData)}")
    print(f"Total decisions rows: {((ratesData['target'] == 2).sum())+((ratesData['target'] == 1).sum())}")
    print(f"Decitions per day: {(((ratesData['target'] == 2).sum())+((ratesData['target'] == 1).sum()))/(inputDays)}")
    ####################################################################################################################

    #thresholdFolderName = "threshold_estimation"


    # Generate Backtesting dataset
    requestedBacktestingDays = uniqueDays[backtestDaysRequested:]
    backtestRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedBacktestingDays)]

    # Generate threshold_estimation dataset
    requestedthreshold_estimationDays = uniqueDays[(threshold_estimationDaysRequested+backtestDaysRequested):backtestDaysRequested]
    threshold_estimationRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedthreshold_estimationDays)]

    #Generate validation dataset
    requestedValidationDays = uniqueDays[((threshold_estimationDaysRequested+backtestDaysRequested+validationDaysRequested)):(threshold_estimationDaysRequested+backtestDaysRequested)]
    validationRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedValidationDays)]

    # Generate training dataset
    requestedTrainingDays = uniqueDays[((threshold_estimationDaysRequested+backtestDaysRequested+trainingDaysRequested+validationDaysRequested)):(threshold_estimationDaysRequested+backtestDaysRequested+validationDaysRequested)]
    trainingRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedTrainingDays)]

    if generateDataset == 1:
        if os.path.exists("datasetNew"):
            shutil.rmtree("datasetNew")
            print(f"Deleted existing folder: {"datasetNew"}")
            GenerateDataSet(trainingRatesData, trainingFolderName, windowSize, getNoMovementEvery,
                            numberOfClassesTraining, datasetName, includeRSI, includeOBV, includeBB, atrFactor,
                            significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                            backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
            GenerateDataSet(validationRatesData, validationFolderName, windowSize, getNoMovementEvery,
                            numberOfClassesValidation, datasetName, includeRSI, includeOBV, includeBB, atrFactor,
                            significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                            backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
            GenerateDataSet(threshold_estimationRatesData, thresholdFolderName, windowSize, getNoMovementEvery,
                            numberOfClassesthreshold_estimation, datasetName, includeRSI, includeOBV, includeBB,
                            atrFactor, significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                            backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
            GenerateDataSet(backtestRatesData, backtestFolderName, windowSize, getNoMovementEvery,
                            numberOfClassesBacktesting, datasetName, includeRSI, includeOBV, includeBB, atrFactor,
                            significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                            backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)


    if createCSV == 1:
        createCSVFromDataset(ratesData, "fullDataset")
       # createCSVFromDataset(trainingRatesData, "trainingDataset")
       # createCSVFromDataset(validationRatesData, "validationDataset")
        #ratesDataBacktest = backtestRatesData[['target', 'conflict']].copy()
        #ratesDataBacktest.loc[ratesDataBacktest['conflict'] == True, 'target'] = 0
        #ratesDataBacktest = ratesDataBacktest.drop(columns=['conflict'])
        #ratesDataBacktest = ratesDataBacktest[ratesDataBacktest['target'] != 0]
        #ratesDataBacktest.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
        #ratesDataBacktest["time"] = datetime.timestamp(ratesDataBacktest["time"])

        #createCSVFromDataset(ratesDataBacktest, "backtestDataset.csv")


#Starts MT5 and gets rates data according to configuration of input to function
#Returns an array of the rates
def GetDataFromMT5(symbol, timeFrame, start, end):
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe",
               login=5048695455, password="_8UkDpIy", server="MetaQuotes-Demo"):
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




def makePlot(ratesData, saveFolderName, includeRSI, includeOBV, includeBB, ratesTimeFrame):
    #Day-Change Guard
    if ratesData.index[0].date() != ratesData.index[-1].date():
        print("Skipping: Day change detected.")
        return False

    #Connection/Gap Guard
    #Calculate the difference between consecutive timestamps
    time_deltas = ratesData.index.to_series().diff().dropna()

    # Check if any gap is NOT equal to X minute
    if(ratesTimeFrame == mt5.TIMEFRAME_M5):
        if not (time_deltas == pd.Timedelta(minutes=5)).all():
            print("Skipping: Gap in minute data detected.")
            return False
    elif (ratesTimeFrame == mt5.TIMEFRAME_M3):
        if not (time_deltas == pd.Timedelta(minutes=3)).all():
            print("Skipping: Gap in minute data detected.")
            return False
    elif(ratesTimeFrame == mt5.TIMEFRAME_M1):
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

def GenerateDataSet(ratesData, saveFolderName, window, getNoMovementEvery, numberOfClasses, datasetName, includeRSI,
                    includeOBV, includeBB, atrFactor, significantMovementPeriod, trainingDaysRequested,
                    validationDaysRequested, backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame):
    # Create variables used by function
   # dateEnd = datetime(2026, 3, 20) + timedelta(days=1)
    #dataset_No_TIs_T80V20B5_20260320_2classTV_3classB
    dateStr = (dateEnd - timedelta(days=1)).strftime('%Y%m%d')
    folderPrefix = (f"{datasetName}_{-trainingDaysRequested}_{-validationDaysRequested}_{-backtestDaysRequested}"
                    f"_{-threshold_estimationDaysRequested}_{dateStr}_"
                    f"{window}_{atrFactor}_{significantMovementPeriod}")
    saveFolderName = os.path.join(folderPrefix, saveFolderName)
    total_rows = len(ratesData)
    noMovementCounter = 0

    #Check number of classes configured
    if numberOfClasses == 2:
        folders = ["upMovement", "downMovement"]
    elif numberOfClasses == 3:
        folders = ["upMovement", "downMovement", "noMovement"]
    elif "backtesting" in saveFolderName.lower():
        folders = []
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
        #if (subset["conflict"] == 0).all() and "backtesting" not in saveFolderName.lower():
        if (subset["conflict"] == 0).all():
            if current_target == 1:
                makePlot(subset, os.path.join(saveFolderName, "upMovement"), includeRSI, includeOBV, includeBB,ratesTimeFrame)
            elif current_target == 2:
                makePlot(subset, os.path.join(saveFolderName, "downMovement"), includeRSI, includeOBV, includeBB,ratesTimeFrame)
            elif current_target == 0:
                if numberOfClasses == 3:
                    if noMovementCounter % getNoMovementEvery == 0:
                        makePlot(subset, os.path.join(saveFolderName, "noMovement"), includeRSI, includeOBV, includeBB,ratesTimeFrame)
                    noMovementCounter += 1
        #elif (subset["conflict"] == 0).all():
        #    makePlot(subset, os.path.join(saveFolderName), includeRSI, includeOBV, includeBB)
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