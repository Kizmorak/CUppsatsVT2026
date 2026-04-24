import dataCollection

from datetime import datetime, timedelta
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import pandas as pd
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap

import multiprocessing

from dataCollection import numberOfClassesTraining, numberOfClassesValidation, includeBB, includeOBV, includeRSI, \
    ratesSymbol, ratesTimeFrame


def main(dateEnd, dateStart, numberOfClassesTraining, numberOfClassesValidation, numberOfClassesBacktesting,
         numberOfClassesthreshold_estimation, datasetName, includeBB, includeOBV, includeRSI, ratesSymbol,
         ratesTimeFrame, MAWindowSize, MAPrice, BBPeriod, BBStandardDeviations, atrPeriod, atrFactor, RSIPeriod,
         significantMovementPeriod, windowSize, getNoMovementEvery, timeOfDayStart, timeOfDayEnd, backtestDaysRequested,
         threshold_estimationDaysRequested, validationDaysRequested, trainingDaysRequested, thresholdFolderName,
         trainingFolderName, validationFolderName, backtestFolderName, createCSV):

    print("This is the main function")
    generateStartTime = datetime.now()
    print("Main variables set")

    print("Getting data from MT5")
    ratesData = dataCollection.GetDataFromMT5(ratesSymbol, ratesTimeFrame, dateStart, dateEnd)
    print("MT5 data collected")

    print(ratesData.head())

    # TECHNICAL INDICATORS CALCULATIONS##################################################################################
    ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
    ratesData = ratesData.set_index("time")
    # ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
    ratesData = dataCollection.MACalculator(ratesData, MAWindowSize, MAPrice)
    ratesData = dataCollection.BollingerBandsCalculator(ratesData, BBPeriod, BBStandardDeviations)
    ratesData = dataCollection.RelativeStrengthIndexCalculator(ratesData, RSIPeriod)
    ratesData = dataCollection.OnBalanceVolume(ratesData)
    ratesData = dataCollection.AverageTrueRangeCalculator(ratesData, atrPeriod)
    ratesData = dataCollection.ForwardReturns(ratesData, significantMovementPeriod, atrFactor)
    uniqueDays = sorted(list(set(ratesData.index.date)))  # All days contained within the dataset
    ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
    # Prepares data for chart
    ratesData.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    print(f"Total upp movement: {(ratesData['target'] == 1).sum()}")
    print(f"Total down movement: {(ratesData['target'] == 2).sum()}")
    print(f"Total data rows: {len(ratesData)}")
    print(f"Total decisions rows: {((ratesData['target'] == 2).sum()) + ((ratesData['target'] == 1).sum())}")
    #print(f"Decitions per day: {(((ratesData['target'] == 2).sum()) + ((ratesData['target'] == 1).sum())) / (inputDays)}")
    ####################################################################################################################

    # Generate Backtesting dataset
    requestedBacktestingDays = uniqueDays[backtestDaysRequested:]
    backtestRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedBacktestingDays)]

    # Generate threshold_estimation dataset
    requestedthreshold_estimationDays = uniqueDays[
        (threshold_estimationDaysRequested + backtestDaysRequested):backtestDaysRequested]
    threshold_estimationRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedthreshold_estimationDays)]

    # Generate validation dataset
    requestedValidationDays = uniqueDays[
        ((threshold_estimationDaysRequested + backtestDaysRequested + validationDaysRequested)):(
                    threshold_estimationDaysRequested + backtestDaysRequested)]
    validationRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedValidationDays)]

    # Generate training dataset
    requestedTrainingDays = uniqueDays[((
                threshold_estimationDaysRequested + backtestDaysRequested + trainingDaysRequested + validationDaysRequested)):(
                                                   threshold_estimationDaysRequested + backtestDaysRequested + validationDaysRequested)]
    trainingRatesData = ratesData[pd.Index(ratesData.index.date).isin(requestedTrainingDays)]

    if dataCollection.generateDataset == 1:
        if os.path.exists("datasetNew"):
            shutil.rmtree("datasetNew")
            print(f"Deleted existing folder: {"datasetNew"}")

        dataCollection.GenerateDataSet(trainingRatesData, trainingFolderName, windowSize, getNoMovementEvery,
                                       numberOfClassesTraining, datasetName, includeRSI, includeOBV, includeBB,
                                       atrFactor,
                                       significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                                       backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
        dataCollection.GenerateDataSet(validationRatesData, validationFolderName, windowSize, getNoMovementEvery,
                                       numberOfClassesValidation, datasetName, includeRSI, includeOBV, includeBB,
                                       atrFactor,
                                       significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                                       backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
        dataCollection.GenerateDataSet(threshold_estimationRatesData, thresholdFolderName, windowSize, getNoMovementEvery,
                        numberOfClassesthreshold_estimation, datasetName, includeRSI, includeOBV, includeBB, atrFactor,
                                       significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                                       backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)
        dataCollection.GenerateDataSet(backtestRatesData, backtestFolderName, windowSize, getNoMovementEvery,
                        numberOfClassesBacktesting, datasetName, includeRSI, includeOBV, includeBB, atrFactor,
                                       significantMovementPeriod, trainingDaysRequested, validationDaysRequested,
                                       backtestDaysRequested, threshold_estimationDaysRequested, ratesTimeFrame)



    if createCSV == 1:
        dataCollection.createCSVFromDataset(ratesData, "fullDataset")
    # createCSVFromDataset(trainingRatesData, "trainingDataset")
    # createCSVFromDataset(validationRatesData, "validationDataset")
    # ratesDataBacktest = backtestRatesData[['target', 'conflict']].copy()
    # ratesDataBacktest.loc[ratesDataBacktest['conflict'] == True, 'target'] = 0
    # ratesDataBacktest = ratesDataBacktest.drop(columns=['conflict'])
    # ratesDataBacktest = ratesDataBacktest[ratesDataBacktest['target'] != 0]
    # ratesDataBacktest.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
    # ratesDataBacktest["time"] = datetime.timestamp(ratesDataBacktest["time"])

    # createCSVFromDataset(ratesDataBacktest, "backtestDataset.csv")
    generateEndTime = datetime.now()
    print(f"Dataset generation started at: {generateStartTime}")
    print(f"Dataset generation ended at: {generateEndTime}")
    print(f"Total time: {generateEndTime - generateStartTime}")
    return "This is the main function"

if __name__ == '__main__':
    # Define your account configurations
    print("Configure datasets")
    configs = [
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(150)),
            "trainingDaysRequested": -(104),  # Only change value in parentheses
            "validationDaysRequested": -(26),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "OBV",

            "includeBB": 0,
            "includeOBV": 1,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M5,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 4,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(115)),
            "trainingDaysRequested": -(76),  # Only change value in parentheses
            "validationDaysRequested": -(19),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "OBV",

            "includeBB": 0,
            "includeOBV": 1,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 3,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(145)),
            "trainingDaysRequested": -(100),  # Only change value in parentheses
            "validationDaysRequested": -(25),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "OBV",

            "includeBB": 0,
            "includeOBV": 1,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(105)),
            "trainingDaysRequested": -(68),  # Only change value in parentheses
            "validationDaysRequested": -(17),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "OBV",

            "includeBB": 0,
            "includeOBV": 1,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M1,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1.5,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(150)),
            "trainingDaysRequested": -(104),  # Only change value in parentheses
            "validationDaysRequested": -(26),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "BB",

            "includeBB": 1,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M5,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 4,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(115)),
            "trainingDaysRequested": -(76),  # Only change value in parentheses
            "validationDaysRequested": -(19),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "BB",

            "includeBB": 1,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 3,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(145)),
            "trainingDaysRequested": -(100),  # Only change value in parentheses
            "validationDaysRequested": -(25),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "BB",

            "includeBB": 1,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(105)),
            "trainingDaysRequested": -(68),  # Only change value in parentheses
            "validationDaysRequested": -(17),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "BB",

            "includeBB": 1,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M1,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1.5,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(150)),
            "trainingDaysRequested": -(104),  # Only change value in parentheses
            "validationDaysRequested": -(26),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "RSI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 1,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M5,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 4,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(115)),
            "trainingDaysRequested": -(76),  # Only change value in parentheses
            "validationDaysRequested": -(19),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "RSI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 1,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 3,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(145)),
            "trainingDaysRequested": -(100),  # Only change value in parentheses
            "validationDaysRequested": -(25),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "RSI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 1,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(105)),
            "trainingDaysRequested": -(68),  # Only change value in parentheses
            "validationDaysRequested": -(17),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "RSI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 1,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M1,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1.5,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        }
    ]

    # List of dataset names, use 1 or choose your own name
    # datasetName = "all_TIs"
    # datasetName = "No_TIs"
    # datasetName = "No_BB"
    # datasetName = "No_BB_No_RSI"
    # datasetName = "No_BB_No_OBV"
    # datasetName = "No_RSI"
    # datasetName = "No_RSI_No_OBV"
    # datasetName = "No_OBV"

    # 1. Initialize the Pool with a limit of 8
    with multiprocessing.Pool(processes=4) as pool:

        # 2. Use apply_async because your main() function uses multiple keyword arguments.
        # This will send all configs to the pool, but only 8 will run at a time.
        tasks = [pool.apply_async(main, kwds=config) for config in configs]

        # 3. Optional: Wait for all tasks to complete and get their return values
        for task in tasks:
            try:
                result = task.get()
                print(f"Task completed: {result}")
            except Exception as e:
                print(f"A task failed with error: {e}")

    print("All datasets have been generated.")

#    processes = []
#    # Launch a process for each config
#    for config in configs:
#       print("Launching dataset generation")
#        p = multiprocessing.Process(target=main, kwargs=config)
#        p.start()
#        processes.append(p)
#
#    # Keep the main script alive while workers run
#    for p in processes:
#        p.join()


"""
{
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(115)),
            "trainingDaysRequested": -(76),  # Only change value in parentheses
            "validationDaysRequested": -(19),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "No_TI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 3,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(105)),
            "trainingDaysRequested": -(68),  # Only change value in parentheses
            "validationDaysRequested": -(17),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "No_TI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M1,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1.5,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 2,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(150)),
            "trainingDaysRequested": -(104),  # Only change value in parentheses
            "validationDaysRequested": -(26),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses

            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",

            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",

            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,

            "datasetName": "No_TI",

            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 0,

            "createCSV": 1,

            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M5,

            "MAWindowSize": 15,
            "MAPrice": "close",

            "BBPeriod": 20,
            "BBStandardDeviations": 2,

            "atrFactor": 1,
            "atrPeriod": 14,

            "RSIPeriod": 14,

            "significantMovementPeriod": 4,

            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        },
        {
            "dateEnd": datetime(2026, 1, 31) + timedelta(days=1),
            "dateStart": datetime(2026, 1, 31) + timedelta(days=1) - timedelta(days=(145)),
            "trainingDaysRequested": -(100),  # Only change value in parentheses
            "validationDaysRequested": -(25),  # Only change value in parentheses
            "threshold_estimationDaysRequested": -(10),  # Only change value in parentheses
            "backtestDaysRequested": -(10),  # Only change value in parentheses
    
            "timeOfDayStart": "08:30",
            "timeOfDayEnd": "15:00",
    
            # Dataset folder structure naming
            "trainingFolderName": "train",
            "validationFolderName": "val",
            "backtestFolderName": "backtesting",
            "thresholdFolderName": "threshold_estimation",
    
            "numberOfClassesTraining": 2,
            "numberOfClassesValidation": 2,
            "numberOfClassesBacktesting": 3,
            "numberOfClassesthreshold_estimation": 3,
    
            "datasetName": "No_TI",
    
            "includeBB": 0,
            "includeOBV": 0,
            "includeRSI": 0,
    
            "createCSV": 1,
    
            "ratesSymbol": "XAUUSD",
            "ratesTimeFrame": mt5.TIMEFRAME_M3,
    
            "MAWindowSize": 15,
            "MAPrice": "close",
    
            "BBPeriod": 20,
            "BBStandardDeviations": 2,
    
            "atrFactor": 1,
            "atrPeriod": 14,
    
            "RSIPeriod": 14,
    
            "significantMovementPeriod": 2,
    
            "windowSize": 10,  # in function: GenerateDataSet
            "getNoMovementEvery": 1,  # in function: GenerateDataSet
        }

"""