import datetime
import os
import shutil

import numpy as np
import mplfinance as mpf
import pandas as pd
import MetaTrader5 as mt5
import multiprocessing
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import dataCollection
#import dataCollection
#from dataCollection import ratesSymbol, ratesTimeFrame
import test_model

ratesSymbol = "XAUUSD"
ratesTimeFrame = mt5.TIMEFRAME_M5
# 1. Define the worker function
def run_trader(path, login, password, server, TIConfiguration):
    # Each process initializes its own connection
    if not mt5.initialize(path=path, login=login, password=password, server=server):
        print(f"Failed to initialize account {login}")
        return

    print(f"Worker started for account: {login}")

    includeBB = 0
    includeRSI = 0
    includeOBV = 0

    if "All_TIs" in TIConfiguration:
        includeBB = 1
        includeRSI = 1
        includeOBV = 1
    elif TIConfiguration == "No_BB":
        includeRSI = 1
        includeOBV = 1
    elif "OBV" in TIConfiguration:
        includeOBV = 1
    elif "RSI" in TIConfiguration:
        includeRSI = 1
    elif TIConfiguration == "No_RSI":
        includeBB = 1
        includeOBV = 1
    elif "BB" in TIConfiguration:
        includeBB = 1
    elif TIConfiguration == "No_OBV":
        includeRSI = 1
        includeBB = 1

    last_min = datetime.datetime.now().minute

    TIConfiguration = TIConfiguration
    model = test_model.TestingModel(TIConfiguration)
    lastCandle = waitForNewCandle(ratesSymbol, ratesTimeFrame)
    try:
        while True:
            #current_min = datetime.datetime.now().minute
            currencCandle = waitForNewCandle(ratesSymbol, ratesTimeFrame)
            if currencCandle is not None and currencCandle > lastCandle and datetime.time(8, 30) <= datetime.datetime.now().time() <= datetime.time(23):
                print(f"New minute detected! It is now {datetime.datetime.now().strftime('%H:%M')}")
                lastCandle = waitForNewCandle(ratesSymbol, ratesTimeFrame)

                #last_min = current_min

                ratesData = pd.DataFrame(mt5.copy_rates_from_pos(ratesSymbol, ratesTimeFrame, 0, 60))
                # TECHNICAL INDICATORS CALCULATIONS##################################################################################
                ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
                ratesData = ratesData.set_index("time")
                # ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
                ratesData = dataCollection.MACalculator(ratesData, dataCollection.MAWindowSize, dataCollection.MAPrice)
                ratesData = dataCollection.BollingerBandsCalculator(ratesData, dataCollection.BBPeriod,
                                                                    dataCollection.BBStandardDeviations)
                ratesData = dataCollection.RelativeStrengthIndexCalculator(ratesData, dataCollection.RSIPeriod)
                ratesData = dataCollection.OnBalanceVolume(ratesData)
                ratesData = dataCollection.AverageTrueRangeCalculator(ratesData, dataCollection.atrPeriod)
                # Prepares data for chart
                ratesData.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
                ####################################################################################################################
                print(ratesData.columns)

                ratesData = ratesData.tail(10)

                saveFolderName = os.path.join("inputGraph", TIConfiguration)
                if os.path.exists(saveFolderName):
                    shutil.rmtree(saveFolderName)
                    print(f"Deleted existing folder: "+saveFolderName)
                os.makedirs(saveFolderName)

                dataCollection.makePlot(ratesData, saveFolderName, includeRSI, includeOBV, includeBB, ratesTimeFrame)

                #GET MODEL PREDICTION HERE!
                direction = model.image_to_prediction()  # Kör modell mot bild
                print (direction)
                # GET MODEL PREDICTION HERE!

                atr = ratesData["averageTrueRange"].iloc[-1]

                order_type = direction

                #symbol_info = mt5.symbol_info("XAUUSD")
                #digits = symbol_info.digits
                tick = mt5.symbol_info_tick(ratesSymbol)
                spread = tick.ask - tick.bid
                #digits = mt5.symbol_info(ratesSymbol).digits

                if direction == "buy":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask  # Buy at the Ask
                    sl = price - (atr * 1) # SL is below price
                    tp = price + (atr * 1)  # TP is above price

                elif direction == "sell":
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid  # Sell at the Bid
                    sl = price + (atr * 1)  # SL is above price
                    tp = price - (atr * 1)  # TP is below price

                expiration_time = int(time.time() + (60*5))
                #price_rounded = round(price, digits)
                #sl_rounded = round(sl, digits)
                #tp_rounded = round(tp, digits)

                if order_type == mt5.ORDER_TYPE_BUY or order_type == mt5.ORDER_TYPE_SELL:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": ratesSymbol,
                        "volume": 0.1,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 1,
                        "comment": "python script open",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    tick = mt5.symbol_info_tick(ratesSymbol)
                    print(f"Buying {ratesSymbol} | Chart Price: {price} | Actual Spread: {tick.ask - tick.bid}")
                    # send a trading request
                    result = mt5.order_send(request)

                    #ERROR CODE FROM MT5
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print("2. order_send failed, retcode={}".format(result.retcode))
                        # request the result as a dictionary and display it element by element
                        result_dict = result._asdict()
                        for field in result_dict.keys():
                            print("   {}={}".format(field, result_dict[field]))
                            # if this is a trading request structure, display it element by element as well
                            if field == "request":
                                traderequest_dict = result_dict[field]._asdict()
                                for tradereq_filed in traderequest_dict:
                                    print("       traderequest: {}={}".format(tradereq_filed,
                                                                              traderequest_dict[tradereq_filed]))

                    elif result.retcode == mt5.TRADE_RETCODE_DONE:
                        market_context = {}
                        for i, row in ratesData.iterrows():
                            market_context[f'tick_{i}_close'] = row['close']
                            market_context[f'tick_{i}_vol'] = row['tick_volume']
                        csvData = {**market_context, **request, **result._asdict()}
                        csvdf = pd.DataFrame([csvData])

                        csv_file = TIConfiguration+'trade_dataset.csv'
                        file_exists = os.path.isfile(csv_file)
                        csvdf.to_csv(csv_file, mode='a', index=False, header=not file_exists)
                        print("Dataset updated with trade and market context.")
                    # ERROR CODE FROM MT5
            elif currencCandle is not None and currencCandle > lastCandle:
                #last_min = current_min
                lastCandle = waitForNewCandle(ratesSymbol, ratesTimeFrame)
                print(f"New minute detected outside trading hours! It is now {datetime.datetime.now().strftime('%H:%M')}")

            time.sleep(1)  # Wait 1 second before next loop
    except KeyboardInterrupt:
        print(f"Shutting down {login}")
    finally:
        mt5.shutdown()

def waitForNewCandle(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if rates is None or len(rates) == 0:
        return None

    return rates[0]['time']


if __name__ == '__main__':
    # Define your account configurations
    print("Configure accounts")
    account_configs = [
        {
            "path": r"N:\MT5Terminals\Terminal1\MetaTrader 5\terminal64.exe",
            "login": 5048694908, "password": "MlW_G6Mj", "server": "MetaQuotes-Demo",
            "TIConfiguration": "BB_70_10_10_10_20260131_10_15_2__20260420_163426"
        },
        {
            "path": r"N:\MT5Terminals\Terminal2\MetaTrader 5\terminal64.exe",
            "login": 5048694942, "password": "-1QeDlGw", "server": "MetaQuotes-Demo",
            "TIConfiguration": "BB_70_10_10_10_20260131_10_15_2__20260420_164230"
        },
        {
            "path": r"N:\MT5Terminals\Terminal3\MetaTrader 5\terminal64.exe",
            "login": 5048694953, "password": "!q4wIpQy", "server": "MetaQuotes-Demo",
            "TIConfiguration": "NO_TI_70_10_10_10_20260131_10_15_2__20260420_152155"
        },
        {
            "path": r"N:\MT5Terminals\Terminal4\MetaTrader 5\terminal64.exe",
            "login": 105399222, "password": "Jl*yWb1q", "server": "MetaQuotes-Demo",
            "TIConfiguration": "NO_TI_70_10_10_10_20260131_10_15_2__20260420_153000"
        },
        {
            "path": r"N:\MT5Terminals\Terminal5\MetaTrader 5\terminal64.exe",
            "login": 105399234, "password": "Ke!1CyDy", "server": "MetaQuotes-Demo",
            "TIConfiguration": "OBV_70_10_10_10_20260131_10_15_2__20260420_161638"
        },
        {
            "path": r"N:\MT5Terminals\Terminal6\MetaTrader 5\terminal64.exe",
            "login": 105399322, "password": "-vL3GrUs", "server": "MetaQuotes-Demo",
            "TIConfiguration": "OBV_70_10_10_10_20260131_10_15_2__20260420_162532"
        },
        {
            "path": r"N:\MT5Terminals\Terminal7\MetaTrader 5\terminal64.exe",
            "login": 105399330, "password": "@6DxNeZj", "server": "MetaQuotes-Demo",
            "TIConfiguration": "RSI_70_10_10_10_20260131_10_15_2__20260420_174417"
        },
        {
            "path": r"N:\MT5Terminals\Terminal8\MetaTrader 5\terminal64.exe",
            "login": 105399341, "password": "-4TrOzHd", "server": "MetaQuotes-Demo",
            "TIConfiguration": "RSI_70_10_10_10_20260131_10_15_2__20260420_175031"
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

    processes = []
    # Launch a process for each config
    for config in account_configs:
        print("Launching account")
        p = multiprocessing.Process(target=run_trader, kwargs=config)
        p.start()
        processes.append(p)

    # Keep the main script alive while workers run
    for p in processes:
        p.join()



# Connect to Account A (Terminal 1)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal1\MetaTrader 5\terminal64.exe",
#               login=5048694908, password="MlW_G6Mj", server="MetaQuotes-Demo")
#E_LhU3Ih  (Read only password)
#mt5.shutdown()
#print("Terminal 1 online")

# Connect to Account B (Terminal 2)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal2\MetaTrader 5\terminal64.exe",
#               login=5048694942, password="-1QeDlGw", server="MetaQuotes-Demo")
#I!Km0aWm  (Read only password)
#mt5.shutdown()
#print("Terminal 2 online")

# Connect to Account C (Terminal 3)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal3\MetaTrader 5\terminal64.exe",
#               login=5048694953, password="!q4wIpQy", server="MetaQuotes-Demo")
#@bK0OdUm  (Read only password)
#mt5.shutdown()
#print("Terminal 3 online")

"""

        
        105818716
        !iTu3fZs
        
        1:1 Risk/reward
        Low spread
        Stocks
        
"""