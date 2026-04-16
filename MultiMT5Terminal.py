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
from dataCollection import ratesSymbol, ratesTimeFrame
import test_model


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

    if TIConfiguration == "All_TIs":
        includeBB = 1
        includeRSI = 1
        includeOBV = 1
    elif TIConfiguration == "No_BB":
        includeRSI = 1
        includeOBV = 1
    elif TIConfiguration == "No_BB_No_RSI":
        includeOBV = 1
    elif TIConfiguration == "No_BB_No_OBV":
        includeRSI = 1
    elif TIConfiguration == "No_RSI":
        includeBB = 1
        includeOBV = 1
    elif TIConfiguration == "No_RSI_No_OBV":
        includeBB = 1
    elif TIConfiguration == "No_OBV":
        includeRSI = 1
        includeBB = 1

    last_min = datetime.datetime.now().minute

    TIConfiguration = TIConfiguration + "_model"
    model = test_model.TestingModel(TIConfiguration)

    try:
        while True:
            current_min = datetime.datetime.now().minute
            # Your trading logic goes here
            # Example: check prices or send orders
            if current_min != last_min and datetime.time(8, 30) <= datetime.datetime.now().time() <= datetime.time(15):
                # --- DO SOMETHING HERE ---
                print(f"New minute detected! It is now {datetime.datetime.now().strftime('%H:%M')}")

                last_min = current_min

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

                ratesData = ratesData.tail(30)

                saveFolderName = os.path.join("inputGraph", TIConfiguration)
                if os.path.exists(saveFolderName):
                    shutil.rmtree(saveFolderName)
                    print(f"Deleted existing folder: "+saveFolderName)
                os.makedirs(saveFolderName)

                makePlot(ratesData, saveFolderName, includeBB, includeRSI, includeOBV)

                #GET MODEL PREDICTION HERE!
                direction = model.image_to_prediction()  # Kör modell mot bild
                print (direction)
                # GET MODEL PREDICTION HERE!

                atr = ratesData["averageTrueRange"].iloc[-1]

                order_type = direction

                tick = mt5.symbol_info_tick(ratesSymbol)
                spread = tick.ask - tick.bid
                #digits = mt5.symbol_info(ratesSymbol).digits

                if direction == "buy":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask  # Buy at the Ask
                    sl = price - (atr * 5) - spread  # SL is below price
                    tp = price + (atr * 5) + spread  # TP is above price

                elif direction == "sell":
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid  # Sell at the Bid
                    sl = price + (atr * 5) + spread  # SL is above price
                    tp = price - (atr * 5) - spread  # TP is below price

                expiration_time = int(time.time() + 60)


                if order_type == mt5.ORDER_TYPE_BUY or order_type == mt5.ORDER_TYPE_SELL:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": ratesSymbol,
                        "volume": 0.1,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 0,
                        "comment": "python script open",
                        "type_time": mt5.ORDER_TIME_SPECIFIED,
                        "expiration": expiration_time,
                        "type_filling": mt5.ORDER_FILLING_FOK,
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
                    # ERROR CODE FROM MT5
            elif current_min != last_min:
                last_min = current_min
                print(f"New minute detected outside trading hours! It is now {datetime.datetime.now().strftime('%H:%M')}")

            time.sleep(1)  # Wait 1 second before next loop
    except KeyboardInterrupt:
        print(f"Shutting down {login}")
    finally:
        mt5.shutdown()

def makePlot(ratesData, saveFolderName, includeBB, includeRSI, includeOBV, includeTIs = 1, includeMA30 = 0):
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
    file_path = os.path.join(saveFolderName,"myImage.png")

    #Save figure
    fig.savefig(file_path)

    print(f"Grafen är sparad som: {file_path}")

    #Show graph
    #plt.show()
    #Close graph
    plt.close()
    return ""


if __name__ == '__main__':
    # Define your account configurations
    print("Configure accounts")
    account_configs = [

        {
            "path": r"N:\MT5Terminals\Terminal8\MetaTrader 5\terminal64.exe",
            "login": 105399341, "password": "-4TrOzHd", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_OBV"
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
{
            "path": r"N:\MT5Terminals\Terminal1\MetaTrader 5\terminal64.exe",
            "login": 5048694908, "password": "MlW_G6Mj", "server": "MetaQuotes-Demo",
            "TIConfiguration": "All_TIs"
        },
        {
            "path": r"N:\MT5Terminals\Terminal2\MetaTrader 5\terminal64.exe",
            "login": 5048694942, "password": "-1QeDlGw", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_TIs"
        },
        {
            "path": r"N:\MT5Terminals\Terminal3\MetaTrader 5\terminal64.exe",
            "login": 5048694953, "password": "!q4wIpQy", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_BB"
        },
        {
            "path": r"N:\MT5Terminals\Terminal4\MetaTrader 5\terminal64.exe",
            "login": 105399222, "password": "Jl*yWb1q", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_BB_No_RSI"
        },
        {
            "path": r"N:\MT5Terminals\Terminal5\MetaTrader 5\terminal64.exe",
            "login": 105399234, "password": "Ke!1CyDy", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_BB_No_OBV"
        },
        {
            "path": r"N:\MT5Terminals\Terminal6\MetaTrader 5\terminal64.exe",
            "login": 105399322, "password": "-vL3GrUs", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_RSI"
        },
        {
            "path": r"N:\MT5Terminals\Terminal7\MetaTrader 5\terminal64.exe",
            "login": 105399330, "password": "@6DxNeZj", "server": "MetaQuotes-Demo",
            "TIConfiguration": "No_RSI_No_OBV"
        },
        
        105818716
        !iTu3fZs
        
        1:1 Risk/reward
        Low spread
        Stocks
        
"""