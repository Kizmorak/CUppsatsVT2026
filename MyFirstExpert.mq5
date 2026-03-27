//+------------------------------------------------------------------+
//|                                                MyFirstExpert.mq5 |
//|                                   Copyright 2026, Erik Holmström |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Erik Holmström"
#property link      "https://www.mql5.com"
#property version   "1.00"




//--- input parameters
#include <Trade\Trade.mqh>

datetime m_LastBarTime; // Stores the time of the last processed candle
CTrade m_Trade;
int myCount;
int handleATR;

struct TradeSignal {
   datetime time;
   int type; // 0 = Buy, 1 = Sell
   double atr;
};

TradeSignal signals[]; // Array to store our CSV data
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
  string fileName = "backtestDataset.csv";
   
   // Open the file for reading in the "Common" or "Files" folder
   int fileHandle = FileOpen(fileName, FILE_READ|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   
   if(fileHandle != INVALID_HANDLE) {
      while(!FileIsEnding(fileHandle)) {
         TradeSignal newSignal;
         // Read the first column as a string and convert to datetime
         newSignal.time = StringToTime(FileReadString(fileHandle));
         // Read the second column as an integer
         newSignal.type = (int)FileReadNumber(fileHandle);
         //Read the third column as a double
         newSignal.atr = (double)FileReadNumber(fileHandle);
         
         // Add to our array
         int size = ArraySize(signals);
         ArrayResize(signals, size + 1);
         signals[size] = newSignal;
      }
      FileClose(fileHandle);
      Print("Loaded ", ArraySize(signals), " signals from CSV.");
      
      Print("--- CSV Data Loaded ---");
      int totalSignals = ArraySize(signals);
      
      for(int i = 0; i < totalSignals; i++)
      {
         string signalTime = TimeToString(signals[i].time, TIME_DATE|TIME_MINUTES);
         //double signalatr = signals[i].atr;
   
         // Updated logic: 1 is BUY, 2 is SELL
         string signalType = "UNKNOWN";
         if(signals[i].type == 1) signalType = "BUY";
         else if(signals[i].type == 2) signalType = "SELL";
         
         //Print("Index [", i, "]: Time = ", signalTime, " | Type = ", signalType, " | ATR = " + signalatr);
         Print("Index [", i, "]: Time = ", signalTime, " | Type = ", signalType);
         
         if(i > 50) break;
      }
      Print("--- End of Array ---");
   } else {
      Print("Failed to open file. Error: ", GetLastError());
   }
  
   m_LastBarTime = iTime(_Symbol, _Period, 0);
   
   handleATR = iATR(_Symbol, _Period, 14);
   if(handleATR == INVALID_HANDLE){
      Print("Failed to create ATR handle");
      return(INIT_FAILED);
   }
   Print("START");
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!IsNewBar()) return;
   
   Print("New candle detected at: ", TimeToString(TimeCurrent()));
   
   datetime currentTime = iTime(_Symbol, _Period, 0);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   for(int i = 0; i < ArraySize(signals); i++) {
      if(signals[i].time == currentTime) {
         //double currentAtr = signals[i].atr;
         double sl_level, tp_level;
         
         double bufferATR[]; // Array to hold the value
         ArraySetAsSeries(bufferATR, true); // Arrange so [0] is the newest candle
         
         // Copy the latest ATR value (1 value from the current candle)
         if(CopyBuffer(handleATR, 0, 0, 1, bufferATR) < 0)
         {
            Print("Error copying ATR data: ", GetLastError());
            return;
         }
         
         double currentAtr = bufferATR[0]; // This is your ATR value!
         
         if(signals[i].type == 1) // BUY
         {
            sl_level = ask - (5 * currentAtr);
            tp_level = ask + (5 * currentAtr);
            
            m_Trade.Buy(0.1, _Symbol, ask, sl_level, tp_level, "CSV Buy with ATR");
         }
         else if(signals[i].type == 2) // SELL
         {
            sl_level = bid + (5 * currentAtr);
            tp_level = bid - (5 * currentAtr);
            
            m_Trade.Sell(0.1, _Symbol, bid, sl_level, tp_level, "CSV Sell with ATR");
         }
      }
   }

  /* double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   
   double marginRequired;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 0.1, ask, marginRequired))
   {
      Print("Error calculating margin: ", GetLastError());
      return;
   }
   
   Print("Current margin: ", freeMargin, " | Needed: ", marginRequired);
   
   if(freeMargin > marginRequired)
   {
      //m_Trade.Buy(0.1, _Symbol, ask, 0, 0, "New Candle Trade"); //OPENS A BUY OPTION
      m_Trade.Sell(0.1, _Symbol, bid, 0, 0, "Sell Candle"); //OPENS A SELL OPTION
   }
   else
   {
      Print("Out of funds! Attempting to close the oldest position...");
      uint total = PositionsTotal();
      if(total > 0)
      {
         ulong oldestTicket = 0;
         datetime oldestTime = 0;

         for(int i = 0; i < (int)total; i++)
         {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket))
            {
               datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);
               if(oldestTime == 0 || posTime < oldestTime)
               {
                  oldestTime = posTime;
                  oldestTicket = ticket;
               }
            }
         }
         
         if(oldestTicket > 0)
         {
            m_Trade.PositionClose(oldestTicket);
            Print("Closed oldest position: #", oldestTicket);
         }
      } // End of 'if total > 0'
   } // <--- YOU WERE MISSING THIS: End of 'else'
   /*
   myCount++;  */
}
   
   //THIS CODE OPENS A SHORT POSITION
   //else{
   //   m_Trade.Sell(0.1, _Symbol, bid, 0, 0, "Sell Candle");
   //}
   
//+------------------------------------------------------------------+
//| NewBar function                                                  |
//+------------------------------------------------------------------+
bool IsNewBar(){
   datetime currentTime = iTime(_Symbol, _Period, 0);
   
   if(currentTime == m_LastBarTime){
      return false;
   }

   m_LastBarTime = currentTime;
   return true;
}