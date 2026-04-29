import pandas as pd
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

#==========================================
#1. CONFIGURATION & TICKER LIST
#==========================================
#List of 99 SP's 100 tickers we chose to represent the market
TICKERS = [
    'AAPL', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AMGN', 'AMZN', 'AMT', 'APH', 
    'ADI', 'ANET', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 
    'C', 'CAT', 'CHTR', 'CVX', 'CI', 'COST', 'CRM', 'CSCO', 'CVS', 'DHR', 
    'DIS', 'DOW', 'ELV', 'EMR', 'ETN', 'XOM', 'META', 'FDX', 'FIS', 'GD', 
    'GE', 'GILD', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU', 'ISRG', 'JPM', 
    'KDP', 'KLAC', 'KMB', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 
    'MDLZ', 'MDT', 'MET', 'MRK', 'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 
    'ORCL', 'PANW', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SBUX', 
    'SCHW', 'SO', 'SPGI', 'SYK', 'T', 'TGT', 'TJX', 'TMO', 'TMUS', 'TSLA', 
    'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'VRTX', 'WFC', 'WMT'
]

#16 years of data provides almost 4000 trading days for each stock
START_DATE = "2000-01-01"
END_DATE = "2020-12-31"
#local file to save work so you dont have to download from yahoo all the time
CACHE_FILE = "stock_data.pkl"

#==========================================
#2. DATA PREPARATION LOGIC (Cleaning)
#==========================================
def prepare_data(force_download=False):
    #Check if we already saved the data locally to skip the wait
    if os.path.exists(CACHE_FILE) and not force_download:
        print(f"--- Loading cached data from {CACHE_FILE} ---")
        return pd.read_pickle(CACHE_FILE)

    print(f"--- Data Pipeline ---")
    print(f"Downloading {len(TICKERS)} stocks...")
    
    #multi_level_index=False is critical for the 2026 yfinance update (it gives a key error sometimes)
    #download all 99 stocks at once
    raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, multi_level_index=False)
    #handle the 'Price'/'Close' MultiIndex hierarchy seen in latest yfinance
    #selecting only "Close" for our math
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Close']
    else:
        data = raw_data
    #clculate log returns, turning prices like $200 into % changes
    #ln(Price Today/Price yesterday) --> continuously compounded return
    returns = np.log(data/data.shift(1))
    #Calculate Target: 20-day Forward Realized Volatility
    #Shifted -20 so target at 't' is volatility from 't+1 to t+20'
    #TARGET CALCULATION: 
    #1'rolling(20).std()' finds the standard deviation (volatility) of the last 20 days.
    #2'shift(-20)' moves that value 20 days back in time.
    #Result: Today's "target" is now the volatility that will actually happen in the FUTURE.data = raw_data
    target_vol = returns.rolling(window=20).std().shift(-20)
    #save to pkl for faster loading next time
    processed_data = (returns, target_vol)
    pd.to_pickle(processed_data, CACHE_FILE)
    print(f"--- Data saved to {CACHE_FILE} ---")
    return returns, target_vol

# ==========================================
# 3. PYTORCH DATASET CLASS
# ==========================================
class VolatilityDataset(Dataset):
    def __init__(self, returns_df, target_df, lookback=60):
        self.lookback = lookback #how maany days of history the AI sees (for now 60 days)
        self.num_stocks = returns_df.shape[1]#num of stocks = 99
        self.returns_values = returns_df.values #convert df to a fast numpy array
        self.target_values = target_df.values
        self.samples = []
        #Build valid (time_idx, stock_idx) pairs, skipping NaNs
        #For every stock(s) and every day(t)
        for s in range(self.num_stocks):
            for t in range(lookback, len(self.returns_values)):
                #check for validity of window and target
                if not np.isnan(self.returns_values[t-lookback:t, s]).any() and \
                   not np.isnan(self.target_values[t, s]):
                    self.samples.append((t, s))#store as a time stock pair
        print(f"Dataset initialized with {len(self.samples)} valid windows.")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, s = self.samples[idx]
        #look up which day (t) and which stock (s) this index refers to
        #input Sequence: (lookback, 1)
        #'unsqueeze(-1)' makes it shape (60, 1) so the LSTM likes it
        x = torch.tensor(self.returns_values[t-self.lookback:t, s], dtype=torch.float32).unsqueeze(-1)
        #one-hot Encoding for stock identification: (99,)
        one_hot = torch.zeros(self.num_stocks)
        one_hot[s] = 1.0
        #target: log-realized volatility (more stable for MSE), future volatility take log of it 
        y = torch.tensor(np.log(self.target_values[t, s]), dtype=torch.float32)
        return x, one_hot, y

#==========================================
#4. MAIN EXECUTION (FOR TESTING)
#==========================================
if __name__ == "__main__":
    returns, targets = prepare_data()
    # Create Dataset (Default 60-day lookback)
    dataset = VolatilityDataset(returns, targets, lookback=60)
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Sanity Check: Test one batch
    try:
        x, s_id, y = next(iter(loader))
        print("\n--- Preprocessing Success ---")
        print(f"Sequence Shape:  {x.shape}   (Batch, Lookback, Features)")
        print(f"Embedding Shape: {s_id.shape}  (Batch, Num_Stocks)")
        print(f"Target Shape:    {y.shape}      (Batch,)")
        print(f"\nExample Target (Log-Vol): {y[0].item():.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")
