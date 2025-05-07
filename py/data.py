import os
import wrds
from pathlib import Path
import pandas as pd
import yfinance as yf


#Dates
start_date = "1998-01-01"
end_date = "2018-12-31"
'''
Importing all of the data in
Period: 1990 -2018 (monthly)
- All companies must stay within the 
Stocks:
[x] permno -> crsp a indexes.dsp500list v2
[x] Returns
[x] Volume


Sector:
[x] cmp.sectorc
[x] Sector names - match when pickled data is imported


Liquidity risk factors
[x] Turn over (turn) (vol/shrout) 
[x] Turn over volatility (SDTurn) (std. of turn)
[x] log market equity (mvel1) (prc)
[x] dollar volume (dolvol) (#shares *price)
[x] AMIHUD ILLQ: can get through "returns" and "volume"
[x] # zero trading days (zero trade) ( zero trading day:vol =0)
[x] bid-ask spread (baspread)

Sentiment (new)
[x] Volatility index
[] Put call ratio
[x] Turnover
[x] Daily news sentiment
[x] enhanced sentiment

Factor variables:
[x] Excess return of a stock
    [x]Risk free return
[x] Market return 
[x] Size factor( Small minus big)
[x] Value factor (High minus low)
[x] Momentum factor (MOM) (Ups minus downs)
[x] Profitability factor (Robust minus weak (RMW))
[x] Investment factor (CMA)
B
'''
#--------------------------------------------------------
# File paths
BASE_DIR = Path(__file__).parent.parent
STOCK_DATA_PATH = BASE_DIR / "data" / "stock_ff_sector.parquet"
VIX_DATA_PATH = BASE_DIR / "data" / "vix_data.parquet"
SENTIMENT_DATA_PATH = BASE_DIR / "data" / "sentiment_ung.csv"
NEWS_DATA_PATH = BASE_DIR / "data" / "news_sentiment_data.csv"
#sanity checks
print("Resolved STOCK_DATA_PATH:", STOCK_DATA_PATH.resolve())
print("Resolved VIX_DATA_PATH:", VIX_DATA_PATH.resolve())
print("Resolved SENTIMENT_DATA_PATH:", SENTIMENT_DATA_PATH.resolve())
print("Resolved SENTIMENT_DATA_PATH:", NEWS_DATA_PATH.resolve())
#--------------------------------------------------------

def load_query(name: str) -> str:
    """Load SQL query from a file in the query.sql directory."""
    query_path = Path(__file__).parent / f"{name}.sql"
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    print(f"Query '{name}.sql' loaded successfully from: {query_path.resolve()}")
    return query_path.read_text()

def load_stock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load stock data from local file if it exists, 
    otherwise query the WRDS database.
    """
    if os.path.exists(STOCK_DATA_PATH):
        print("Loading stock parquet")
        df = pd.read_parquet(STOCK_DATA_PATH, engine='pyarrow')
    else:
        print("No local parquet found -> Querying WRDS")
        db = wrds.Connection()
        query = load_query("query")
        params = (start_date, end_date)
        df = db.raw_sql(query, params=params)
        df.to_parquet(STOCK_DATA_PATH)
        db.close()
    return df

def load_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load VIX data from local file if it exists, 
    otherwise download from Yahoo Finance.
    """
    if os.path.exists(VIX_DATA_PATH):
        print("Loading VIX parquet.")
        vix = pd.read_parquet(VIX_DATA_PATH, engine='pyarrow')
    else:
        print("No VIX found, downloading from Yahoo")
        vix = yf.download("^VIX", start=start_date, end=end_date)
        # Simplify columns and rename for consistency
        vix.columns = vix.columns.get_level_values(0)
        vix = vix[['Close']].rename(columns={'Close': 'vix_close'})
        vix.to_parquet(VIX_DATA_PATH)
    return vix

def load_process(start_date = "1998-01-01", end_date = "2018-12-31"):
    
    # Load stock_ff_sect
    stock_ff_sector = load_stock_data(start_date, end_date)
    stock_ff_sector['date'] = pd.to_datetime(stock_ff_sector['date'], format='%Y-%m-%d')
    stock_ff_sector['mbrenddt'] = pd.to_datetime(stock_ff_sector['mbrenddt'], format='%Y-%m-%d')
    stock_ff_sector['mbrstartdt'] = pd.to_datetime(stock_ff_sector['mbrstartdt'], format='%Y-%m-%d')
    stock_ff_sector['gsector'] = pd.to_numeric(stock_ff_sector['gsector'], errors='coerce')
    stock_ff_sector = stock_ff_sector.set_index('date')

    # GICS sector mapping
    gsector_map = {
        10: "Energy",
        15: "Materials",
        20: "Industrials",
        25: "Consumer Discretionary",
        30: "Consumer Staples",
        35: "Health Care",
        40: "Financials",
        45: "Information Technology",
        50: "Communication Services",
        55: "Utilities",
        60: "Real Estate"
    }
    stock_ff_sector['gsector_name'] = stock_ff_sector['gsector'].map(gsector_map)

    # Sentiment csv
    sentiment = pd.read_csv(SENTIMENT_DATA_PATH, sep=',')
    sentiment['Date'] = pd.to_datetime(sentiment['Date'], format='%Y%m')
    sentiment = sentiment.set_index('Date').resample("D").ffill()

    # VIX
    vix = load_vix_data(start_date, end_date)

    # news sentiment data
    news_sentiment = pd.read_csv(NEWS_DATA_PATH, sep=",")
    news_sentiment['date'] = pd.to_datetime(news_sentiment['date'], format='%d/%m/%Y')
    news_sentiment = news_sentiment.set_index('date')

     # Merge all data into a single DataFrame
    full_df = stock_ff_sector.merge(sentiment, left_index=True, right_index=True, how='left')
    full_df = full_df.merge(vix, left_index=True, right_index=True, how='left')
    full_df = full_df.merge(news_sentiment, left_index=True, right_index=True, how='left')

    return full_df

if __name__ == '__main__':
    full_df = load_process()
    print("Data loaded and processed. Full DataFrame shape:", full_df.shape)