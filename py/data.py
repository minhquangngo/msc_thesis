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
[x] Momentum factor (MOM)
[x] Profitability factor (Robust minus weak (RMW))
[x] Investment factor (CMA)

'''

#Establish wrds connection
db = wrds.Connection()

#------------------------------------------------------------------
#Stocks 
'''
Database:
- crsp.dsf: permno, stock date, permno, volume, return
- crsp.dsenames: permno, ticker, sector
- crsp.ccmxpf_linktable : gvkey match with permno
- comp.company: gsector
'''

def load_query(name: str) -> str:
    """Load SQL query from a file in the `queries` folder."""
    return Path(f"{name}.sql").read_text()

params = (start_date, end_date)
query = load_query("query")
stock_ff_sector = db.raw_sql(query, params=params)

stock_ff_sector['permno'].unique()
stock_ff_sector.shape

##parquet it
stock_ff_sector.to_parquet("../data/stock_ff_sector.parquet")

#------------------------------------------------------------------
#Data conversions
stock_ff_sector = pd.read_parquet("../data/stock_ff_sector.parquet", engine = 'pyarrow')
stock_ff_sector.dtypes
stock_ff_sector['date'] = pd.to_datetime(stock_ff_sector['date'], format='%Y-%m-%d')
stock_ff_sector['gsector'] = pd.to_numeric(stock_ff_sector['gsector'], errors = 'coerce')
stock_ff_sector = stock_ff_sector.set_index('date')
#------------------------------------------------------------------
#GICS matching
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

#------------------------------------------------------------------
#Sentiment data
##Ung enhanced baker
sentiment = pd.read_csv("../data/sentiment_ung.csv",sep = ',')
sentiment['Date'] = pd.to_datetime(sentiment['Date'], format='%Y%m')
sentiment = sentiment.set_index('Date').resample("D").ffill()
pd.api.types.is_datetime64_any_dtype(sentiment.index) ## Sentiment date is date-time

## VIX
vix = yf.download("^VIX", start=start_date, end=end_date)
vix.columns = vix.columns.get_level_values(0)  # Keep just 'Open', 'Close', etc.
vix= vix[['Close']].rename(columns={'Close':'vix_close'})
vix = vix.iloc[:, :2]
pd.api.types.is_datetime64_any_dtype(vix.index) ## index VIX is date-time


## Daily news sentiment
news_sentiment = pd.read_csv("../data/news_sentiment_data.csv", sep = ",")
news_sentiment['date'] = pd.to_datetime(news_sentiment['date'], format = '%d/%m/%Y')
news_sentiment = news_sentiment.set_index('date')

full_df = stock_ff_sector.merge(sentiment, left_index=True, right_index=True, how='left')
full_df = full_df.merge(vix, left_index= True, right_index= True, how = 'left')
full_df = full_df.merge(news_sentiment,left_index = True, right_index = True, how = 'left')
#Close WRDS connection
db.close()
