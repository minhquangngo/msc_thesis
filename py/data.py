import wrds
from pathlib import Path
import pandas as pd
import yfinance as yf
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
[] Daily news sentiment
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

#Dates
start_date = "1998-01-01"
end_date = "2018-12-31"
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

##pickle it
stock_ff_sector.to_pickle("../data/stock_ff_sector.pkl")

#------------------------------------------------------------------
#Data conversions
stock_ff_sector = pd.read_pickle("../data/stock_ff_sector.pkl")
stock_ff_sector.dtypes
stock_ff_sector['date'] = pd.to_datetime(stock_ff_sector['date'], format='%Y-%m-%d')
stock_ff_sector['gsector'] = pd.to_numeric(stock_ff_sector['gsector'], errors = 'coerce')
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
stock_ff_sector['gsector_name'] = stock_ff_sector['gsector'].map(gsector_mcrsp_fundnor']
stock_ff_sector[stock_ff_sector['gsector'].isna() == True]
print("Unmapped gsector codes:", unmapped)
#------------------------------------------------------------------
#Sentiment data
##Ung enhanced baker
sentiment = pd.read_csv("../data/sentiment_ung.csv",sep = ',')
sentiment.iloc[:, 0] = pd.to_datetime(sentiment.iloc[:, 0], format='%Y%m')
sentiment.dtypes

## VIX
vix = yf.download("^VIX", start=start_date, end=end_date)

## Put call ratio


#Close WRDS connection
db.close()
