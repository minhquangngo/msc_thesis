import wrds
from pathlib import Path
import pandas as pd
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
[] Sector names - match when pickled data is imported


Liquidity risk factors
[x] Turn over (turn) (vol/shrout)
[x] Turn over volatility (SDTurn) (std. of turn)
[x] log market equity (mvel1) (prc)
[x] dollar volume (dolvol) (#shares *price)
[x] AMIHUD ILLQ: can get through "returns" and "volume"
[x] # zero trading days (zero trade) ( zero trading day:vol =0)
[x] bid-ask spread (baspread)

[x]Sentiment

Factor variables:
[x] Excess return of a stock
    [x]Risk free return
[x] Market return 
[x] Size factor( Small minus big)
[x] Value factor (High minus low)
[x] Momentum factor (MOM)
[x] Profitability factor (Robust minus weak (RMW))
[x] Investment factor (CMA)

[x] SAVE IT
'''

#Establish wrds connection
db = wrds.Connection()

#Dates
start_date = "1998-01-01"
end_date = "2018-12-31"

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

query = load_query("query")
params = (start_date, end_date)
stock_ff_sector = db.raw_sql(query, params=params)

stock_ff_sector['permno'].unique()
stock_ff_sector.shape

##pickle it
stock_ff_sector.to_pickle("../data/stock_ff_sector.pkl")
stock_ff_sector = pd.read_pickle("../data/stock_ff_sector.pkl")
stock_ff_sector.dtypes
#TODO: change types, merge with sentiment
#Sentiment data
sentiment = pd.read_csv("../data/sentiment_ung.csv",sep = ',')
sentiment.iloc[:, 0] = pd.to_datetime(sentiment.iloc[:, 0], format='%Y%m')
#Close WRDS connection
db.close()
