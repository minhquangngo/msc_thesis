import wrds

'''
Importing all of the data in
Period: 1990 -2018 (monthly)
- All companies must stay within the 
Stocks:
[x] permno -> crsp a indexes.dsp500list v2
[x] Returns
[x] Volume


Sector:
[x] cmp.sector
[] Sector names


Liquidity risk factors
[x] Turn over (turn) (vol/shrout)
[x] Turn over volatility (SDTurn) (std. of turn)
[x] log market equity (mvel1) (prc)
[x] dollar volume (dolvol) (#shares *price)
[x] AMIHUD ILLQ: can get through "returns" and "volume"
[x] # zero trading days (zero trade) ( zero trading day:vol =0)
[x] bid-ask spread (baspread)

Sentiment

Factor variables:
[x] Excess return of a stock
    [x]Risk free return
[x] Market return 
[x] Size factor( Small minus big)
[x] Value factor (High minus low)
[x] Momentum factor (MOM)
[x] Profitability factor (Robust minus weak (RMW))
[x] Investment factor (CMA)

[] SAVE IT
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



stock_ff_sector = db.raw_sql(
    f"""
    select
        stock.date,
        stock.permno,
        names.ticker,
        stock.vol,
        stock.ret,
        stock.shrout,
        stock.prc,
        stock.ask,
        stock.bid,
        cmp.gsector,
        ff5.mktrf,
        ff5.smb,
        ff5.hml,
        ff5.rmw,
        ff5.umd,
        ff5.cma,
        ff5.rf
    from crsp.dsf as stock

    inner join crsp_a_indexes.dsp500list_v2 as sp500
        on stock.permno = sp500.permno
        and stock.date between sp500.mbrstartdt and coalesce(sp500.mbrenddt, stock.date)

    left join crsp.dsenames as names
        on stock.permno = names.permno
        and stock.date between names.namedt and coalesce(names.nameendt, stock.date)

    left join crsp.ccmxpf_linktable as link -- query this in to get the gvkey, connecting stock with comp
        on stock.permno = link.lpermno
        and link.usedflag = 1 --primary link specific to the 2 joined 
        and stock.date between link.linkdt and coalesce(link.linkenddt, stock.date)

    left join comp.company as cmp on link.gvkey=cmp.gvkey 

    left join ff.fivefactors_daily as ff5 on stock.date = ff5.date

    where stock.date between '{start_date}' and '{end_date}'
    order by stock.date, stock.permno;
    """
)

stock_ff_sector['permno'].unique()
#Close WRDS connection
db.close()
