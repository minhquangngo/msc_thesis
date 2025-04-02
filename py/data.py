import wrds
'''
Importing all of the data in
Period: 1990 -2018 (monthly)
Stocks:
[x] permno -> crsp a indexes.dsp500list v2
[x] Returns
[x] Volume


Sector:


Liquidity risk factors
[x] Turn over (turn) (vol/shrout)
[x] Turn over volatility (SDTurn) (std. of turn)
[x] log market equity (mvel1) (prc)
[] dollar volume (dolvol)
[x] AMIHUD ILLQ: can get through "returns" and "volume"
[] # zero trading days (zero trade)
[] bid-ask spread (baspread)

Sentiment

Factor variables


'''

#Establish wrds connection
db = wrds.Connection()
db.list_libraries()
db.list_tables(library='crsp')

#Dates
start_date = "1998-01-01"
end_date = "2018-12-31"

#Stocks 
'''
Database:
- crsp.dsf: stock date, permno, volume, return
- crsp_a_indexes.dsp500list_v2: permno
'''
stock = db.raw_sql(
    f"""
select 
    stock.date, stock.permno, 
    stock.vol, stock.ret,
    stock.shrout, stock.prc
from crsp.dsf as stock
inner join (
    select permno
    from crsp_a_indexes.dsp500list_v2
        where mbrenddt >= '{end_date}' or mbrenddt is null
            and mbrstartdt <= '{start_date}'
    group by permno
) as constituents
on stock.permno = constituents.permno
where stock.date between '{start_date}' and '{end_date}'
order by stock.date, stock.permno;
""")

stock.permno.unique()

#Close WRDS connection
db.close()
