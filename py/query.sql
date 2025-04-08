WITH StockData AS (
    SELECT
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
    FROM crsp.dsf AS stock

    INNER JOIN crsp_a_indexes.dsp500list_v2 AS sp500
        ON stock.permno = sp500.permno
        AND stock.date BETWEEN sp500.mbrstartdt AND COALESCE(sp500.mbrenddt, stock.date)
    
    LEFT JOIN crsp.dsenames AS names
        ON stock.permno = names.permno
        AND stock.date BETWEEN names.namedt AND COALESCE(names.nameendt, stock.date)
    
    LEFT JOIN crsp.ccmxpf_lnkused AS link
        ON stock.permno = link.upermno
        AND link.usedflag = 1
        AND stock.date BETWEEN link.ulinkdt AND COALESCE(link.ulinkenddt, stock.date)
    
    LEFT JOIN comp.company AS cmp
        ON link.ugvkey = cmp.gvkey 
    
    LEFT JOIN ff.fivefactors_daily AS ff5
        ON stock.date = ff5.date
    
    WHERE stock.date BETWEEN %s AND %s
),
OptionData AS (
    SELECT
        p.date,
        p.secid,
        linktab.permno,
        p.volume AS put_volume,
        c.volume AS call_volume,
        CAST(p.volume AS FLOAT) / NULLIF(c.volume, 0) AS put_call_ratio
    
    FROM optionm.opvold AS p
    
    JOIN optionm.opvold AS c
        ON p.secid = c.secid
        AND p.date = c.date
        AND c.cp_flag = 'C'
    
    JOIN wrdsapps.opcrsphist AS linktab
        ON p.secid = linktab.secid
        AND p.date BETWEEN linktab.sdate AND COALESCE(linktab.edate, CURRENT_DATE)
    
    WHERE p.cp_flag = 'P'
)
SELECT 
    s.date,
    s.permno,
    s.ticker,
    s.vol,
    s.ret,
    s.shrout,
    s.prc,
    s.ask,
    s.bid,
    s.gsector,
    s.mktrf,
    s.smb,
    s.hml,
    s.rmw,
    s.umd,
    s.cma,
    s.rf,
    o.put_volume,
    o.call_volume,
    o.put_call_ratio
FROM StockData AS s

LEFT JOIN OptionData AS o
    ON s.date = o.date AND s.permno = o.permno

ORDER BY s.date, s.permno;