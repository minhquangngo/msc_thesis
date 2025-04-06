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

LEFT JOIN crsp.ccmxpf_linktable AS link
    ON stock.permno = link.lpermno
    AND link.usedflag = 1
    AND stock.date BETWEEN link.linkdt AND COALESCE(link.linkenddt, stock.date)

LEFT JOIN comp.company AS cmp ON link.gvkey = cmp.gvkey 

LEFT JOIN ff.fivefactors_daily AS ff5 ON stock.date = ff5.date

WHERE stock.date BETWEEN %s AND %s

ORDER BY stock.date, stock.permno;
