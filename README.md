we can edit this later just adding a few improvement ideas/potential issues rn:

1. change to parquet files / use pyarrow
2. use polars
3. we're not getting enough clusters in hdbscan??
4. change backtesting workflow to alphamark


current data cleaning workflow (fix later this isn't ideal):
1. try forward fill sic2 vals using past dates
2. fill in the group median for each industry using sic2
3. for remaining, fill date median
4. for all missing sic2 vals, we assign code 99


reqd file structure for data:
have a data folder at root, with raw and processed folders. raw contains the monthly data, with csv under a characteristics_by_month folder