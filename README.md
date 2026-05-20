we can edit this later just adding a few improvement ideas/potential issues rn:

1. change to parquet files / use pyarrow
2. use polars
3. we're not getting enough clusters in hdbscan??
4. think abt spherical k means stopping rule? maybe better would be checking for assignment change and checking sum of squared diff for all centroids? because in high dimensions coord change might be very small but impactful?
5. is hdbscan even useful considering curse of dimensionality?
6. reconsider all hyperparam vals
7. data cleaning methods might be cooked (currently taking sic2 medians acc to date) + is winsorization needed?


current data cleaning workflow (fix later this isn't ideal):
1. try forward fill sic2 vals using past dates
2. fill in the group median for each industry using sic2
3. for remaining, fill date median
4. for all missing sic2 vals, we assign code 99


reqd file structure for data:
have a data folder at root, with raw and processed folders. raw contains the monthly data, with csv under a characteristics_by_month folder