we can edit this later just adding a few improvement ideas/potential issues rn:

1. change to parquet files / use pyarrow
2. use polars
3. we're not getting enough clusters in hdbscan??
4. think abt spherical k means stopping rule? maybe better would be checking for assignment change and checking sum of squared diff for all centroids? because in high dimensions coord change might be very small but impactful?
5. is hdbscan even useful considering curse of dimensionality?
6. reconsider all hyperparam vals
7. data cleaning methods might be cooked (nothing rn just ignoring data where we can)


reqd file structure for data:
have a data folder at root, with raw and processed folders. raw contains the monthly data, with csv under a characteristics_by_month folder