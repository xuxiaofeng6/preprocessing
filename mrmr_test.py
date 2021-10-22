import pandas as pd

import pymrmr

#df = pd.read_csv('test_colon_s3.csv')
df = pd.read_csv('Rank.csv')
print(df)

pymrmr.mRMR(df, 'MIQ', 10)