import pandas as pd

df = pd.concat([pd.read_excel('../data/train.xlsx'), pd.read_excel('../data/dev.xlsx')])

for idx in range(df.shape[0]):
    print(df.iloc[idx, 0])
