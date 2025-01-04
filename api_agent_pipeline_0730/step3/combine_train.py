import pandas as pd

df = pd.concat([pd.read_excel('../data/train.xlsx'), pd.read_excel('../data/dev.xlsx')])
df.to_excel('train_withdev.xlsx', index=False)
