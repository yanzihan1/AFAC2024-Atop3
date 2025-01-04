import pandas as pd
import sys

df_train =  pd.read_excel('../data/train.xlsx')
df_dev = pd.read_excel('../data/dev.xlsx')

df_dev = pd.concat([df_train, df_dev])
#df_dev
df_testb = pd.read_csv('../data/test_b.txt',sep='\001',header=None,names=['query'],quoting=3)
df_testb['result'] = pd.read_csv(sys.argv[1],sep='\001',header=None,names=['result'],quoting=3).iloc[:,0]
#print(df_testa)
df_merge = df_testb.merge(df_dev.drop_duplicates(),on=['query'],how='left')
print(len(df_merge))
import json
cnt = 0
#print(df_merge)
fw = open(sys.argv[2],'w')
for i in range(df_merge.shape[0]):
    if not pd.isna(df_merge.iloc[i,2]):
        pred = json.loads(df_merge.iloc[i,1])
        label = json.loads(df_merge.iloc[i,2])
        if pred != label:
            cnt += 1
            print(i,pred)
            print(label)
            fw.write(json.dumps(label,ensure_ascii=False) + '\n')
            continue
    fw.write(json.dumps(json.loads(df_merge.iloc[i,1]),ensure_ascii=False)+'\n')
print(cnt)
