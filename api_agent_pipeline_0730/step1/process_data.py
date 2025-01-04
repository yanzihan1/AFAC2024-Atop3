import re
import sys
import pandas as pd

prefix = sys.argv[1]
pattern = r"query：(.*?)。请生成query中涉及的基金或股票实体的表征"
res = []
for line in sys.stdin:
    line = line.strip().split("\t")
    query = re.search(pattern, line[0]).group(1)
    res.append([query,line[1],line[2]])

df = pd.DataFrame(res)
df.drop_duplicates([0,1]).groupby(0).head(5).to_csv(prefix+"_5.txt",sep='\t',header=None,index=False)
df.drop_duplicates([0,1]).groupby(0).head(15).to_csv(prefix+"_15.txt",sep='\t',header=None,index=False)
df.drop_duplicates([0,1]).groupby(0).head(20).to_csv(prefix+"_20.txt",sep='\t',header=None,index=False)
df.drop_duplicates([0,1]).groupby(0).head(50).to_csv(prefix+"_50.txt",sep='\t',header=None,index=False)
