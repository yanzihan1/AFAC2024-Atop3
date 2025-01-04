import json
import sys

for line in sys.stdin:
    sl = line.strip().split('\t')
    if len(sl) > 1:
        res_dic = {"input":sl[0],"output":sl[1]}
    else:
        res_dic = {"input":sl[0]}
    print(json.dumps(res_dic,ensure_ascii=False))
