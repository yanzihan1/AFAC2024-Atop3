import re
import sys
import json
import random

dic = {}
standard_names = []
with open('../data/standard_fund.txt','r') as f:
    for word in f:
        word = word.strip()
        standard_names.append(word)

with open('../data/standard_stock.txt','r') as f:
    for word in f:
        word = word.strip()
        standard_names.append(word)

def get_jaccard(a,b):
        lis_a = [a[i] for i in range(len(a))]
        lis_b = [b[i] for i in range(len(b))]
        return len(list(set(a)&set(b))) / len(set((set(a)|set(b))))


with open(sys.argv[1],'r') as f:
    cnt = 0
    for input_ in f:
        j = json.loads(input_)
        lis = [item['required_parameters'] if 'required_parameters' in item  else "" for item in j['relevant APIs']]
        
        for item in lis:
            if type(item[0]) == list:
                possible_item = ""
                possible_item = item[0][0]
                if possible_item != "" and possible_item not in standard_names:
                    cnt += 1
                    dis = 0
                    real_item = possible_item
                    for item in standard_names:
                        tmp_dis = get_jaccard(possible_item, item)
                        if tmp_dis > dis:
                            dis = tmp_dis
                            real_item = item
                    if dis > 0.25 and dis < 0.999:
                        #print(possible_item, real_item)
                        input_ = input_.replace(possible_item, real_item)
        print(input_.strip())
