import sys
prompt = "query：{QUERY}。请生成query中涉及的基金或股票实体的表征："
for line in sys.stdin:
    splited_line = line.strip().split('\001')
    if len(splited_line) != 3:
        continue
    query = prompt.replace('{QUERY}',splited_line[0])
    print('\t'.join([query,splited_line[1],splited_line[2]]))
