import sys

if __name__ == "__main__":
    res = []
    for line in open(sys.argv[1], 'r'):
        res.append(line.strip().split('，'))

    for idx, line in enumerate(open(sys.argv[2],'r')):
        res[idx].append(line.strip().split('，'))
    
    fw = open(sys.argv[3], 'w')
    for item in res:
        fw.write('，'.join(item[:20]))
