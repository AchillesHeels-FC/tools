# coding = utf-8
import os
import sys
from collections import defaultdict


def read_file(path):
    info_dict = defaultdict(list)
    lines = open(path).readlines()
    lines = list(map(lambda x: str(x).replace('\n', ''), lines))
    for line in lines:
        line_list = line.split('\t')
        key = line_list[0]
        values = line_list[1:]
        info_dict[key] = values
    return info_dict


def merage_func(x, y):
    length = max([len(line) for line in list(y.values())])
    for i in x:
        if i in y:
            y[i].extend(x[i])
        else:
            y[i].extend(['0'] * length)
            y[i].extend(x[i])
    return y


def save_info(lines, path):
    file = open(path, 'w')
    for key in lines:
        value = '\t'.join(lines[key])
        file.write(str(key) + '\t' + value + '\n')
    file.close()


def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    save_file = r'./res'
    file1_dict = read_file(file1)
    file2_dict = read_file(file2)
    info_ob = merage_func(file1_dict, file2_dict)
    save_info(info_ob, save_file)


if __name__ == '__main__':
    main()
