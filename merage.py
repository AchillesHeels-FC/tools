# coding = utf-8
import os
import sys
import pandas as pd


def read_file(path):
    file_type = os.path.splitext(path)[-1]
    if file_type == '.csv':
        df = pd.read_csv(path, sep=',', header=0)
    else:
        df = pd.read_table(path, sep=',', header=0)
    return df


def merage_func(x, y, col_name):
    df_x = x.copy()
    df_y = y.copy()
    df = pd.merge(df_x, df_y, how='outer', on=[col_name])
    return df


def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    colnames = sys.argv[3]
    save_file = sys.argv[4]
    file1_df = read_file(file1)
    file2_df = read_file(file2)
    res_df = merage_func(file1_df, file2_df, colnames)
    res_df.to_csv(save_file)


if __name__ == '__main__':
    main()
