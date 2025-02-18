# データをチェックするプログラム

import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent

train = pd.read_csv(DIR / 'train.csv')

def get_datatype():
    for c, d in zip(train.columns, train.iloc[0]):
        print('{} : {} : {}'.format(c, type(d), d))

def count_nan():
    for col in train.columns:
        data_num = len(train[col])
        nan_num = sum(train[col].isnull())
        print('{} : {} / {}'.format(col, nan_num, data_num))


if __name__ == '__main__':
    #get_datatype()
    count_nan()
    breakpoint()
    print('ok')