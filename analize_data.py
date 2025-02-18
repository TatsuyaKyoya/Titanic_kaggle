# データを観察するプログラム

import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import math

DIR = Path(__file__).resolve().parent

def unique_count(df: pd.DataFrame, col: str):
    uniques = df[col].unique()
    for u in uniques:
        survived = len(df[(df[col] == u) & (df['Survived'] == 1)])
        non_survived = len(df[(df[col] == u) & (df['Survived'] == 0)])
        print(f'{u}: survived {survived} dead {non_survived}')
    pass

def meta_count(df: pd.DataFrame, uniques: dict, col: str):
    for k, v in uniques.items():
        survived = len(df[(df[col].isin(v)) & (df['Survived'] == 1)])
        non_survived = len(df[(df[col].isin(v)) & (df['Survived'] == 0)])
        print(f'{k}: survived {survived} dead {non_survived}')
    pass

def cabinet_meta_count(df: pd.DataFrame):
    col = 'Cabin'
    uniques = df[col].unique()
    results = defaultdict(list)
    for u in uniques:
        if str(u) == 'nan':
            results['nan'].append(u)
        else:
            results[u[0]].append(u)
    meta_count(df, results, col)
    pass

def ticket_meta_count(df: pd.DataFrame):
    col = 'Ticket'
    uniques = df[col].unique()
    results = defaultdict(list)
    for u in uniques:
        if re.search(r'\d', u[0]) != None:
            results['num'].append(u)
        elif re.search(r'[a-zA-Z]', u[0]) != None:
            results[u[0]].append(u)
        else:
            results['others'].append(u)
    meta_count(df, results, col)
    pass

def main():
    path = DIR / 'train.csv'
    data = pd.read_csv(path)
    print('cabinet meta count')
    cabinet_meta_count(data)
    print('\nticket meta count')
    ticket_meta_count(data)
    breakpoint()
    print('ok')

if __name__ == '__main__':
    main()
