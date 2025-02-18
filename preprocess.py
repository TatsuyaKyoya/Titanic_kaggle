# 前処理プログラム

import pandas as pd
import numpy as np
from pathlib import Path
import re

DIR = Path(__file__).resolve().parent

def load_data(path: Path)->pd.DataFrame:
    data = pd.read_csv(path, encoding='utf-8')
    return data

def split_df(df: pd.DataFrame, target: str):
    cols = list(df.columns)
    cols.remove(target)
    return df[cols], df[target]

def drop_cols(df: pd.DataFrame, cols: list[str]|str)->pd.DataFrame:
    if isinstance(cols, str):
        cols = list(cols)
    df = df.drop(columns=cols, inplace=False)
    return df

def fillnan(df: pd.DataFrame, col_way: tuple[str, str])->pd.DataFrame:
    if col_way[1] == 'mean':
        value = df[col_way[0]].mean(numeric_only=True)
    elif col_way[1] == 'median':
        value = df[col_way[0]].mean(numeric_only=True)
    elif col_way[1] == 'mode':
        value = df[col_way[0]].mode().iloc[0]
    df[col_way[0]] = df[col_way[0]].fillna(value, inplace=False)
    return df

def concat_cols(df: pd.DataFrame, cols: list[str, str], newcol: str)->pd.DataFrame:
    df[newcol] = df.apply(lambda row: row[cols[0]] + row[cols[1]], axis=1)
    return df

def cabin_process(df: pd.DataFrame)->pd.DataFrame:
    col = 'Cabin'
    eval = (lambda x: x[0] if str(x) != 'nan' else 'N/A')
    df[col] = df[col].apply(eval)
    return df

def ticket_process(df: pd.DataFrame)->pd.DataFrame:
    col = 'Ticket'
    def convert(s: str):
        if re.search(r'\d', s[0]) != None:
            return 'num'
        elif re.search(r'[a-zA-Z]', s[0]) != None:
            return s[0]
        else:
            return 'N/A'
    df[col] = df[col].apply(convert)
    return df

def train_process(df: pd.DataFrame)->pd.DataFrame:
    drop_c = ['PassengerId', 'Name']
    fill_col = [('Age', 'median'), ('Embarked', 'mode'), ('Fare', 'median')]
    df = drop_cols(df, drop_c)
    for cw in fill_col:
        df = fillnan(df, cw)
    df = concat_cols(df, ['SibSp', 'Parch'], 'Family')
    df = cabin_process(df)
    df = ticket_process(df)
    return df

def test_process(df: pd.DataFrame)->pd.DataFrame:
    df, passid = split_df(df, 'PassengerId')
    drop_c = ['Name']
    fill_col = [('Age', 'median'), ('Embarked', 'mode'), ('Fare', 'median')]
    df = drop_cols(df, drop_c)
    for cw in fill_col:
        df = fillnan(df, cw)
    df = concat_cols(df, ['SibSp', 'Parch'], 'Family')
    df = cabin_process(df)
    df = ticket_process(df)
    return df, passid

def save_df(df: pd.DataFrame, filename: str):
    df.to_csv(DIR/filename, encoding='utf-8', index=False)

def main():
    train = load_data(DIR/'train.csv')
    test = load_data(DIR/'test.csv')
    train = train_process(train)
    test, passid = test_process(test)
    train_ind = len(train)
    df = pd.concat([train, test], axis='index')
    df = pd.get_dummies(df, dtype=int, drop_first=True)
    train, test = df.iloc[:train_ind], df.iloc[train_ind:]
    test = pd.concat([passid, test], axis='columns')
    test = test.drop(columns='Survived', inplace=False)
    save_df(train, 'p_train.csv')
    save_df(test, 'p_test.csv')

if __name__ == '__main__':
    main()
