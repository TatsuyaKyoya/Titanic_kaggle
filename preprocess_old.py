# 前処理用のプログラム

from pathlib import Path
import pandas as pd
import numpy as np

DIR = Path(__file__).resolve().parent

def delete_cols(df: pd.DataFrame, col: str|list[str])->pd.DataFrame:
    df = df.drop(columns=col, inplace=False)
    return df

def fill_nan(df: pd.DataFrame, col: str, way: str)->int|float|str:
    if way == 'mean':
        assert not isinstance(df[col].iloc[0], str), "this col's data must be numeric." 
        value = df[col].mean(numeric_only=True)
    elif way == 'median':
        assert not isinstance(df[col].iloc[0], str), "this col's data must be numeric." 
        value = df[col].median(numeric_only=True)
    elif way == 'mode':
        value = df[col].mode().iloc[0]
    df[col] = df[col].fillna(value, inplace=False)
    return df

def count_nan(df: pd.DataFrame):
    for col in df.columns:
        data_num = len(df[col])
        nan_num = sum(df[col].isnull())
        print('{} : {} / {}'.format(col, nan_num, data_num))

def main():
    train_data = pd.read_csv(DIR /'train.csv')
    drop_c = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = delete_cols(train_data, drop_c)
    fill_col = [('Age', 'median'), ('Embarked', 'mode')]
    for c, w in fill_col:
        df = fill_nan(df, c, w)
    breakpoint()
    df = pd.get_dummies(df, dtype=int, drop_first=True)
    count_nan(df)
    save_path =( DIR / 'p_train.csv')
    df.to_csv(save_path, encoding='utf-8', index=False)
    print('ok')

if __name__ == '__main__':
    main()