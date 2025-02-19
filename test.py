# submit用のデータを予測させる

import pandas as pd
import pickle
from pathlib import Path
import preprocess_old as pp

def split_df(df: pd.DataFrame, target: str, to_numpy: bool=False)->tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)
    non_target = list(set(cols) - set(target))
    if to_numpy:
        return df[non_target].to_numpy(), df[target].to_numpy()
    return df[non_target], df[target]

def main():
    dir = Path(__file__).resolve().parent
    test_path = dir / 'p_test.csv'
    df = pd.read_csv(test_path)
    data, passid = split_df(df, 'PassengerId')
    modelfile = 'models/2025021937_macc1.0_RF.pkl'
    modelpath = dir/modelfile
    model = pickle.load(open(modelpath, 'rb'))
    pred = model.predict(data.to_numpy())
    pred = [int(p) for p in pred]
    submit_df = pd.DataFrame({'PassengerId': passid, 'Survived': pred})
    submit_file = dir / 'submission.csv'
    submit_df.to_csv(submit_file, encoding='utf-8', header=True, index=False)

if __name__ == '__main__':
    main()
    