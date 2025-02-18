# モデルの訓練を行う

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from datetime import datetime
import optuna
import json

def split_df(df: pd.DataFrame, target: str, to_numpy: bool)->tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)
    non_target = list(set(cols) - set(target))
    if to_numpy:
        return df[non_target].to_numpy(), df[target].to_numpy()
    return df[non_target], df[target]

def cross_validation(model, data, target):
    sfkf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, data, target, cv=sfkf, scoring='roc_auc')
    print('cross validation scores: {}'.format(scores))
    print('average score: {}'.format(np.mean(scores)))
    return np.mean(scores)

def simple_validation(model, data, target):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=0, shuffle=True)
    model = model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    print('test score: {}'.format(score))

def simple_training(model, data, target):
    model.fit(data,target)
    return model

class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __call__(self, trial: optuna.Trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, 10),
            'max_depth': trial.suggest_int('max_depth', 2, 10, 1),
            'random_state': trial.suggest_int('random_state', 0, 10, 2)
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, self.X, self.y, scoring='roc_auc', cv=cv)
        return np.mean(scores)

def hypara_tuning(data, target):
    study = optuna.create_study(direction='maximize')
    objective = Objective(data, target)
    study.optimize(objective)
    return study.best_params, study.best_value

def save_model(model, score, model_name, params):
    score = str(round(score, 1))
    dir = data_path = Path(__file__).resolve().parent / 'models'
    para_dir = dir.parent / 'params'
    if not dir.exists():
        dir.mkdir(parents=True)
    if not para_dir.exists():
        para_dir.mkdir(parents=True)
    today = datetime.now().strftime('%Y%m%d%M')
    filename = dir / ('{}_macc{}_{}.pkl'.format(today, score, model_name))
    pickle.dump(model, open(filename, 'wb'))
    with open(para_dir/'{}_macc{}_{}.json'.format(today, score, model_name),'w', encoding='utf-8' ) as f:
        json.dump(params, f, indent=2)
    pass

def main():
    dir = Path(__file__).resolve().parent
    data_path = dir / 'p_train.csv'
    df = pd.read_csv(data_path, encoding='utf-8')
    data, target = split_df(df, 'Survived', True)
    #model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #model = RandomForestClassifier(random_state=0)
    #ave_score = cross_validation(model, data, target)
    #simple_validation(model, data, target)
    best_params, best_value = hypara_tuning(data=data, target=target)
    model = RandomForestClassifier(**best_params)
    model = simple_training(model, data, target)
    save_model(model, best_value, 'RF', best_params)

if __name__ == '__main__':
    main()