# import library
import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    # read dataset
    df = pd.read_csv("input/train.csv")
    # make fake column
    df["kfolds"] = -1

    # shuffle the data and drop the index
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=43)

    # train index: train_idx, test_index: val_idx
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfolds'] = fold
    
    # save the new dataset
    df.to_csv("input/train_kfolds.csv", index=False)

""" 
Note:
Stratified K-Folds cross-validator:

[kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=43)]

Provides train/test indices to split data in train/test sets.
This cross-validation object is a variation of KFold that returns stratified folds. 
The folds are made by preserving the percentage of samples for each class.
"""