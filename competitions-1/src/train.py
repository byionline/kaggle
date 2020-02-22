# import library
import os
import pandas as pd 
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from . import dispatcher

#TRAINING_DATA = None
TRAINING_DATA = os.environ.get("TRAINING_DATA")
#FOLD = None
"""
TypeError: only list-like objects are allowed to be passed to isin(), you passed a [NoneType]
that's why change to int
"""
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}
# main
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfolds.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfolds==FOLD]
    # setting trainig and validation target 
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    # drop columns 
    train_df = train_df.drop(["id", "target", "kfolds"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfolds"], axis=1)
    # order of variable is same
    valid_df = valid_df[train_df.columns]

    #encode the variables
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))
    
    # now train
    clf = dispatcher.MODELS[MODEL] # GETTING MODEL FROM `dispatcher` via environment variable:'MODEL'
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]  # probability : proba
    #print(preds)

    # calculate AOC 
    print(metrics.roc_auc_score(yvalid, preds))

"""
Note:
Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
this implementation can be used with binary, multiclass and multilabel classification, 
but some restrictions apply (see Parameters).
 
 sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)[source]¶

"""

