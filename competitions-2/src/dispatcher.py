from sklearn import ensemble

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=250, verbose=2), # default n_estimators=10 
    "extratrees": ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=250, verbose=2) # default n_estimators=10 
}
