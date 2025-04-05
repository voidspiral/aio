
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from sklearn import tree
import joblib
from feature import *
import sys
import warnings

warnings.filterwarnings("ignore")


def huber_approx_obj(y_pred, y_test):
    """
    Huber loss, adapted from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    """
    d = y_pred - y_test
    h = 5  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def load_datasets(fs_type, datname):
    df = pd.read_csv(datname)

    features = log_features + perc_features+ romio_features
    if fs_type == "GekkoFS":
        features = features + gkfs_features
    elif fs_type == "Lustre":
        features = features + lustre_features

    df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.15)

    X_train, X_test = df_train[features], df_test[features]

    y_train, y_test = df_train["agg_perf_by_slowest_LOG10"], df_test["agg_perf_by_slowest_LOG10"]

    return X_train, X_test, y_train, y_test


def model_train(fs_type, X_train, X_test, y_train, y_test):
    regressor = xgb.XGBRegressor(obj=huber_approx_obj, n_estimators=64, max_depth=128, colsample_bytree=0.8)
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    error = np.median(10 ** (np.abs(y_test - y_pred_test))-1)
    print(error)
    if fs_type == "GekkoFS":
        joblib.dump(regressor, './models/sgModel.pkl')
    elif fs_type == "Lustre":
        joblib.dump(regressor, './models/slModel.pkl')


def main():
    datafile = "data.csv"
    filesystem = ["Lustre", "GekkoFS"]
    for fs_type in filesystem:
        X_train, X_test, y_train, y_test = load_datasets(fs_type, datafile)
        model_train(fs_type, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

