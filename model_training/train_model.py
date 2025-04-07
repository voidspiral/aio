import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from sklearn import tree
import joblib
from feature import *
import sys
import warnings
import os

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


def load_datasets(fs_type, datafile):
    """加载数据集并准备训练"""
    # 读取数据
    df = pd.read_csv(datafile)

    # 检查并处理非数值列
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Warning: Found non-numeric columns: {non_numeric_cols.tolist()}")
        print("Attempting to convert to numeric...")

        for col in non_numeric_cols:
            try:
                # 尝试转换为数值
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Successfully converted {col} to numeric")
            except:
                print(f"Could not convert {col} to numeric, filling with 0")
                raise


    df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.15)


    features = log_features + perc_features + romio_features
    if fs_type == "GekkoFS":
        features = features + gkfs_features
    elif fs_type == "Lustre":
        features = features + lustre_features


    X_train, X_test = df_train[features], df_test[features]
    y_train, y_test = df_train["agg_perf_by_slowest_LOG10"], df_test["agg_perf_by_slowest_LOG10"]

    return X_train, X_test, y_train, y_test


def model_train(fs_type, X_train, X_test, y_train, y_test):
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 models 目录的绝对路径
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    regressor = xgb.XGBRegressor(obj=huber_approx_obj, n_estimators=64, max_depth=128, colsample_bytree=0.8)
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    error = np.median(10 ** (np.abs(y_test - y_pred_test)) - 1)
    print(error)
    if fs_type == "GekkoFS":
        model_path = os.path.join(models_dir, "sgModel.pkl")
        joblib.dump(regressor, model_path)
        print(f"Model saved to {model_path}")
    elif fs_type == "Lustre":
        model_path = os.path.join(models_dir, "slModel.pkl")
        joblib.dump(regressor, model_path)
        print(f"Model saved to {model_path}")


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    root_dir = os.path.dirname(current_dir)

    collectors_dir = os.path.join(root_dir, "collectors")

    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    filesystem = ["Lustre", "GekkoFS"]
    for fs_type in filesystem:
        # 使用绝对路径
        datafile = os.path.join(collectors_dir, f"{fs_type.lower()}.csv")
        print(f"Training model for {fs_type} using {datafile}")
        X_train, X_test, y_train, y_test = load_datasets(fs_type, datafile)
        model_train(fs_type, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

