import sys
from typing import Any

import lightgbm
import xgboost
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(model_name: str) -> Any:
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    if model_name == "xgboost":
        model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, eval_metric="mlogloss", random_state=42)
    elif model_name == "lightgbm":
        model = lightgbm.LGBMClassifier(n_estimators=5, max_depth=2, random_state=42)
    elif model_name == "sklearn":
        model = LogisticRegression(random_state=42, max_iter=10)
    else:
        raise ValueError("model_name error")
    model.fit(X_train, y_train)
    return model


def predict_result(model) -> Any:
    dataset = load_iris()
    _, X_test, _, _ = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    return y_pred


if __name__ == "__main__":
    result = train_model(sys.argv[1])
    __return__ = result
