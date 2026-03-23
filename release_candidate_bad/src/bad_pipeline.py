from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # FIX 1: was missing, needed by tests + imputation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# FIX 2: was Path("data/raw/customer_churn.csv") — wrong dataset name
APPROVED_DATASET_PATH = Path("data/raw/titanic_train.csv")

# FIX 3: removed hardcoded LOCAL_FALLBACK_PATH with absolute C:/Users/... path
#         now falls back safely to the approved dataset path
_DEFAULT_PATH = str(APPROVED_DATASET_PATH)


# FIX 4: return type was `int`, should be `pd.DataFrame`
def load_dataset(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return dataframe


# FIX 5: was using mean imputation — tests require LinearRegression imputation
def impute_age_with_regression(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    feature_cols = ["Pclass", "SibSp", "Parch", "Fare"]

    known = dataframe[dataframe["Age"].notna()]
    unknown = dataframe[dataframe["Age"].isna()]

    if unknown.empty:
        return dataframe

    model = LinearRegression()
    model.fit(known[feature_cols], known["Age"])
    predicted_ages = model.predict(unknown[feature_cols])

    dataframe.loc[dataframe["Age"].isna(), "Age"] = predicted_ages
    return dataframe


def train_model(
    dataframe: pd.DataFrame,
) -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    features = dataframe[["Pclass", "SibSp", "Parch", "Fare"]]
    target = dataframe["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    model = LogisticRegression(max_iter=200)  # FIX 6: 100 was too low, caused convergence warning
    model.fit(x_train, y_train)
    return model, x_test, y_test


def main() -> None:
    np.random.seed(42)
    # FIX 7: removed hardcoded absolute fallback path — uses approved dataset path instead
    dataset_path = os.getenv("DATASET_PATH", _DEFAULT_PATH)
    dataframe = load_dataset(dataset_path)
    imputed_frame = impute_age_with_regression(dataframe)
    model, x_test, y_test = train_model(imputed_frame)
    predictions = model.predict(x_test)
    print("rows", len(predictions))


if __name__ == "__main__":
    main()
