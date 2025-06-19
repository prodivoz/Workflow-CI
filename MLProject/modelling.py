import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

mlflow.autolog()

INPUT_PATH = "games_preprocessed/games_preprocessed.csv"

def load_data(path):
    df = pd.read_csv(path)
    df["price_class"] = pd.qcut(df["price"], q=3, labels=["low", "medium", "high"])
    df = df.drop(columns=["price"])
    return df

def preprocess(df):
    X = df.drop(columns=["price_class"])
    y = df["price_class"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )

    X_final = pd.concat([X[num_cols], X_cat], axis=1)
    return X_final, y

if __name__ == "__main__":
    with mlflow.start_run() as run:
        df = load_data(INPUT_PATH)
        X, y = preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")
        print(f"✅ MLFLOW_RUN_ID={run.info.run_id}")
