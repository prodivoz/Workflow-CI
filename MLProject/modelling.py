import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Aktifkan autolog
mlflow.autolog()

# Lokasi file input
INPUT_PATH = "games_preprocessed/games_preprocessed.csv"

def load_data(path):
    df = pd.read_csv(path)
    # Preprocessing label
    df["price_class"] = pd.qcut(df["price"], q=3, labels=["low", "medium", "high"])
    df = df.drop(columns=["price"])  # Hapus kolom target lama
    return df

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["price_class"], random_state=42)

    train_dataset = mlflow.data.from_pandas(train_df, name="train")
    test_dataset = mlflow.data.from_pandas(test_df, name="test")

    x_train = train_dataset.df.drop(columns=["price_class"])
    y_train = train_dataset.df["price_class"]

    x_test = test_dataset.df.drop(columns=["price_class"])
    y_test = test_dataset.df["price_class"]

    return x_train, y_train, x_test, y_test

def train_model(x_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)
    return model

if __name__ == "__main__":
    with mlflow.start_run() as run:
        df = load_data(INPUT_PATH)
        x_train, y_train, x_test, y_test = split_data(df)
        model = train_model(x_train, y_train)

        # Log model secara eksplisit (meskipun autolog akan otomatis juga)
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ MLFLOW_RUN_ID={run.info.run_id}")
