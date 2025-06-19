import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ========== MLflow Tracking Setup ==========
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Model ML Eksperimen")

# ========== PyFunc Wrapper ==========
class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        return self.model.predict(model_input)

# ========== Main Function ==========
def train_and_log():
    print("📥 Loading data...")
    df = pd.read_csv("games_preprocessed/games_preprocessed.csv")
    df["price_class"] = pd.qcut(df["price"], q=3, labels=["low", "medium", "high"])

    X = df.drop(columns=["price", "price_class"])
    y = df["price_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing
    cat_cols = X_train.select_dtypes(include="object").columns
    num_cols = X_train.select_dtypes(exclude="object").columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = pd.DataFrame(
        encoder.fit_transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )
    X_test_cat = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )

    X_train = pd.concat([X_train[num_cols], X_train_cat], axis=1)
    X_test = pd.concat([X_test[num_cols], X_test_cat], axis=1)

    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Gunakan run aktif dari mlflow CLI
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("⚠️ No active MLflow run. This script should be executed via `mlflow run`.")

    print(f"🚀 Active MLflow Run ID: {run.info.run_id}")

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"])

    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    })

    # Simpan dan log confusion matrix
    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/confusion_matrix.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "medium", "high"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

    # Simpan model ke file dan log
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Log model sebagai PyFunc
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SklearnWrapper(),
        artifacts={"model_path": model_path},
        input_example=X_test.iloc[:1],
        signature=mlflow.models.infer_signature(X_test, y_pred)
    )

    print("✅ Metrics logged")
    print("📦 Model saved to:", mlflow.get_artifact_uri("model"))

    # Debug: list isi folder
    artifact_uri = mlflow.get_artifact_uri("model")
    if artifact_uri.startswith("file://"):
        print("📂 Contents of model artifact folder:")
        os.system("ls -R " + artifact_uri.replace("file://", ""))

if __name__ == "__main__":
    train_and_log()
