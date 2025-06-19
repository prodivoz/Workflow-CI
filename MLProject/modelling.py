import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Set up MLflow tracking
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username     = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password     = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Model ML Eksperimen")
#mlflow.sklearn.autolog()

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def main():
    df = pd.read_csv("games_preprocessed/games_preprocessed.csv")
    df['price_class'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])

    X = df.drop(columns=['price', 'price_class'])
    y = df['price_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cat_cols = X_train.select_dtypes(include='object').columns
    num_cols = X_train.select_dtypes(exclude='object').columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[cat_cols])
    X_test_encoded = encoder.transform(X_test[cat_cols])

    encoded_feature_names = encoder.get_feature_names_out(cat_cols)
    X_train_cat = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
    X_test_cat = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

    X_train = pd.concat([X_train[num_cols], X_train_cat], axis=1)
    X_test = pd.concat([X_test[num_cols], X_test_cat], axis=1)

    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    with mlflow.start_run() as run:
        params = {"n_estimators": 150, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted})

        os.makedirs("figures", exist_ok=True)
        fig_path = "figures/confusion_matrix_rf.png"
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low', 'medium', 'high'])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Random Forest")
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        model_path = "model.pkl"
        joblib.dump(model, model_path)

        pyfunc_path = "model_pyfunc"
        mlflow.pyfunc.save_model(
            path=pyfunc_path,
            python_model=SklearnWrapper(),
            artifacts={"model_path": model_path},
            input_example=X_test.iloc[:1],
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )

        mlflow.log_artifacts(pyfunc_path, artifact_path="model")

        print("✅ Run ID:", run.info.run_id)
        print(f"✅ Accuracy: {acc:.2f}")
        print(f"✅ F1 Macro: {f1_macro:.2f}")
        print(f"✅ F1 Weighted: {f1_weighted:.2f}")
        print("✅ Model logged at:", mlflow.get_artifact_uri("model"))

if __name__ == "__main__":
    main()
