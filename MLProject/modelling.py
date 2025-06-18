import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Load data
mlflow.set_experiment("Model ML Eksperimen")
df = pd.read_csv("games_preprocessed/games_preprocessed.csv")
df['price_class'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])

X = df.drop(columns=['price', 'price_class'])
y = df['price_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cat_cols = X_train.select_dtypes(include='object').columns
num_cols = X_train.select_dtypes(exclude='object').columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

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
    fig_path = "figures/confusion_matrix.png"
    ConfusionMatrixDisplay(cm, display_labels=['low', 'medium', 'high']).plot(cmap='Blues')
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

    joblib.dump(model, "model.pkl")
    mlflow.pyfunc.log_model(
        artifact_path="model_docker",
        python_model=SklearnWrapper(),
        artifacts={"model_path": "model.pkl"},
        input_example=X_test.iloc[:1],
        signature=mlflow.models.infer_signature(X_test, y_pred)
    )
    print("Run ID:", run.info.run_id)
