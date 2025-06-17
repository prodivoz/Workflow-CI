import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from scipy.stats import uniform
from mlflow.models.signature import infer_signature

os.environ["MLFLOW_TRACKING_USERNAME"] = "ItsNudle"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "11fccfc93b9df7f77b755ca1718edbec8f34442e"

mlflow.set_tracking_uri("https://dagshub.com/ItsNudle/Workflow-CI.mlflow")
mlflow.set_experiment("Modelling dan Tuning Eksperimen")

X = pd.read_csv("MLProject/spam_ham_emails_preprocessing/tfidf.csv")
y = pd.read_csv("MLProject/spam_ham_emails_preprocessing/labels.csv")["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
param_dist = {
    'C': uniform(loc=0.01, scale=0.1),
    'solver': ['liblinear', 'lbfgs']
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=4,
    cv=3,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

y_train_pred = search.predict(X_train)
y_test_pred = search.predict(X_test)

training_acc = accuracy_score(y_train, y_train_pred)
training_prec = precision_score(y_train, y_train_pred)
training_rec = recall_score(y_train, y_train_pred)
training_f1 = f1_score(y_train, y_train_pred)
training_log_loss_value = log_loss(y_train, search.predict_proba(X_train))
training_roc_auc = roc_auc_score(y_train, search.predict_proba(X_train)[:, 1])
training_score = search.best_estimator_.score(X_train, y_train)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
log_loss_value = log_loss(y_test, search.predict_proba(X_test))
roc_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
testing_score = search.best_estimator_.score(X_test, y_test)

with mlflow.start_run():
    mlflow.log_param("best_C", search.best_params_['C'])
    mlflow.log_param("best_solver", search.best_params_['solver'])

    mlflow.log_metric("training_accuracy_score", training_acc)
    mlflow.log_metric("training_precision_score", training_prec)
    mlflow.log_metric("training_recall_score", training_rec)
    mlflow.log_metric("training_f1_score", training_f1)
    mlflow.log_metric("training_log_loss", training_log_loss_value)
    mlflow.log_metric("training_roc_auc", training_roc_auc)
    mlflow.log_metric("training_score", training_score)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("log_loss", log_loss_value)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("testing_score", testing_score)

    joblib.dump(search.best_estimator_, "best_model.pkl")

    signature = infer_signature(X_train, search.predict(X_train))
    input_example = X_train.iloc[:1]

    mlflow.sklearn.log_model(
        sk_model=search.best_estimator_,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

print("✅ Model berhasil dituning dan dicatat di DagsHub MLflow.")
