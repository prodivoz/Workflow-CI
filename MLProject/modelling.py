import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.environ["MLFLOW_TRACKING_USERNAME"] = "ItsNudle"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "11fccfc93b9df7f77b755ca1718edbec8f34442e"

mlflow.set_tracking_uri("https://dagshub.com/ItsNudle/Workflow-CI.mlflow")
mlflow.set_experiment("Model ML Eskperimen")
mlflow.sklearn.autolog()

X = pd.read_csv("Membangun_Model/spam_ham_emails_preprocessing/tfidf.csv")
y = pd.read_csv("Membangun_Model/spam_ham_emails_preprocessing/labels.csv")["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print("Akurasi:", acc)
