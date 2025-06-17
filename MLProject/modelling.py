import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.environ["MLFLOW_TRACKING_USERNAME"] = "ItsNudle"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "28a2bed8301cd660e33707a009cb925162d47426"

dagshub.init(repo_owner='ItsNudle', repo_name='Workflow-CI', mlflow=True)
mlflow.set_experiment("Model ML - Workflow CI - Eksperimen")
mlflow.sklearn.autolog()

X = pd.read_csv("MLProject/spam_ham_emails_preprocessing/tfidf.csv")
y = pd.read_csv("MLProject/spam_ham_emails_preprocessing/labels.csv")["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print("Akurasi:", acc)