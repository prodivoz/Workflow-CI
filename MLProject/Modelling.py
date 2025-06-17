import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Modelling Eksperimen")
mlflow.sklearn.autolog()

X = pd.read_csv("spam_ham_emails_preprocessing/tfidf.csv")
y = pd.read_csv("spam_ham_emails_preprocessing/labels.csv")["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("Akurasi:", acc)

    joblib.dump(model, "model.pkl")
