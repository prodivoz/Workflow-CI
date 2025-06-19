import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("games_preprocessed.csv")

# Buat kolom klasifikasi target (3 kelas)
df['price_class'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])

X = df.drop(columns=['price', 'price_class'])
y = df['price_class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Encode fitur kategorikal
cat_cols = X_train.select_dtypes(include='object').columns
num_cols = X_train.select_dtypes(exclude='object').columns

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

encoded_feature_names = encoder.get_feature_names_out(cat_cols)
X_train_cat = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)
X_test_cat = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

# Gabungkan kembali
X_train = pd.concat([X_train[num_cols], X_train_cat], axis=1)
X_test = pd.concat([X_test[num_cols], X_test_cat], axis=1)

# Samakan kolom
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# Tracking ke MLflow
mlflow.set_experiment("RF_GamesPriceClassification")

with mlflow.start_run():
    params = {
        'n_estimators': 150,
        'max_depth': 10,
        'random_state': 42
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])

    # Logging manual
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("f1_weighted", f1_weighted)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low', 'medium', 'high'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")

    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/confusion_matrix_rf.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

    # Save model
    mlflow.sklearn.log_model(model, "rf_classifier")

    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Macro: {f1_macro:.2f}")
    print(f"F1 Weighted: {f1_weighted:.2f}")
