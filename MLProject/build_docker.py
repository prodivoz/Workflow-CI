import mlflow
import os
import subprocess

EXPERIMENT_NAME = "Model ML Eksperimen"
MODEL_NAME = "model-mlflow"
ARTIFACT_NAME = "model_docker"

# Ambil ID eksperimen
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise Exception(f"Experiment '{EXPERIMENT_NAME}' tidak ditemukan.")
experiment_id = experiment.experiment_id
print(f"✔️ Found experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

# Ambil run terakhir dari eksperimen
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time desc"],
    max_results=1
)
if runs.empty:
    raise Exception(f"Tidak ada run dalam experiment '{EXPERIMENT_NAME}'.")

run_id = runs.iloc[0]["run_id"]
print(f"✔️ Latest run_id: {run_id}")

# Path ke model yang akan dibuild ke Docker
model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/{ARTIFACT_NAME}"
print(f"📦 Model URI: {model_uri}")

# Jalankan build docker
try:
    subprocess.run([
        "mlflow", "models", "build-docker",
        "-m", model_uri,
        "-n", MODEL_NAME
    ], check=True)
    print(f"✅ Docker image '{MODEL_NAME}' berhasil dibuat.")
except subprocess.CalledProcessError as e:
    print(f"❌ Gagal membuild Docker image: {e}")
    exit(1)
