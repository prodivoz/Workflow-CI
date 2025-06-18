import mlflow
import subprocess

# Ambil ID eksperimen berdasarkan nama
experiment = mlflow.get_experiment_by_name("Model ML Eksperimen")
if experiment is None:
    raise ValueError("Experiment 'Model ML Eksperimen' tidak ditemukan.")
experiment_id = experiment.experiment_id

# Ambil run_id terakhir dari eksperimen
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
if runs.empty:
    raise ValueError("Tidak ada run ditemukan di experiment ini.")
run_id = runs.iloc[0]["run_id"]
print(f"✔️ Found latest run_id: {run_id}")

# Path model yang akan dibuild ke Docker
model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/model_docker"
image_name = "model-mlflow"

# Jalankan perintah build docker dari Python
try:
    subprocess.run([
        "mlflow", "models", "build-docker",
        "-m", model_uri,
        "-n", image_name
    ], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Gagal membuild Docker image: {e}")
    raise
