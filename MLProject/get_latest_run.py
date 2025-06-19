import mlflow

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

# Cetak run_id agar bisa ditangkap oleh GitHub Actions
print(f"run_id: {run_id}")
