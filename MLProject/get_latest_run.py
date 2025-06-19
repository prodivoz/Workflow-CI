import mlflow

# Ambil experiment dan run terakhir
experiment = mlflow.get_experiment_by_name("Model ML Eksperimen")
if experiment is None:
    raise ValueError("Experiment tidak ditemukan.")

experiment_id = experiment.experiment_id
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
if runs.empty:
    raise ValueError("Tidak ada run ditemukan.")

run_id = runs.iloc[0]["run_id"]
print(f"experiment_id: {experiment_id}")
print(f"run_id: {run_id}")
