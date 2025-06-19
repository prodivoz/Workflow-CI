# Gunakan Python image ringan
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Salin seluruh isi artifact model ke dalam image
COPY model_artifact /app/model

# Install MLflow dan dependensi model
RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn pandas numpy

# Expose port API default MLflow
EXPOSE 5000

# Jalankan MLflow model sebagai REST API
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5000", "--no-conda"]
