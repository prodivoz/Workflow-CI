FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y wget git gcc python3-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

ENV PATH="/opt/conda/bin:$PATH"

RUN conda env create -f conda.yaml && \
    conda env list

RUN apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Use correct environment name (mlflow-env instead of mlflow_env)
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "mlflow-env", "mlflow", "run", "."]
