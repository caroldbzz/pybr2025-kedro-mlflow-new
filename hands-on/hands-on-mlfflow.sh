# Parte 1: Install MLflow
conda create -n <env_name> python=3.13 -y
conda activate <env_name>
pip install uv
uv pip install mlflow
uv pip install jupyter
mlflow server --host 127.0.0.1 --port 8080

# Parte 3: Run all in the notebook
jupyter notebook hands-on/mlflow-case.ipynb

# Parte 4: Check again the UI