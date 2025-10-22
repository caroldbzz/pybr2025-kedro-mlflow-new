# Dockerfile

FROM python:3.13-slim

WORKDIR /home/kedro

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code
COPY . .

# Set default Kedro environment
ENV KEDRO_ENV=local

# Set environment variable for the MLflow tracking URI inside the container
ENV MLFLOW_TRACKING_URI=file:/home/kedro/data/mlruns

CMD ["kedro", "run"]